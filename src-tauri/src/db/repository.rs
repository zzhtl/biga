//! 数据仓库层
//! 
//! 提供数据访问接口，封装所有 SQL 操作

use crate::config::constants::BATCH_SIZE;
use crate::db::models::*;
use crate::error::AppError;
use crate::utils::canonical_stock_symbol;
use crate::utils::volume_metrics::{
    calculate_turnover_rate, calculate_volume_ratio_series, DEFAULT_VOLUME_RATIO_PERIOD,
};
use sqlx::{QueryBuilder, sqlite::SqlitePool};
use std::collections::BTreeMap;

const VALID_HISTORICAL_BAR_FILTER: &str = "open > 0 AND close > 0 AND high > 0 AND low > 0 AND high >= low AND high >= open AND high >= close AND low <= open AND low <= close";

fn historical_symbol_variants(symbol: &str) -> Vec<String> {
    let trimmed = symbol.trim();
    let upper = trimmed.to_ascii_uppercase();
    let digits: String = upper.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut variants = Vec::new();

    for candidate in [
        trimmed.to_string(),
        upper.clone(),
        upper.to_ascii_lowercase(),
        digits.clone(),
    ] {
        if !candidate.is_empty() && !variants.contains(&candidate) {
            variants.push(candidate);
        }
    }

    if digits.len() == 6 {
        for suffix in ["SZ", "SH"] {
            let candidate = format!("{digits}.{suffix}");
            if !variants.contains(&candidate) {
                variants.push(candidate);
            }
        }
    }

    variants
}

/// 解析历史数据表中真实存在且最新有效的 symbol。
pub async fn resolve_historical_symbol(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<Option<String>, AppError> {
    let variants = historical_symbol_variants(symbol);
    if variants.is_empty() {
        return Ok(None);
    }

    let mut query_builder = QueryBuilder::new(
        "SELECT symbol FROM historical_data WHERE ",
    );
    query_builder.push(VALID_HISTORICAL_BAR_FILTER);
    query_builder.push(" AND symbol IN (");
    let mut separated = query_builder.separated(", ");
    for variant in &variants {
        separated.push_bind(variant);
    }
    separated.push_unseparated(")");
    query_builder.push(" GROUP BY symbol ORDER BY MAX(date) DESC, COUNT(*) DESC LIMIT 1");

    let row: Option<(String,)> = query_builder
        .build_query_as()
        .fetch_optional(pool)
        .await?;

    Ok(row.map(|(symbol,)| symbol))
}

// =============================================================================
// 股票信息仓库
// =============================================================================

/// 批量插入股票基本信息
pub async fn batch_insert_stock_info(
    pool: &SqlitePool,
    data_list: Vec<StockInfo>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }
    
    let mut tx = pool.begin().await?;
    let mut affected_rows = 0;
    
    for chunk in data_list.chunks(BATCH_SIZE) {
        let mut query_builder =
            QueryBuilder::new("INSERT INTO stock_info (symbol, name, exchange) ");
        query_builder.push_values(chunk, |mut b, data| {
            let symbol = canonical_stock_symbol(&data.symbol);
            let name = if canonical_stock_symbol(&data.name) == symbol {
                symbol.clone()
            } else {
                data.name.trim().to_string()
            };
            b.push_bind(symbol)
                .push_bind(name)
                .push_bind(data.exchange.trim().to_ascii_lowercase());
        });
        query_builder.push(
            " ON CONFLICT(symbol) DO UPDATE SET
                name = CASE
                    WHEN EXCLUDED.name <> EXCLUDED.symbol THEN EXCLUDED.name
                    ELSE stock_info.name
                END,
                exchange = CASE
                    WHEN EXCLUDED.exchange <> '' THEN EXCLUDED.exchange
                    ELSE stock_info.exchange
                END",
        );
        let result = query_builder.build().execute(&mut *tx).await?;
        affected_rows += result.rows_affected();
    }
    
    tx.commit().await?;
    Ok(affected_rows)
}

/// 批量插入股票详细信息
pub async fn batch_insert_stock(
    pool: &SqlitePool,
    data_list: Vec<Stock>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }
    
    let mut tx = pool.begin().await?;
    let mut affected_rows = 0;
    
    for chunk in data_list.chunks(BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO stock (symbol, name, area, industry, market, exchange, list_date, act_name, act_ent_type) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            let exchange = data.exchange.split('.').next_back().unwrap_or("").to_lowercase();
            b.push_bind(canonical_stock_symbol(&data.symbol))
                .push_bind(&data.name)
                .push_bind(&data.area)
                .push_bind(&data.industry)
                .push_bind(&data.market)
                .push_bind(exchange)
                .push_bind(&data.list_date)
                .push_bind(&data.act_name)
                .push_bind(&data.act_ent_type);
        });
        query_builder.push(" ON CONFLICT(symbol) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        affected_rows += result.rows_affected();
    }
    
    tx.commit().await?;
    Ok(affected_rows)
}

/// 通过 symbol 获取单个股票信息
pub async fn get_stock_info(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<StockInfo, AppError> {
    let record: Option<StockInfo> = sqlx::query_as(
        r#"
        SELECT
            COALESCE(symbol, '') as symbol,
            COALESCE(name, '') as name,
            COALESCE(exchange, '') as exchange
        FROM stock_info
        WHERE symbol = ?
        "#,
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;

    match record {
        Some(info) => Ok(info),
        None => Err(AppError::SqlxError(sqlx::Error::RowNotFound)),
    }
}

// =============================================================================
// 历史数据仓库
// =============================================================================

/// 批量插入历史数据
pub async fn batch_insert_historical_data(
    symbol: &str,
    pool: &SqlitePool,
    data_list: Vec<HistoricalData>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }

    let symbol = canonical_stock_symbol(symbol);
    let stock_info = get_stock_info(&symbol, pool).await?;
    let mut tx = pool.begin().await?;
    let mut batch_size: u64 = 0;
    
    for chunk in data_list.chunks(BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO historical_data (symbol, date, open, close, high, low, volume,
            amount, amplitude, turnover_rate, volume_ratio, change, change_percent) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&symbol)
                .push_bind(data.date)
                .push_bind(data.open)
                .push_bind(data.close)
                .push_bind(data.high)
                .push_bind(data.low)
                .push_bind(data.volume)
                .push_bind(data.amount)
                .push_bind(data.amplitude)
                .push_bind(data.turnover_rate)
                .push_bind(data.volume_ratio)
                .push_bind(data.change)
                .push_bind(data.change_percent);
        });

        query_builder.push(" ON CONFLICT(symbol, date) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }
    
    // 更新实时数据
    if let Some(last_history) = data_list.last() {
        let mut realtime_builder = QueryBuilder::new(
            "INSERT INTO realtime_data (symbol, name, date, close, volume, amount, amplitude, turnover_rate, volume_ratio, change, change_percent) ",
        );
        realtime_builder.push_values(&[last_history], |mut b, data| {
            b.push_bind(&symbol)
                .push_bind(&stock_info.name)
                .push_bind(data.date)
                .push_bind(data.close)
                .push_bind(data.volume)
                .push_bind(data.amount)
                .push_bind(data.amplitude)
                .push_bind(data.turnover_rate)
                .push_bind(data.volume_ratio)
                .push_bind(data.change)
                .push_bind(data.change_percent);
        });

        realtime_builder.push(
            r#" ON CONFLICT(symbol) DO UPDATE SET
                name = EXCLUDED.name,
                date = EXCLUDED.date,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                amount = EXCLUDED.amount,
                amplitude = EXCLUDED.amplitude,
                turnover_rate = EXCLUDED.turnover_rate,
                volume_ratio = EXCLUDED.volume_ratio,
                change = EXCLUDED.change,
                change_percent = EXCLUDED.change_percent
            "#,
        );
        let result = realtime_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }

    tx.commit().await?;
    Ok(batch_size)
}

/// 查询历史数据
pub async fn get_historical_data(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    pool: &SqlitePool,
) -> Result<Vec<HistoricalData>, AppError> {
    let actual_symbol = resolve_historical_symbol(symbol, pool)
        .await?
        .unwrap_or_else(|| symbol.to_string());
    let query = format!(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount,
               amplitude, turnover_rate, volume_ratio, change_percent, change
        FROM historical_data
        WHERE symbol = ? AND date >= ? AND date <= ? AND {VALID_HISTORICAL_BAR_FILTER}
        ORDER BY date ASC
        "#
    );
    let rows = sqlx::query_as::<_, HistoricalData>(
        &query,
    )
    .bind(actual_symbol)
    .bind(start_date)
    .bind(end_date)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// 获取最近 N 天的历史数据
pub async fn get_recent_historical_data(
    symbol: &str,
    days: usize,
    pool: &SqlitePool,
) -> Result<Vec<HistoricalData>, AppError> {
    let actual_symbol = resolve_historical_symbol(symbol, pool)
        .await?
        .unwrap_or_else(|| symbol.to_string());
    let query = format!(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount,
               amplitude, turnover_rate, volume_ratio, change_percent, change
        FROM historical_data
        WHERE symbol = ? AND {VALID_HISTORICAL_BAR_FILTER}
        ORDER BY date DESC
        LIMIT ?
        "#
    );
    let rows = sqlx::query_as::<_, HistoricalData>(
        &query,
    )
    .bind(actual_symbol)
    .bind(days as i64)
    .fetch_all(pool)
    .await?;

    // 反转为时间正序
    let mut result = rows;
    result.reverse();
    Ok(result)
}

/// 批量获取多只股票最近 N 天历史数据，返回每只股票时间正序序列。
pub async fn get_recent_historical_data_for_symbols(
    symbols: &[String],
    days: usize,
    pool: &SqlitePool,
) -> Result<Vec<(String, Vec<HistoricalData>)>, AppError> {
    if symbols.is_empty() || days == 0 {
        return Ok(Vec::new());
    }

    let mut query_builder = QueryBuilder::new(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount,
               amplitude, turnover_rate, volume_ratio, change_percent, change
        FROM (
            SELECT symbol, date, open, high, low, close, volume, amount,
                   amplitude, turnover_rate, volume_ratio, change_percent, change,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM historical_data
            WHERE "#,
    );
    query_builder.push(VALID_HISTORICAL_BAR_FILTER);
    query_builder.push(" AND symbol IN (");
    let mut separated = query_builder.separated(", ");
    for symbol in symbols {
        separated.push_bind(symbol);
    }
    separated.push_unseparated(")");
    query_builder.push(
        r#"
        )
        WHERE rn <= "#,
    );
    query_builder.push_bind(days as i64);
    query_builder.push(" ORDER BY symbol ASC, date ASC");

    let rows: Vec<HistoricalData> = query_builder
        .build_query_as()
        .fetch_all(pool)
        .await?;
    let mut grouped: BTreeMap<String, Vec<HistoricalData>> = BTreeMap::new();
    for row in rows {
        grouped.entry(row.symbol.clone()).or_default().push(row);
    }

    Ok(grouped.into_iter().collect())
}

/// 获取历史数据足够长的股票代码列表（用于截面排名）
pub async fn get_symbols_with_min_bars(
    min_bars: i64,
    pool: &SqlitePool,
) -> Result<Vec<String>, AppError> {
    let rows: Vec<(String,)> = sqlx::query_as(
        &format!("SELECT symbol FROM historical_data WHERE {VALID_HISTORICAL_BAR_FILTER} GROUP BY symbol HAVING COUNT(*) >= ? ORDER BY symbol"),
    )
    .bind(min_bars)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

/// 获取同时满足「历史 ≥ min_bars 根」且「流通市值 ≥ min_cap（元）」的股票代码。
///
/// 用于把截面排名限定在可投资的流动大中盘域——截面技术信号在小盘上反向且不可交易。
pub async fn get_symbols_with_min_bars_and_cap(
    min_bars: i64,
    min_cap: f64,
    pool: &SqlitePool,
) -> Result<Vec<String>, AppError> {
    let rows: Vec<(String,)> = sqlx::query_as(&format!(
        "SELECT h.symbol FROM historical_data h \
         JOIN stock_capital c ON c.symbol = h.symbol \
         WHERE {VALID_HISTORICAL_BAR_FILTER} AND c.circulating_market_cap >= ? \
         GROUP BY h.symbol HAVING COUNT(*) >= ? ORDER BY h.symbol"
    ))
    .bind(min_cap)
    .bind(min_bars)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

/// 获取最新收盘价
pub async fn get_latest_close_price(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<Option<f64>, AppError> {
    let actual_symbol = resolve_historical_symbol(symbol, pool)
        .await?
        .unwrap_or_else(|| symbol.to_string());
    let query = format!(
        "SELECT close FROM historical_data WHERE symbol = ? AND {VALID_HISTORICAL_BAR_FILTER} ORDER BY date DESC LIMIT 1"
    );
    let result: Option<(f64,)> = sqlx::query_as(
        &query,
    )
    .bind(actual_symbol)
    .fetch_optional(pool)
    .await?;

    Ok(result.map(|(close,)| close))
}

/// 获取股票列表（带搜索和分页）
pub async fn get_stock_list(
    keyword: Option<&str>,
    page: i64,
    page_size: i64,
    pool: &SqlitePool,
) -> Result<(Vec<StockInfo>, i64), AppError> {
    let offset = (page - 1) * page_size;
    
    let (data, total) = if let Some(kw) = keyword {
        let pattern = format!("%{kw}%");
        
        let data = sqlx::query_as::<_, StockInfo>(
            "SELECT symbol, name, exchange FROM stock_info 
             WHERE symbol LIKE ? OR name LIKE ?
             ORDER BY symbol LIMIT ? OFFSET ?",
        )
        .bind(&pattern)
        .bind(&pattern)
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;
        
        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM stock_info WHERE symbol LIKE ? OR name LIKE ?",
        )
        .bind(&pattern)
        .bind(&pattern)
        .fetch_one(pool)
        .await?;
        
        (data, count.0)
    } else {
        let data = sqlx::query_as::<_, StockInfo>(
            "SELECT symbol, name, exchange FROM stock_info ORDER BY symbol LIMIT ? OFFSET ?",
        )
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;
        
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM stock_info")
            .fetch_one(pool)
            .await?;
        
        (data, count.0)
    };
    
    Ok((data, total))
}

// =============================================================================
// 实时数据仓库
// =============================================================================

/// 获取实时数据列表
pub async fn get_realtime_list(
    keyword: Option<&str>,
    page: i64,
    page_size: i64,
    pool: &SqlitePool,
) -> Result<(Vec<RealtimeData>, i64), AppError> {
    let offset = (page - 1) * page_size;
    
    let (data, total) = if let Some(kw) = keyword {
        let pattern = format!("%{kw}%");
        
        let data = sqlx::query_as::<_, RealtimeData>(
            "SELECT symbol, name, date, close, volume, amount, amplitude, 
                    turnover_rate, volume_ratio, change_percent, change
             FROM realtime_data 
             WHERE symbol LIKE ? OR name LIKE ?
             ORDER BY change_percent DESC LIMIT ? OFFSET ?",
        )
        .bind(&pattern)
        .bind(&pattern)
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;
        
        let count: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM realtime_data WHERE symbol LIKE ? OR name LIKE ?",
        )
        .bind(&pattern)
        .bind(&pattern)
        .fetch_one(pool)
        .await?;
        
        (data, count.0)
    } else {
        let data = sqlx::query_as::<_, RealtimeData>(
            "SELECT symbol, name, date, close, volume, amount, amplitude, 
                    turnover_rate, volume_ratio, change_percent, change
             FROM realtime_data ORDER BY change_percent DESC LIMIT ? OFFSET ?",
        )
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;
        
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM realtime_data")
            .fetch_one(pool)
            .await?;

        (data, count.0)
    };

    Ok((data, total))
}

// =============================================================================
// 股本与量比/换手率
// =============================================================================

/// 写入/更新某股票的股本数据（upsert）
pub async fn upsert_stock_capital(
    pool: &SqlitePool,
    capital: &StockCapital,
) -> Result<(), AppError> {
    let symbol = canonical_stock_symbol(&capital.symbol);
    sqlx::query(
        r#"
        INSERT INTO stock_capital (symbol, circulating_shares, total_shares, circulating_market_cap, pe, pb, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(symbol) DO UPDATE SET
            circulating_shares = EXCLUDED.circulating_shares,
            total_shares = EXCLUDED.total_shares,
            circulating_market_cap = EXCLUDED.circulating_market_cap,
            pe = EXCLUDED.pe,
            pb = EXCLUDED.pb,
            updated_at = CURRENT_TIMESTAMP
        "#,
    )
    .bind(symbol)
    .bind(capital.circulating_shares)
    .bind(capital.total_shares)
    .bind(capital.circulating_market_cap)
    .bind(capital.pe)
    .bind(capital.pb)
    .execute(pool)
    .await?;
    Ok(())
}

/// 读取某股票的股本数据，不存在时返回 None
pub async fn get_stock_capital(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<Option<StockCapital>, AppError> {
    let capital = sqlx::query_as::<_, StockCapital>(
        r#"
        SELECT symbol, circulating_shares, total_shares, circulating_market_cap, pe, pb
        FROM stock_capital WHERE symbol = ?
        "#,
    )
    .bind(symbol)
    .fetch_optional(pool)
    .await?;
    Ok(capital)
}

/// 写入一个报告期的基本面财务指标（按 (symbol, report_date) 幂等更新）。
pub async fn upsert_stock_fundamental(
    pool: &SqlitePool,
    f: &StockFundamental,
) -> Result<(), AppError> {
    let symbol = canonical_stock_symbol(&f.symbol);
    sqlx::query(
        r#"
        INSERT INTO stock_fundamentals
            (symbol, report_date, eps, bps, roe, profit_growth, revenue_growth, debt_ratio, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(symbol, report_date) DO UPDATE SET
            eps = EXCLUDED.eps,
            bps = EXCLUDED.bps,
            roe = EXCLUDED.roe,
            profit_growth = EXCLUDED.profit_growth,
            revenue_growth = EXCLUDED.revenue_growth,
            debt_ratio = EXCLUDED.debt_ratio,
            updated_at = CURRENT_TIMESTAMP
        "#,
    )
    .bind(symbol)
    .bind(&f.report_date)
    .bind(f.eps)
    .bind(f.bps)
    .bind(f.roe)
    .bind(f.profit_growth)
    .bind(f.revenue_growth)
    .bind(f.debt_ratio)
    .execute(pool)
    .await?;
    Ok(())
}

/// 读取某股票的全部基本面报告期（按报告期升序）。
pub async fn get_stock_fundamentals(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<Vec<StockFundamental>, AppError> {
    let rows = sqlx::query_as::<_, StockFundamental>(
        r#"
        SELECT symbol, report_date, eps, bps, roe, profit_growth, revenue_growth, debt_ratio
        FROM stock_fundamentals WHERE symbol = ?
        ORDER BY report_date ASC
        "#,
    )
    .bind(symbol)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// 回填某股票全部历史数据的量比与换手率。
///
/// 量比始终可算（仅依赖成交量序列）；换手率需要流通股本，若无股本数据则保持 0。
/// 返回更新的行数。
pub async fn backfill_volume_metrics(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<u64, AppError> {
    let symbol = canonical_stock_symbol(symbol);
    // 取全部历史数据（按日期正序）
    let rows = sqlx::query_as::<_, HistoricalData>(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount,
               amplitude, turnover_rate, volume_ratio, change_percent, change
        FROM historical_data
        WHERE symbol = ?
        ORDER BY date ASC
        "#,
    )
    .bind(&symbol)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(0);
    }

    let volumes: Vec<f64> = rows.iter().map(|r| r.volume as f64).collect();
    let volume_ratios = calculate_volume_ratio_series(&volumes, DEFAULT_VOLUME_RATIO_PERIOD);
    let circulating_shares = get_stock_capital(&symbol, pool)
        .await?
        .map(|c| c.circulating_shares)
        .unwrap_or(0.0);

    let mut tx = pool.begin().await?;
    let mut updated = 0u64;
    for (i, row) in rows.iter().enumerate() {
        let turnover = calculate_turnover_rate(row.amount, row.close, circulating_shares);
        let result = sqlx::query(
            "UPDATE historical_data SET volume_ratio = ?, turnover_rate = ? WHERE symbol = ? AND date = ?",
        )
        .bind(volume_ratios[i])
        .bind(turnover)
        .bind(&symbol)
        .bind(row.date)
        .execute(&mut *tx)
        .await?;
        updated += result.rows_affected();
    }
    tx.commit().await?;

    // 同步最新一日到 realtime_data
    if let Some(last) = rows.last() {
        let last_turnover = calculate_turnover_rate(last.amount, last.close, circulating_shares);
        sqlx::query(
            "UPDATE realtime_data SET volume_ratio = ?, turnover_rate = ? WHERE symbol = ?",
        )
        .bind(volume_ratios[rows.len() - 1])
        .bind(last_turnover)
        .bind(&symbol)
        .execute(pool)
        .await?;
    }

    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("应创建内存 SQLite");

        sqlx::query(
            r#"
            CREATE TABLE historical_data (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                close REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                volume INTEGER NOT NULL,
                amount REAL NOT NULL,
                amplitude REAL NOT NULL,
                turnover_rate REAL NOT NULL,
                change_percent REAL NOT NULL,
                change REAL NOT NULL,
                volume_ratio REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (symbol, date)
            )
            "#,
        )
        .execute(&pool)
        .await
        .expect("应创建历史数据表");

        pool
    }

    async fn insert_history(
        pool: &SqlitePool,
        symbol: &str,
        date: &str,
        open: f64,
        close: f64,
    ) {
        let high = open.max(close) + 0.1;
        let low = open.min(close) - 0.1;
        sqlx::query(
            r#"
            INSERT INTO historical_data
                (symbol, date, open, close, high, low, volume, amount, amplitude,
                 turnover_rate, change_percent, change, volume_ratio)
            VALUES (?, ?, ?, ?, ?, ?, 1000, 10000, 1.0, 1.0, 1.0, 0.1, 1.0)
            "#,
        )
        .bind(symbol)
        .bind(date)
        .bind(open)
        .bind(close)
        .bind(high)
        .bind(low)
        .execute(pool)
        .await
        .expect("应插入历史数据");
    }

    async fn stock_data_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("应创建内存 SQLite");

        for sql in [
            include_str!("../../migrations/01_create_tables.sql"),
            include_str!("../../migrations/03_volume_metrics.sql"),
            include_str!("../../migrations/04_stock_fundamentals.sql"),
            include_str!("../../migrations/05_capital_valuation.sql"),
            include_str!("../../migrations/06_stock_category.sql"),
        ] {
            for statement in sql.split(';').map(str::trim).filter(|s| !s.is_empty()) {
                sqlx::query(statement)
                    .execute(&pool)
                    .await
                    .expect("应创建股票数据测试表");
            }
        }

        pool
    }

    #[tokio::test]
    async fn test_repository_writes_merge_symbol_variants() {
        let pool = stock_data_pool().await;

        batch_insert_stock_info(
            &pool,
            vec![StockInfo {
                symbol: "002466".to_string(),
                name: "002466".to_string(),
                exchange: "sz".to_string(),
            }],
        )
        .await
        .expect("首次写入应成功");
        batch_insert_stock_info(
            &pool,
            vec![StockInfo {
                symbol: "002466.SZ".to_string(),
                name: "天齐锂业".to_string(),
                exchange: "SZ".to_string(),
            }],
        )
        .await
        .expect("后缀代码写入应成功");

        batch_insert_stock(
            &pool,
            vec![Stock {
                symbol: "002466.SZ".to_string(),
                name: "天齐锂业".to_string(),
                exchange: "002466.SZ".to_string(),
                ..Stock::default()
            }],
        )
        .await
        .expect("带后缀代码的股票详情应写入成功");

        batch_insert_historical_data(
            "002466.SZ",
            &pool,
            vec![HistoricalData {
                symbol: "002466.SZ".to_string(),
                date: chrono::NaiveDate::from_ymd_opt(2026, 7, 15).expect("测试日期应有效"),
                open: 47.0,
                close: 47.43,
                high: 48.0,
                low: 46.5,
                volume: 754_792,
                amount: 3_627_171_202.52,
                amplitude: 4.39,
                turnover_rate: 5.18,
                volume_ratio: 1.09,
                change_percent: 0.17,
                change: 0.08,
            }],
        )
        .await
        .expect("带后缀代码的历史数据应写入成功");

        upsert_stock_capital(
            &pool,
            &StockCapital {
                symbol: "002466.SZ".to_string(),
                circulating_shares: 100.0,
                total_shares: 120.0,
                circulating_market_cap: 4_700.0,
                pe: 10.8,
                pb: 1.75,
            },
        )
        .await
        .expect("带后缀代码的股本应写入成功");

        upsert_stock_fundamental(
            &pool,
            &StockFundamental {
                symbol: "002466.SZ".to_string(),
                report_date: "2026-03-31".to_string(),
                eps: Some(0.5),
                bps: Some(10.0),
                roe: Some(5.0),
                profit_growth: Some(6.0),
                revenue_growth: Some(7.0),
                debt_ratio: Some(8.0),
            },
        )
        .await
        .expect("带后缀代码的基本面应写入成功");

        let updated = backfill_volume_metrics("002466.SZ", &pool)
            .await
            .expect("带后缀代码应能回填规范代码的数据");
        assert_eq!(updated, 1);

        let stock_info: Vec<(String, String, String)> =
            sqlx::query_as("SELECT symbol, name, exchange FROM stock_info")
                .fetch_all(&pool)
                .await
                .expect("应查询股票信息");
        assert_eq!(
            stock_info,
            vec![(
                "002466".to_string(),
                "天齐锂业".to_string(),
                "sz".to_string()
            )]
        );

        for table in [
            "stock",
            "historical_data",
            "realtime_data",
            "stock_capital",
            "stock_fundamentals",
        ] {
            let symbols: Vec<(String,)> =
                sqlx::query_as(&format!("SELECT DISTINCT symbol FROM {table}"))
                    .fetch_all(&pool)
                    .await
                    .expect("应查询规范化后的股票代码");
            assert_eq!(symbols, vec![("002466".to_string(),)], "{table}");
        }
    }

    #[tokio::test]
    async fn test_resolve_historical_symbol_prefers_latest_valid_variant() {
        let pool = test_pool().await;
        insert_history(&pool, "000001.SZ", "2025-01-01", 10.0, 10.2).await;
        insert_history(&pool, "000001", "2026-01-01", 11.0, 11.2).await;

        let resolved = resolve_historical_symbol("000001.SZ", &pool)
            .await
            .expect("解析应成功");

        assert_eq!(resolved.as_deref(), Some("000001"));
    }

    #[tokio::test]
    async fn test_recent_historical_data_filters_zero_price_rows() {
        let pool = test_pool().await;
        insert_history(&pool, "603005.SH", "2026-01-01", 40.0, 41.0).await;
        sqlx::query(
            r#"
            INSERT INTO historical_data
                (symbol, date, open, close, high, low, volume, amount, amplitude,
                 turnover_rate, change_percent, change, volume_ratio)
            VALUES ('603005.SH', '2026-01-02', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            "#,
        )
        .execute(&pool)
        .await
        .expect("应插入无效占位数据");

        let rows = get_recent_historical_data("603005", 5, &pool)
            .await
            .expect("查询应成功");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].symbol, "603005.SH");
        assert_eq!(rows[0].date.to_string(), "2026-01-01");
    }

    #[tokio::test]
    async fn test_recent_historical_data_for_symbols_limits_each_symbol() {
        let pool = test_pool().await;
        insert_history(&pool, "000001", "2026-01-01", 10.0, 10.1).await;
        insert_history(&pool, "000001", "2026-01-02", 10.1, 10.2).await;
        insert_history(&pool, "000001", "2026-01-03", 10.2, 10.3).await;
        insert_history(&pool, "600000", "2026-01-01", 20.0, 20.1).await;
        insert_history(&pool, "600000", "2026-01-02", 20.1, 20.2).await;
        insert_history(&pool, "600000", "2026-01-03", 20.2, 20.3).await;

        let grouped = get_recent_historical_data_for_symbols(
            &["000001".to_string(), "600000".to_string()],
            2,
            &pool,
        )
        .await
        .expect("批量查询应成功");

        assert_eq!(grouped.len(), 2);
        for (_, rows) in grouped {
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0].date.to_string(), "2026-01-02");
            assert_eq!(rows[1].date.to_string(), "2026-01-03");
        }
    }
}
