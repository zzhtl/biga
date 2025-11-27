//! 数据仓库层
//! 
//! 提供数据访问接口，封装所有 SQL 操作

use crate::config::constants::BATCH_SIZE;
use crate::db::models::*;
use crate::error::AppError;
use sqlx::{QueryBuilder, sqlite::SqlitePool};

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
            b.push_bind(&data.symbol)
                .push_bind(&data.name)
                .push_bind(&data.exchange);
        });
        query_builder.push(" ON CONFLICT(symbol) DO NOTHING");
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
            b.push_bind(&data.symbol)
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

    let mut tx = pool.begin().await?;
    let mut batch_size: u64 = 0;
    
    for chunk in data_list.chunks(BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO historical_data (symbol, date, open, close, high, low, volume,
            amount, amplitude, turnover_rate, change, change_percent) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(data.date)
                .push_bind(data.open)
                .push_bind(data.close)
                .push_bind(data.high)
                .push_bind(data.low)
                .push_bind(data.volume)
                .push_bind(data.amount)
                .push_bind(data.amplitude)
                .push_bind(data.turnover_rate)
                .push_bind(data.change)
                .push_bind(data.change_percent);
        });

        query_builder.push(" ON CONFLICT(symbol, date) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }
    
    // 更新实时数据
    if let Some(last_history) = data_list.last() {
        let stock_info = get_stock_info(symbol, pool).await?;
        
        let mut realtime_builder = QueryBuilder::new(
            "INSERT INTO realtime_data (symbol, name, date, close, volume, amount, amplitude, turnover_rate, change, change_percent) ",
        );
        realtime_builder.push_values(&[last_history], |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(&stock_info.name)
                .push_bind(data.date)
                .push_bind(data.close)
                .push_bind(data.volume)
                .push_bind(data.amount)
                .push_bind(data.amplitude)
                .push_bind(data.turnover_rate)
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
    let rows = sqlx::query_as::<_, HistoricalData>(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount, 
               amplitude, turnover_rate, change_percent, change
        FROM historical_data
        WHERE symbol = ? AND date >= ? AND date <= ?
        ORDER BY date ASC
        "#,
    )
    .bind(symbol)
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
    let rows = sqlx::query_as::<_, HistoricalData>(
        r#"
        SELECT symbol, date, open, high, low, close, volume, amount, 
               amplitude, turnover_rate, change_percent, change
        FROM historical_data
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
        "#,
    )
    .bind(symbol)
    .bind(days as i64)
    .fetch_all(pool)
    .await?;

    // 反转为时间正序
    let mut result = rows;
    result.reverse();
    Ok(result)
}

/// 获取最新收盘价
pub async fn get_latest_close_price(
    symbol: &str,
    pool: &SqlitePool,
) -> Result<Option<f64>, AppError> {
    let result: Option<(f64,)> = sqlx::query_as(
        "SELECT close FROM historical_data WHERE symbol = ? ORDER BY date DESC LIMIT 1",
    )
    .bind(symbol)
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
                    turnover_rate, change_percent, change 
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
                    turnover_rate, change_percent, change 
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

