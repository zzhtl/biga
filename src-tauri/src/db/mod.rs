pub mod models;
pub mod prediction;
use crate::constants;
use crate::error::AppError;
use models::{HistoricalData, Stock, StockInfo};
use sqlx::sqlite::SqlitePool;
use sqlx::QueryBuilder;

pub async fn batch_insert_stock_info(
    pool: &SqlitePool,
    data_list: Vec<StockInfo>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }
    let mut tx = pool.begin().await?;
    let mut affected_rows = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
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

// 股票详细信息
pub async fn batch_insert_stock(pool: &SqlitePool, data_list: Vec<Stock>) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }
    let mut tx = pool.begin().await?;
    let mut affected_rows = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO stock (symbol, name, area, industry, market, exchange, list_date, act_name, act_ent_type) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            let exchange = data.exchange.split('.').last().unwrap_or("").to_lowercase();
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

pub async fn batch_insert_historical_data(
    symbol: &str,
    pool: &SqlitePool,
    data_list: Vec<HistoricalData>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }

    let mut tx = pool.begin().await?;

    // 批量生成占位符（自动处理参数展开）
    let mut batch_size: u64 = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO historical_data (symbol, date, open, close, high, low, volume,
            amount, amplitude, turnover_rate, change, change_percent) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(&data.date)
                .push_bind(&data.open)
                .push_bind(&data.close)
                .push_bind(&data.high)
                .push_bind(&data.low)
                .push_bind(&data.volume)
                .push_bind(&data.amount)
                .push_bind(&data.amplitude)
                .push_bind(&data.turnover_rate)
                .push_bind(&data.change)
                .push_bind(&data.change_percent);
        });

        query_builder.push(" ON CONFLICT(symbol, date) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }
    let last_history = data_list.last().unwrap();
    // 保存最新的股票涨跌数据
    let mut realtime_builder = QueryBuilder::new(
        "INSERT INTO realtime_data (symbol, name, date, close, volume, amount, amplitude, turnover_rate, change, change_percent) ",
    );
    let stock_info = get_stock_info(symbol, pool).await?;
    realtime_builder.push_values(&[last_history], |mut b, data| {
        b.push_bind(&data.symbol)
            .push_bind(&stock_info.name)
            .push_bind(&data.date)
            .push_bind(&data.close)
            .push_bind(&data.volume)
            .push_bind(&data.amount)
            .push_bind(&data.amplitude)
            .push_bind(&data.turnover_rate)
            .push_bind(&data.change)
            .push_bind(&data.change_percent);
    });
    // 添加 ON CONFLICT 更新逻辑
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

    tx.commit().await?;

    Ok(batch_size)
}

// 通过symbol获取单个股票信息
pub async fn get_stock_info(symbol: &str, pool: &SqlitePool) -> Result<StockInfo, AppError> {
    // 使用显式类型注解帮助编译器推断
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
