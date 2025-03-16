pub mod models;
use crate::constants;
use crate::error::AppError;
use models::{HistoricalData, StockInfo};
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
    let mut batch_size = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO historical_data (symbol, date, open, close, high, low, volume) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(&data.date)
                .push_bind(&data.open)
                .push_bind(&data.close)
                .push_bind(&data.high)
                .push_bind(&data.low)
                .push_bind(&data.volume);
        });

        query_builder.push(" ON CONFLICT(symbol, date) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }
    // 保存最新的股票涨跌数据
    let last_history = data_list.last().unwrap();
    let last_two_history = data_list.get(data_list.len() - 2).unwrap_or(last_history);
    let change = last_history.close - last_two_history.close;
    let change_percent = change / last_two_history.close;
    let mut realtime_builder = QueryBuilder::new(
        "INSERT INTO realtime_data (symbol, name, date, ytd_close, close, volume, change, change_percent) ",
    );
    let stock_info = get_stock_info(symbol, pool).await?;
    realtime_builder.push_values(&[last_history], |mut b, data| {
        b.push_bind(&data.symbol)
            .push_bind(&stock_info.name)
            .push_bind(&data.date)
            .push_bind(&last_two_history.close)
            .push_bind(&data.close)
            .push_bind(&data.volume)
            .push_bind(change)
            .push_bind(change_percent);
    });
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
