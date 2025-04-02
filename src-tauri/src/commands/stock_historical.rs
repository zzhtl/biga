use crate::db::batch_insert_historical_data;
use crate::error::AppError;
use crate::{api::stock, db::models::HistoricalData};
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_historical_data(
    symbol: String,
    start: String,
    end: String,
    pool: State<'_, SqlitePool>, // 从全局状态中提取连接池
) -> Result<Vec<HistoricalData>, AppError> {
    // 1. 从数据库查询
    let records = sqlx::query_as::<_, HistoricalData>(
        r#"
        SELECT * FROM historical_data
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date DESC
        "#,
    )
    .bind(symbol)
    .bind(start)
    .bind(end)
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}

#[tauri::command]
pub async fn refresh_historical_data(
    symbol: String,
    pool: State<'_, SqlitePool>, // 从全局状态中提取连接池
) -> Result<bool, AppError> {
    // 1. 从API获取数据
    let api_data = stock::fetch_historical_data(&symbol).await?;

    // 2. 存储到数据库
    batch_insert_historical_data(&symbol, &pool, api_data).await?;

    Ok(true)
}
