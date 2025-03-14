use crate::db::batch_insert_stock_info;
use crate::error::AppError;
use crate::{api::stock, db::models::StockInfo};
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_stock_infos(pool: State<'_, SqlitePool>) -> Result<Vec<StockInfo>, AppError> {
    let records = sqlx::query_as!(
        StockInfo,
        r#"
        SELECT
            COALESCE(symbol, '') as symbol,
            COALESCE(name, '') as name,
            COALESCE(exchange, '') as exchange
        FROM stock_info
        "#
    )
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}

#[tauri::command]
pub async fn refresh_stock_infos(pool: State<'_, SqlitePool>) -> Result<bool, AppError> {
    // 1. 从API获取数据
    let api_data = stock::fetch_stock_infos().await?;

    // 2. 更新数据库
    batch_insert_stock_info(&pool, api_data).await?;

    Ok(true)
}
