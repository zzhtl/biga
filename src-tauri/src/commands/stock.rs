use crate::csv::handler::read_csv_to_struct;
use crate::db::{batch_insert_stock, batch_insert_stock_info};
use crate::error::AppError;
use crate::{api::stock, db::models::StockInfo};
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_stock_infos(pool: State<'_, SqlitePool>) -> Result<Vec<StockInfo>, AppError> {
    let records = sqlx::query_as::<_, StockInfo>(
        r#"
        SELECT
            COALESCE(symbol, '') as symbol,
            COALESCE(name, '') as name,
            COALESCE(exchange, '') as exchange
        FROM stock_info
        "#,
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

    // 3. 读取csv文件保存股票详细数据
    let stocks = read_csv_to_struct("data/stock_basic.csv").unwrap_or(Vec::new());
    batch_insert_stock(&pool, stocks).await?;

    Ok(true)
}
