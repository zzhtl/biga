use crate::db::models::Stock;
use crate::error::AppError;
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_stock_list(pool: State<'_, SqlitePool>) -> Result<Vec<Stock>, AppError> {
    let records = sqlx::query_as!(
        Stock,
        r#"
        SELECT
            COALESCE(symbol, '') as symbol,
            COALESCE(name, '') as name,
            COALESCE(area, '') as area,
            COALESCE(industry, '') as industry,
            COALESCE(market, '') as market,
            COALESCE(exchange, '') as exchange,
            COALESCE(list_date, '') as list_date,
            COALESCE(act_name, '') as act_name,
            COALESCE(act_ent_type, '') as act_ent_type
        FROM stock
        "#
    )
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}
