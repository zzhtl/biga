use crate::db::models::Stock;
use crate::error::AppError;
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_stock_list(
    pool: State<'_, SqlitePool>,
    search: String,
) -> Result<Vec<Stock>, AppError> {
    let search = search.trim();
    let search_pattern = format!("%{search}%");

    let records = sqlx::query_as::<_, Stock>(
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
            WHERE ? = '' OR (symbol LIKE ? OR name LIKE ? OR industry LIKE ?)
            "#,
    )
    .bind(search)
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .bind(search_pattern)
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}
