use crate::db::models::Stock;
use crate::error::AppError;
use crate::commands::pagination::{normalize_page, PagedResponse};
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_stock_list(
    pool: State<'_, SqlitePool>,
    search: String,
    page: u32,
    page_size: u32,
) -> Result<PagedResponse<Stock>, AppError> {
    let search = search.trim();
    let search_pattern = format!("%{search}%");
    let (page, page_size, offset) = normalize_page(page, page_size);

    let total = sqlx::query_scalar::<_, i64>(
        r#"
            SELECT COUNT(*)
            FROM stock
            WHERE ? = '' OR (symbol LIKE ? OR name LIKE ? OR industry LIKE ? OR category LIKE ?)
            "#,
    )
    .bind(search)
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .fetch_one(&*pool)
    .await?;

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
                COALESCE(act_ent_type, '') as act_ent_type,
                COALESCE(category, '') as category
            FROM stock
            WHERE ? = '' OR (symbol LIKE ? OR name LIKE ? OR industry LIKE ? OR category LIKE ?)
            ORDER BY category, symbol
            LIMIT ? OFFSET ?
            "#,
    )
    .bind(search)
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .bind(search_pattern.clone())
    .bind(search_pattern)
    .bind(i64::from(page_size))
    .bind(offset)
    .fetch_all(&*pool)
    .await?;

    Ok(PagedResponse {
        data: records,
        total,
        page,
        page_size,
    })
}
