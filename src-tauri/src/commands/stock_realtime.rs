use crate::db::models::RealtimeData;
use crate::error::AppError;
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_realtime_data(
    search: String,
    column: String,
    sort: String,
    pool: State<'_, SqlitePool>,
) -> Result<Vec<RealtimeData>, AppError> {
    let order_by = format!("{} {}", column, sort);
    let records = if search.trim().is_empty() {
        sqlx::query_as::<_, RealtimeData>(&format!(
            r#"
                SELECT
                COALESCE(symbol, '') as symbol,
                COALESCE(name, '') as name,
                date,
                COALESCE(close, 0.00) as close,
                COALESCE(volume, 0) as volume,
                COALESCE(amount, 0.00) as amount,
                COALESCE(amplitude, 0.00) as amplitude,
                COALESCE(turnover_rate, 0.00) as turnover_rate,
                COALESCE(change_percent, 0.00) as change_percent,
                COALESCE(change, 0.00) as change
                FROM realtime_data ORDER BY {}
                "#,
            order_by
        ))
        .fetch_all(&*pool)
        .await?
    } else {
        let search_pattern = format!("%{}%", search);
        sqlx::query_as::<_, RealtimeData>(&format!(
            r#"
                SELECT
                COALESCE(symbol, '') as symbol,
                COALESCE(name, '') as name,
                date,
                COALESCE(close, 0.00) as close,
                COALESCE(volume, 0) as volume,
                COALESCE(amount, 0.00) as amount,
                COALESCE(amplitude, 0.00) as amplitude,
                COALESCE(turnover_rate, 0.00) as turnover_rate,
                COALESCE(change_percent, 0.00) as change_percent,
                COALESCE(change, 0.00) as change
                FROM realtime_data
                WHERE symbol LIKE ? OR name LIKE ?
                ORDER BY {}
                "#,
            order_by
        ))
        .bind(search_pattern.clone())
        .bind(search_pattern)
        .fetch_all(&*pool)
        .await?
    };
    Ok(records)
}
