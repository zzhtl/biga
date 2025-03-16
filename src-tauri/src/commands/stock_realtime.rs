use crate::db::models::RealtimeData;
use crate::error::AppError;
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_realtime_data(pool: State<'_, SqlitePool>) -> Result<Vec<RealtimeData>, AppError> {
    let records = sqlx::query_as!(
        RealtimeData,
        r#"
        SELECT
        COALESCE(symbol, '') as symbol,
        COALESCE(name, '') as name,
        COALESCE(date, '') as date,
        COALESCE(ytd_close, 0.00) as ytd_close,
        COALESCE(close, 0.00) as close,
        COALESCE(volume, 0) as volume,
        COALESCE(change, 0.00) as change,
        COALESCE(change_percent, 0.00) as change_percent
        FROM realtime_data
        "#
    )
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}
