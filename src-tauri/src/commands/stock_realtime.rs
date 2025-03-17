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
        COALESCE(close, 0.00) as close,
        COALESCE(volume, 0) as volume,
        COALESCE(amount, 0.00) as amount,
        COALESCE(amplitude, 0.00) as amplitude,
        COALESCE(turnover_rate, 0.00) as turnover_rate,
        COALESCE(change_percent, 0.00) as change_percent,
        COALESCE(change, 0.00) as change
        FROM realtime_data
        "#
    )
    .fetch_all(&*pool)
    .await?;

    Ok(records)
}
