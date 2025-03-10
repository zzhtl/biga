pub mod models;
use models::HistoricalData;
use sqlx::sqlite::SqlitePool;

use crate::error::AppError;

pub async fn create_historical_data(
    pool: &SqlitePool,
    data: &HistoricalData,
) -> Result<i64, AppError> {
    let mut conn = pool.acquire().await?;
    let id = sqlx::query!(
        r#"INSERT INTO historical_data
            (symbol, date, open, close, high, low, volume)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
        data.symbol,
        data.date,
        data.open,
        data.close,
        data.high,
        data.low,
        data.volume
    )
    .execute(&mut *conn)
    .await?
    .last_insert_rowid();

    Ok(id)
}
