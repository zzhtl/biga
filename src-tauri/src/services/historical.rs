//! 历史数据服务

use crate::db::{models::*, repository, DbPool};
use crate::error::AppError;

/// 获取历史数据
pub async fn get_historical_data(
    symbol: &str,
    start_date: &str,
    end_date: &str,
    pool: &DbPool,
) -> Result<Vec<HistoricalData>, AppError> {
    repository::get_historical_data(symbol, start_date, end_date, pool).await
}

/// 获取最近历史数据
pub async fn get_recent_data(
    symbol: &str,
    days: usize,
    pool: &DbPool,
) -> Result<Vec<HistoricalData>, AppError> {
    repository::get_recent_historical_data(symbol, days, pool).await
}

/// 保存历史数据
pub async fn save_historical_data(
    symbol: &str,
    data: Vec<HistoricalData>,
    pool: &DbPool,
) -> Result<u64, AppError> {
    repository::batch_insert_historical_data(symbol, pool, data).await
}

/// 获取最新价格
pub async fn get_latest_price(symbol: &str, pool: &DbPool) -> Result<Option<f64>, AppError> {
    repository::get_latest_close_price(symbol, pool).await
}

