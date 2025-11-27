//! 股票服务

use crate::db::{models::*, repository, DbPool};
use crate::error::AppError;

/// 获取股票列表
pub async fn get_stock_list(
    keyword: Option<String>,
    page: i64,
    page_size: i64,
    pool: &DbPool,
) -> Result<(Vec<StockInfo>, i64), AppError> {
    repository::get_stock_list(keyword.as_deref(), page, page_size, pool).await
}

/// 获取单个股票信息
pub async fn get_stock_info(symbol: &str, pool: &DbPool) -> Result<StockInfo, AppError> {
    repository::get_stock_info(symbol, pool).await
}

/// 保存股票信息
pub async fn save_stock_info(
    data: Vec<StockInfo>,
    pool: &DbPool,
) -> Result<u64, AppError> {
    repository::batch_insert_stock_info(pool, data).await
}

/// 保存股票详细信息
pub async fn save_stock_details(data: Vec<Stock>, pool: &DbPool) -> Result<u64, AppError> {
    repository::batch_insert_stock(pool, data).await
}

