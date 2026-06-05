use crate::db::get_historical_data as query_historical_data;
use crate::db::models::HistoricalData;
use crate::error::AppError;
use crate::services::historical::{refresh_stock_full, RefreshSummary};
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_historical_data(
    symbol: String,
    start: String,
    end: String,
    pool: State<'_, SqlitePool>, // 从全局状态中提取连接池
) -> Result<Vec<HistoricalData>, AppError> {
    query_historical_data(&symbol, &start, &end, &pool).await
}

/// 刷新单只股票的全部所需数据：历史K线 + 股本/估值(PE/PB) + 基本面 + 量比/换手率回填。
/// 一次刷新更新全部相关表，避免零散重复操作。返回各步更新汇总（前端用于日志/提示）。
#[tauri::command]
pub async fn refresh_historical_data(
    symbol: String,
    pool: State<'_, SqlitePool>, // 从全局状态中提取连接池
) -> Result<RefreshSummary, AppError> {
    refresh_stock_full(&symbol, &pool).await
}
