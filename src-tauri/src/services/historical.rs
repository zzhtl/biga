//! 历史数据服务

use crate::services::market_data::{canonical_symbol, ensure_data, QueryBudget, Resource};
use crate::db::{models::*, repository, DbPool};
use crate::error::AppError;

/// 单只股票一键全量刷新的结果汇总
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RefreshSummary {
    /// 可供分析的历史K线条数（最多 3000 条）
    pub bars: u64,
    /// 股本+估值(PE/PB)是否更新成功
    pub capital_updated: bool,
    /// 写入的基本面报告期行数
    pub fundamental_reports: u32,
    pub requests: u32,
    pub cache_hits: u32,
    pub warnings: Vec<String>,
}

/// **一键刷新单只股票的全部所需数据**：历史K线 + 股本/估值(PE/PB) + 基本面财务指标
/// + 量比/换手率回填。一次调用更新全部相关表，避免零散重复操作。
///
/// 缓存优先，最多三个外部请求。取数失败时返回已有数据及警告，无历史数据才报错。
pub async fn refresh_stock_full(symbol: &str, pool: &DbPool) -> Result<RefreshSummary, AppError> {
    let symbol = canonical_symbol(symbol)?;
    let budget = QueryBudget::new(3);
    let history = ensure_data(pool, &symbol, Resource::History, &budget).await?;
    let data = repository::get_recent_historical_data(&symbol, 3000, pool).await?;
    if data.is_empty() {
        return Err(AppError::InvalidInput(history.warning.unwrap_or_else(|| "没有可用历史数据".into())));
    }
    let capital = ensure_data(pool, &symbol, Resource::Valuation, &budget).await?;
    let fundamentals = ensure_data(pool, &symbol, Resource::Fundamentals, &budget).await?;
    let (requests, cache_hits) = budget.counts();
    Ok(RefreshSummary {
        bars: data.len() as u64,
        capital_updated: !capital.stale,
        fundamental_reports: repository::get_stock_fundamentals(&symbol, pool).await?.len() as u32,
        requests,
        cache_hits,
        warnings: [history, capital, fundamentals].into_iter().filter_map(|r| r.warning).collect(),
    })
}

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

