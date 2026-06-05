//! 历史数据服务

use crate::api::stock;
use crate::db::{models::*, repository, DbPool};
use crate::error::AppError;

/// 单只股票一键全量刷新的结果汇总
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RefreshSummary {
    /// 写入的历史K线条数
    pub bars: u64,
    /// 股本+估值(PE/PB)是否更新成功
    pub capital_updated: bool,
    /// 写入的基本面报告期行数
    pub fundamental_reports: u32,
}

/// **一键刷新单只股票的全部所需数据**：历史K线 + 股本/估值(PE/PB) + 基本面财务指标
/// + 量比/换手率回填。一次调用更新全部相关表，避免零散重复操作。
///
/// 各非历史步骤失败不阻断主流程（优雅降级）：历史拉取/入库失败才返回 Err。
pub async fn refresh_stock_full(symbol: &str, pool: &DbPool) -> Result<RefreshSummary, AppError> {
    // 1. 历史K线（主流程，失败即返回 Err）
    let api_data = stock::fetch_historical_data(symbol).await?;
    let bars = repository::batch_insert_historical_data(symbol, pool, api_data).await?;

    // 2. 股本 + 估值（ssjy 一次返回 lt/sz/hs/lb/pe/sjl）
    let mut capital_updated = false;
    if let Ok(quote) = stock::fetch_stock_capital(symbol).await {
        if let Some(close) = repository::get_latest_close_price(symbol, pool).await? {
            if close > 0.0 {
                let capital = StockCapital {
                    symbol: symbol.to_string(),
                    circulating_shares: quote.circulating_market_cap / close,
                    total_shares: quote.total_market_cap / close,
                    circulating_market_cap: quote.circulating_market_cap,
                    pe: quote.pe,
                    pb: quote.pb,
                };
                repository::upsert_stock_capital(pool, &capital).await?;
                capital_updated = true;
            }
        }
    }

    // 3. 基本面财务指标（cwzb：ROE/EPS/BPS/增长率等，非技术预测维度）
    let mut fundamental_reports = 0u32;
    if let Ok(reports) = stock::fetch_financial_indicators(symbol).await {
        for f in &reports {
            if repository::upsert_stock_fundamental(pool, f).await.is_ok() {
                fundamental_reports += 1;
            }
        }
    }

    // 4. 量比/换手率回填（量比始终可算；换手率依赖上面的股本）
    repository::backfill_volume_metrics(symbol, pool).await?;

    Ok(RefreshSummary {
        bars,
        capital_updated,
        fundamental_reports,
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

