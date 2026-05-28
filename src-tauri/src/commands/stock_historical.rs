use crate::db::{
    backfill_volume_metrics, batch_insert_historical_data, get_latest_close_price, get_historical_data as query_historical_data,
    upsert_stock_capital,
};
use crate::db::models::{HistoricalData, StockCapital};
use crate::error::AppError;
use crate::api::stock;
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

#[tauri::command]
pub async fn refresh_historical_data(
    symbol: String,
    pool: State<'_, SqlitePool>, // 从全局状态中提取连接池
) -> Result<bool, AppError> {
    // 1. 从API获取数据
    let api_data = stock::fetch_historical_data(&symbol).await?;

    // 2. 存储到数据库
    batch_insert_historical_data(&symbol, &pool, api_data).await?;

    // 3. 获取股本并回填量比/换手率（股本接口失败不阻断主流程，量比仍可计算）
    if let Err(e) = refresh_volume_metrics(&symbol, &pool).await {
        eprintln!("回填 {symbol} 量比/换手率失败: {e}");
    }

    Ok(true)
}

/// 拉取流通市值推导流通股本，并回填该股票的量比与换手率
async fn refresh_volume_metrics(symbol: &str, pool: &SqlitePool) -> Result<(), AppError> {
    if let Ok(quote) = stock::fetch_stock_capital(symbol).await {
        // 用最新收盘价把流通市值/总市值换算成股本
        if let Some(close) = get_latest_close_price(symbol, pool).await? {
            if close > 0.0 {
                let capital = StockCapital {
                    symbol: symbol.to_string(),
                    circulating_shares: quote.circulating_market_cap / close,
                    total_shares: quote.total_market_cap / close,
                    circulating_market_cap: quote.circulating_market_cap,
                };
                upsert_stock_capital(pool, &capital).await?;
            }
        }
    }
    // 无论是否拿到股本，都回填一次（量比始终可算）
    backfill_volume_metrics(symbol, pool).await?;
    Ok(())
}
