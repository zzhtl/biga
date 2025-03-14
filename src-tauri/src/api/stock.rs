use crate::db::models::{HistoricalData, HistoricalDataItem, StockInfo, StockInfoItem};
use crate::error::AppError;
use chrono::NaiveDate;
use std::env;

// 查看全部股票名称以及代码
const ALL_SYMBOL_API: &str = "https://api.zhituapi.com/hs/list/all?token=ZHITU_TOKEN_LIMIT_TEST";
// 查看股票历史
const HISTORY_API: &str = "https://api.zhituapi.com/hs/history/fsjy";

pub async fn fetch_stock_infos() -> Result<Vec<StockInfo>, AppError> {
    let stock_infos: Vec<StockInfoItem> = reqwest::get(ALL_SYMBOL_API).await?.json().await?;

    parse_stock_info(stock_infos)
}

fn parse_stock_info(items: Vec<StockInfoItem>) -> Result<Vec<StockInfo>, AppError> {
    items
        .into_iter()
        .map(|item| {
            Ok(StockInfo {
                symbol: item.symbol,
                name: item.name,
                exchange: item.exchange,
            })
        })
        .collect()
}

pub async fn fetch_historical_data(symbol: &str) -> Result<Vec<HistoricalData>, AppError> {
    let token = env::var("DDE0C310-D0DE-4767-9712-A424BAC4326D").unwrap();

    let response = reqwest::get(&format!("{}/{}/d?token={}", HISTORY_API, symbol, token))
        .await?
        .json::<Vec<HistoricalDataItem>>() // 直接解析为结构体数组
        .await?;

    parse_historical_data(response, symbol)
}

fn parse_historical_data(
    items: Vec<HistoricalDataItem>,
    symbol: &str,
) -> Result<Vec<HistoricalData>, AppError> {
    items
        .into_iter()
        .map(|item| {
            Ok(HistoricalData {
                symbol: symbol.to_string(),
                date: NaiveDate::parse_from_str(&item.date, "%Y-%m-%d")?,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close,
                volume: item.volume,
            })
        })
        .collect()
}
