use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct StockInfo {
    #[sqlx(rename = "symbol")]
    pub symbol: String, // 股票代码(含市场前缀)
    pub name: String,     // 股票简称
    pub exchange: String, // 交易所(sh/sz)
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct StockInfoItem {
    #[serde(rename = "dm")]
    pub symbol: String, // 股票代码(含市场前缀)
    #[serde(rename = "mc")]
    pub name: String, // 股票简称
    #[serde(rename = "jys")]
    pub exchange: String, // 交易所(sh/sz)
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct HistoricalData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub open: f64,   // 开盘价
    pub close: f64,  // 收盘价
    pub high: f64,   // 最高价
    pub low: f64,    // 最低价
    pub volume: i64, // 成交量
}

#[derive(Debug, Deserialize)]
pub struct HistoricalDataItem {
    #[serde(rename = "d")] // 与 JSON 字段名 "d" 映射
    pub date: String,
    #[serde(rename = "o")]
    pub open: f64,
    #[serde(rename = "h")]
    pub high: f64,
    #[serde(rename = "l")]
    pub low: f64,
    #[serde(rename = "c")]
    pub close: f64,
    #[serde(rename = "v")]
    pub volume: i64,
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct RealtimeData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "name")]
    pub name: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub ytd_close: f64,      // 昨天收盘价
    pub close: f64,          // 今日收盘价
    pub volume: i64,         // 成交量
    pub change: f64,         // 涨跌额
    pub change_percent: f64, // 涨跌幅
}
