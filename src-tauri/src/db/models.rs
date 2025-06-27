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

#[derive(Default, Debug, Serialize, Deserialize, FromRow)]
pub struct Stock {
    #[sqlx(rename = "symbol")]
    pub symbol: String, // 股票代码(含市场前缀)
    pub name: String,     // 股票简称
    pub area: String,     // 地域
    pub industry: String, // 所属行业
    pub market: String,   // 市场类型:主板、创业板、科创板
    #[serde(rename = "ts_code")]
    #[sqlx(rename = "exchange")]
    pub exchange: String, // 交易所(sh/sz)
    pub list_date: String, // 上市日期
    pub act_name: String, // 实控人名称
    pub act_ent_type: String, // 实控人企业性质
}

#[derive(Debug, Serialize, Deserialize, FromRow, Clone)]
pub struct HistoricalData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub open: f64,           // 开盘价
    pub close: f64,          // 收盘价
    pub high: f64,           // 最高价
    pub low: f64,            // 最低价
    pub volume: i64,         // 成交量
    pub amount: f64,         // 成交额（元）
    pub amplitude: f64,      // 振幅（%）
    pub turnover_rate: f64,  //换手率（%）
    pub change_percent: f64, // 涨跌幅（%）
    pub change: f64,         // 涨跌额（元）
}

#[derive(Debug, Deserialize)]
pub struct HistoricalDataItem {
    #[serde(rename = "t")]
    pub date: String, // 日期
    #[serde(rename = "o")]
    pub open: f64, // 开盘价
    #[serde(rename = "h")]
    pub high: f64, // 最高价（元）
    #[serde(rename = "l")]
    pub low: f64, // 最低价（元）
    #[serde(rename = "c")]
    pub close: f64, // 收盘价（元）
    #[serde(rename = "v")]
    pub volume: f64, // 成交量（手）
    #[serde(rename = "a")]
    pub amount: f64, // 成交额（元）
    #[serde(rename = "pc")]
    pub pre_close: f64, // 前收盘价
}

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct RealtimeData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "name")]
    pub name: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub close: f64,          // 收盘价
    pub volume: i64,         // 成交量
    pub amount: f64,         // 成交额（元）
    pub amplitude: f64,      // 振幅（%）
    pub turnover_rate: f64,  //换手率（%）
    pub change_percent: f64, // 涨跌幅（%）
    pub change: f64,         // 涨跌额（元）
}
