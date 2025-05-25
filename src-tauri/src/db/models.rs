use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use chrono::NaiveDateTime;

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
    pub volume: i64, // 成交量（手）
    #[serde(rename = "a")]
    pub amount: f64, // 成交额（元）
    #[serde(rename = "pc")]
    pub pre_close: f64, // 前收盘价
    #[serde(rename = "sf")]
    pub stock_flag: f64, // 股票标识
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

#[derive(Debug, Serialize, Deserialize, FromRow, Clone)]
pub struct StockPredictionModel {
    pub id: i64,
    pub symbol: String,
    pub model_name: String,
    pub model_type: String,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    #[serde(skip_serializing)]
    pub model_data: Vec<u8>,
    pub parameters: String,
    pub metrics: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StockPredictionModelInfo {
    pub id: i64,
    pub symbol: String,
    pub model_name: String,
    pub model_type: String,
    pub created_at: NaiveDateTime,
    pub updated_at: NaiveDateTime,
    pub parameters: String,
    pub metrics: String,
}

#[derive(Debug, Serialize, Deserialize, FromRow, Clone)]
pub struct StockPrediction {
    pub id: i64,
    pub symbol: String,
    pub model_id: i64,
    pub prediction_date: NaiveDate,
    pub target_date: NaiveDate,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    pub confidence: f64,
    pub features_used: String,
    pub created_at: NaiveDateTime,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub symbol: String,
    pub model_name: Option<String>,
    pub days_to_predict: i32,
    pub train_test_split: Option<f64>,
    pub lookback_days: Option<i32>,
    pub features: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionModelConfig {
    pub model_name: String,
    pub model_type: String,
    pub parameters: serde_json::Value,
    pub features: Vec<String>,
    pub lookback_days: i32,
    pub train_test_split: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PredictionResult {
    pub symbol: String,
    pub target_date: NaiveDate,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    pub confidence: f64,
    pub model_info: StockPredictionModelInfo,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelEvaluationMetrics {
    pub rmse: f64,
    pub mae: f64,
    pub r_squared: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}
