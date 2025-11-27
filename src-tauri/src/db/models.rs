//! 数据模型定义

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

// =============================================================================
// 股票基本信息
// =============================================================================

/// 股票基本信息
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct StockInfo {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    pub name: String,
    pub exchange: String,
}

/// 股票基本信息（API响应格式）
#[derive(Debug, Serialize, Deserialize)]
pub struct StockInfoItem {
    #[serde(rename = "dm")]
    pub symbol: String,
    #[serde(rename = "mc")]
    pub name: String,
    #[serde(rename = "jys")]
    pub exchange: String,
}

impl From<StockInfoItem> for StockInfo {
    fn from(item: StockInfoItem) -> Self {
        Self {
            symbol: item.symbol,
            name: item.name,
            exchange: item.exchange,
        }
    }
}

/// 股票详细信息
#[derive(Default, Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Stock {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    pub name: String,
    pub area: String,
    pub industry: String,
    pub market: String,
    #[serde(rename = "ts_code")]
    #[sqlx(rename = "exchange")]
    pub exchange: String,
    pub list_date: String,
    pub act_name: String,
    pub act_ent_type: String,
}

// =============================================================================
// 历史数据
// =============================================================================

/// 股票历史数据
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct HistoricalData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: i64,
    pub amount: f64,
    pub amplitude: f64,
    pub turnover_rate: f64,
    pub change_percent: f64,
    pub change: f64,
}

/// 历史数据（API响应格式）
#[derive(Debug, Deserialize)]
pub struct HistoricalDataItem {
    #[serde(rename = "t")]
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
    pub volume: f64,
    #[serde(rename = "a")]
    pub amount: f64,
    #[serde(rename = "pc")]
    pub pre_close: f64,
}

impl HistoricalDataItem {
    /// 转换为历史数据模型
    pub fn to_historical_data(&self, symbol: &str) -> Option<HistoricalData> {
        let date = chrono::NaiveDate::parse_from_str(&self.date, "%Y-%m-%d").ok()?;
        let change = self.close - self.pre_close;
        let change_percent = if self.pre_close > 0.0 {
            change / self.pre_close * 100.0
        } else {
            0.0
        };
        let amplitude = if self.pre_close > 0.0 {
            (self.high - self.low) / self.pre_close * 100.0
        } else {
            0.0
        };

        Some(HistoricalData {
            symbol: symbol.to_string(),
            date,
            open: self.open,
            close: self.close,
            high: self.high,
            low: self.low,
            volume: (self.volume * 100.0) as i64, // 手转股
            amount: self.amount,
            amplitude,
            turnover_rate: 0.0, // API未提供
            change_percent,
            change,
        })
    }
}

// =============================================================================
// 实时数据
// =============================================================================

/// 实时行情数据
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RealtimeData {
    #[sqlx(rename = "symbol")]
    pub symbol: String,
    #[sqlx(rename = "name")]
    pub name: String,
    #[sqlx(rename = "date")]
    pub date: NaiveDate,
    pub close: f64,
    pub volume: i64,
    pub amount: f64,
    pub amplitude: f64,
    pub turnover_rate: f64,
    pub change_percent: f64,
    pub change: f64,
}

// =============================================================================
// 预测模型相关
// =============================================================================

/// 预测模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionModelInfo {
    pub id: String,
    pub name: String,
    pub stock_code: String,
    pub created_at: u64,
    pub model_type: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub accuracy: f64,
}
