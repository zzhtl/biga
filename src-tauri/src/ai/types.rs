use crate::{db::models::{HistoricalData, StockFundamental}, services::market_data::DataReceipt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Provider { Claude, Codex }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Transport { Local, Ssh }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConnectionConfig {
    pub id: String,
    pub name: String,
    pub provider: Provider,
    pub transport: Transport,
    pub executable: String,
    pub model: Option<String>,
    pub host: Option<String>,
    pub user: Option<String>,
    pub port: Option<u16>,
    pub identity_file: Option<String>,
}

impl AiConnectionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.name.len() > 100 { return Err("连接名称不能为空且不能超过 100 字节".into()); }
        if self.executable.is_empty() || self.executable.starts_with('-') || self.executable.contains(['\0', '\n', '\r']) { return Err("CLI 路径无效".into()); }
        if self.model.as_ref().is_some_and(|v| v.len() > 200 || v.contains(['\0', '\n', '\r'])) { return Err("模型名称无效".into()); }
        if self.transport == Transport::Ssh {
            let host = self.host.as_deref().unwrap_or("");
            if host.is_empty() || host.starts_with('-') || !host.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-:[]".contains(&b)) { return Err("SSH 主机或别名无效".into()); }
            if let Some(user) = &self.user {
                if !user.is_empty() && (user.starts_with('-') || !user.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))) { return Err("SSH 用户名无效".into()); }
            }
            if self.port == Some(0) { return Err("SSH 端口必须大于 0".into()); }
            if self.identity_file.as_ref().is_some_and(|v| v.contains(['\0', '\n', '\r'])) { return Err("SSH 密钥路径无效".into()); }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPredictionRequest {
    pub symbol: String,
    pub connection_id: String,
    pub days: usize,
    #[serde(default)]
    pub regenerate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub data_hash: String,
    pub symbol: String,
    pub as_of: String,
    pub bars: Vec<HistoricalData>,
    pub valuation: Option<Value>,
    pub fundamentals: Vec<StockFundamental>,
    pub technical: Value,
    pub sources: Vec<DataReceipt>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForecastPoint {
    pub day: usize,
    pub lower: f64,
    pub expected: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub text: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assessment {
    pub summary: String,
    pub direction: String,
    pub factors: Vec<Evidence>,
    pub bullish: String,
    pub base: String,
    pub bearish: String,
    pub risks: Vec<String>,
    pub invalidation: Vec<String>,
    pub forecast: Vec<ForecastPoint>,
}

impl Assessment {
    pub fn validate(&self, days: usize, available: &[&str]) -> Result<(), String> {
        if self.summary.trim().is_empty() || !["up", "down", "sideways"].contains(&self.direction.as_str()) || self.forecast.len() != days { return Err("AI 报告缺少有效摘要、方向或完整预测点".into()); }
        if self.factors.is_empty() || self.risks.is_empty() || self.invalidation.is_empty() || [&self.bullish, &self.base, &self.bearish].iter().any(|s| s.trim().is_empty()) { return Err("AI 报告缺少分析依据、情景或风险条件".into()); }
        for (i, point) in self.forecast.iter().enumerate() {
            if point.day != i + 1 || ![point.lower, point.expected, point.upper].iter().all(|v| v.is_finite() && *v > 0.0) || point.lower > point.expected || point.expected > point.upper { return Err("AI 预测点的日期顺序、价格或情景上下界无效".into()); }
        }
        for factor in &self.factors {
            if factor.text.trim().is_empty() || factor.sources.is_empty() || factor.sources.iter().any(|s| !available.contains(&s.as_str())) { return Err("AI 引用了本次分析未提供的数据".into()); }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataQuery { pub resource: String, pub history_days: usize }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AiEnvelope { pub kind: String, pub queries: Vec<DataQuery>, pub report: Option<Assessment> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiPredictionReport {
    pub connection_name: String,
    pub provider: Provider,
    pub model: String,
    pub generated_at: String,
    pub snapshot: Snapshot,
    pub assessment: Assessment,
    pub dates: Vec<String>,
    pub usage: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AiJob {
    pub id: String,
    pub symbol: String,
    pub connection_id: String,
    pub status: String,
    pub stage: String,
    pub created_at: String,
    pub error: Option<String>,
    pub requests: i64,
    pub cache_hits: i64,
    pub report_json: Option<String>,
}

pub fn output_schema() -> Value {
    fn object(properties: Value) -> Value {
        let required: Vec<_> = properties.as_object().expect("schema properties are objects").keys().cloned().collect();
        json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})
    }
    let text = json!({"type":"string"});
    let strings = json!({"type":"array","items":text});
    let point = object(json!({"day":{"type":"integer"},"lower":{"type":"number"},"expected":{"type":"number"},"upper":{"type":"number"}}));
    let factor = object(json!({"text":text,"sources":{"type":"array","items":{"type":"string","enum":["history","valuation","fundamentals","technical"]}}}));
    let report = object(json!({"summary":text,"direction":{"type":"string","enum":["up","down","sideways"]},"factors":{"type":"array","items":factor},"bullish":text,"base":text,"bearish":text,"risks":strings,"invalidation":strings,"forecast":{"type":"array","items":point}}));
    object(json!({"kind":{"type":"string","enum":["query","report"]},"queries":{"type":"array","items":object(json!({"resource":{"type":"string","enum":["history","valuation","fundamentals","technical"]},"history_days":{"type":"integer"}}))},"report":{"anyOf":[report,{"type":"null"}]}}))
}
