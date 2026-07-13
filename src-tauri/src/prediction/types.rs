//! 预测模块类型定义

use serde::{Deserialize, Serialize};
use crate::prediction::analysis::{PatternRecognition, SupportResistance};
use crate::prediction::strategy::{MultiFactorScore, MultiTimeframeSignal};

// =============================================================================
// 预测请求/响应类型
// =============================================================================

/// 训练请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub stock_code: String,
    pub model_name: String,
    pub start_date: String,
    pub end_date: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub model_type: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub dropout: f64,
    pub train_test_split: f64,
}

/// 预测请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub prediction_days: usize,
    pub use_candle: bool,
}

/// 纯技术分析请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalOnlyRequest {
    pub stock_code: String,
    pub history_days: Option<usize>,
    pub prediction_days: usize,
}

// =============================================================================
// 预测结果类型
// =============================================================================

/// 单日预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub target_date: String,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    /// 信号强度（0.25–0.92），非"方向正确概率"；方向不可预测，真实不确定性见 `interval`。
    pub confidence: f64,
    pub trading_signal: Option<String>,
    pub signal_strength: Option<f64>,
    pub technical_indicators: Option<TechnicalIndicatorValues>,
    pub prediction_reason: Option<String>,
    pub key_factors: Option<Vec<String>>,
    /// 校准涨跌区间带（方向不可测但波动可测；点预测仅供参考，区间才是诚实的不确定性）
    #[serde(default)]
    pub interval: Option<PredictionInterval>,
    /// 95% 压力区间，用于观察低概率但影响较大的尾部波动。
    #[serde(default)]
    pub stress_interval: Option<PredictionInterval>,
}

/// 校准涨跌区间带。
///
/// 由近 20 日已实现波动率 × √预测天数构造，z 倍数经 `examples/interval_calibration.rs`
/// 在多票 walk-forward 上校准到目标覆盖率（如名义 80% 带经验覆盖 ~80%）。
/// 这是对"单股方向无 alpha、但波动率可预测"事实的诚实表达。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    /// 名义覆盖率（如 0.80 表示历史走步样本经验覆盖目标约 80%）
    pub confidence: f64,
    /// 区间下沿：相对发起日真实价的累计涨跌幅（%）
    pub lower_change_percent: f64,
    /// 区间上沿：相对发起日真实价的累计涨跌幅（%）
    pub upper_change_percent: f64,
    /// 区间下沿价格
    pub lower_price: f64,
    /// 区间上沿价格
    pub upper_price: f64,
    /// 区间生成方法，便于前端和回测追溯口径。
    #[serde(default)]
    pub method: String,
    /// 波动率估计使用的历史窗口（交易日）。
    #[serde(default)]
    pub lookback_days: usize,
}

/// 技术指标值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorValues {
    pub rsi: f64,
    pub macd_histogram: f64,
    pub kdj_j: f64,
    pub cci: f64,
    pub obv_trend: f64,
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub macd_golden_cross: bool,
    pub macd_death_cross: bool,
    pub kdj_golden_cross: bool,
    pub kdj_death_cross: bool,
    pub kdj_overbought: bool,
    pub kdj_oversold: bool,
}

/// 预测响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub predictions: Vec<Prediction>,
    pub last_real_data: Option<LastRealData>,
    /// 预测口径、风险事实与不确定性诊断。旧响应反序列化时允许缺省。
    #[serde(default)]
    pub diagnostics: Option<PredictionDiagnostics>,
}

/// 风险等级。它表示已触发事实规则的最高严重度，不是风险发生概率。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    #[default]
    Low,
    Medium,
    High,
}

impl RiskLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "低风险",
            Self::Medium => "中风险",
            Self::High => "高风险",
        }
    }
}

/// 风险来源分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskCategory {
    Data,
    Uncertainty,
    Volatility,
    Trend,
    Signal,
    Liquidity,
    Model,
}

/// 单条可追溯风险告警。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskWarning {
    /// 稳定代码，供前端去重、筛选和排序。
    pub code: String,
    pub category: RiskCategory,
    pub severity: RiskLevel,
    pub title: String,
    pub detail: String,
    /// 触发该告警的可核验数值或状态。
    pub evidence: Vec<String>,
}

/// 风险面板使用的原始度量，不合成为未经校准的风险分数。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub history_samples: usize,
    pub data_staleness_days: Option<i64>,
    pub daily_volatility_percent: f64,
    pub volatility_percentile: f64,
    pub interval_80_width_percent: Option<f64>,
    pub interval_80_lower_percent: Option<f64>,
    pub stress_95_lower_percent: Option<f64>,
    pub support_distance_percent: Option<f64>,
    pub resistance_distance_percent: Option<f64>,
    pub atr_percent: Option<f64>,
}

/// 当前预测的风险汇总。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSummary {
    pub level: RiskLevel,
    pub level_label: String,
    pub warnings: Vec<RiskWarning>,
    pub metrics: RiskMetrics,
}

impl Default for RiskSummary {
    fn default() -> Self {
        Self {
            level: RiskLevel::Low,
            level_label: RiskLevel::Low.label().to_string(),
            warnings: Vec::new(),
            metrics: RiskMetrics::default(),
        }
    }
}

/// 预测诊断元数据。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictionDiagnostics {
    /// `historical_unconditional_drift` 或 `candle_model`。
    pub point_estimate_kind: String,
    pub point_estimate_note: String,
    pub uncertainty_method: String,
    pub risk_summary: RiskSummary,
}

/// 最新真实数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LastRealData {
    pub date: String,
    pub price: f64,
    pub change_percent: f64,
}

// =============================================================================
// 模型相关类型
// =============================================================================

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub stock_code: String,
    pub created_at: u64,
    pub model_type: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub accuracy: f64,
    pub training_start_date: Option<String>,
    pub training_end_date: Option<String>,
    pub training_samples: Option<usize>,
    pub test_samples: Option<usize>,
    pub mae: Option<f64>,
    pub rmse: Option<f64>,
}

/// 训练结果
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metadata: ModelInfo,
    pub accuracy: f64,
    pub test_samples: usize,
    pub mae: f64,
    pub rmse: f64,
}

/// 评估结果
#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub model_id: String,
    pub model_name: String,
    pub stock_code: String,
    pub test_samples: usize,
    pub accuracy: f64,
    pub direction_accuracy: f64,
    pub mse: f64,
    pub mae: f64,
    pub rmse: f64,
    pub evaluation_date: String,
    pub evaluation_scope: String,
    pub evaluation_note: String,
}

// =============================================================================
// 回测相关类型
// =============================================================================

/// 回测请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub start_date: String,
    pub end_date: String,
    pub prediction_days: usize,
    pub backtest_interval: usize,
}

/// 回测报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub stock_code: String,
    pub model_name: String,
    pub backtest_period: String,
    pub total_predictions: usize,
    pub backtest_entries: Vec<BacktestEntry>,
    pub overall_price_accuracy: f64,
    pub overall_direction_accuracy: f64,
    pub average_prediction_error: f64,
    pub accuracy_trend: Vec<f64>,
    pub daily_accuracy: Vec<DailyAccuracy>,
    pub price_error_distribution: Vec<f64>,
    pub direction_correct_rate: f64,
    pub volatility_vs_accuracy: Vec<(f64, f64)>,
    #[serde(default)]
    pub rmse: f64,
    #[serde(default)]
    pub baseline_direction_accuracy: f64,
    #[serde(default)]
    pub direction_edge: f64,
    #[serde(default)]
    pub predicted_up_ratio: f64,
    #[serde(default)]
    pub actual_up_ratio: f64,
    #[serde(default)]
    pub interval_80_samples: usize,
    #[serde(default)]
    pub interval_80_coverage: f64,
    #[serde(default)]
    pub stress_95_samples: usize,
    #[serde(default)]
    pub stress_95_coverage: f64,
    #[serde(default)]
    pub average_interval_80_width: f64,
    #[serde(default)]
    pub average_stress_95_width: f64,
}

/// 单次回测记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestEntry {
    pub prediction_date: String,
    pub predictions: Vec<Prediction>,
    pub actual_prices: Vec<f64>,
    pub actual_changes: Vec<f64>,
    pub price_accuracy: f64,
    pub direction_accuracy: f64,
    pub avg_prediction_error: f64,
}

/// 按日回测准确率
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyAccuracy {
    pub date: String,
    pub price_accuracy: f64,
    pub direction_accuracy: f64,
    pub prediction_count: usize,
    pub market_volatility: f64,
}

// =============================================================================
// 方向枚举
// =============================================================================

/// 涨跌方向
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Flat,
}

impl Direction {
    pub fn from_change_percent(change: f64) -> Self {
        if change > 0.5 {
            Direction::Up
        } else if change < -0.5 {
            Direction::Down
        } else {
            Direction::Flat
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            Direction::Up => "上涨".to_string(),
            Direction::Down => "下跌".to_string(),
            Direction::Flat => "横盘".to_string(),
        }
    }
}

// =============================================================================
// 专业预测类型
// =============================================================================

/// 买卖点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuySellPoint {
    pub point_type: String,
    pub signal_strength: f64,
    pub price_level: f64,
    pub stop_loss: f64,
    pub take_profit: Vec<f64>,
    pub risk_reward_ratio: f64,
    pub reasons: Vec<String>,
    pub confidence: f64,
}

/// 专业预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPrediction {
    pub buy_points: Vec<BuySellPoint>,
    pub sell_points: Vec<BuySellPoint>,
    pub support_resistance: SupportResistance,
    pub multi_timeframe: MultiTimeframeSignal,
    pub divergence: VolumePriceDivergence,
    pub current_advice: String,
    pub risk_level: String,
    pub candle_patterns: Vec<PatternRecognition>,
    pub volume_analysis: VolumeAnalysisInfo,
    pub multi_factor_score: MultiFactorScore,
}

/// 量价/指标背离概要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePriceDivergence {
    pub has_bullish_divergence: bool,
    pub has_bearish_divergence: bool,
    pub divergence_strength: f64,
    pub warning_message: String,
}

/// 成交量分析概要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnalysisInfo {
    pub volume_trend: String,
    pub volume_price_sync: bool,
    pub accumulation_signal: f64,
    pub obv_trend: String,
}

/// 专业预测响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPredictionResponse {
    pub predictions: PredictionResponse,
    pub professional_analysis: ProfessionalPrediction,
}
