//! 预测模块类型定义

use serde::{Deserialize, Serialize};

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
    pub confidence: f64,
    pub trading_signal: Option<String>,
    pub signal_strength: Option<f64>,
    pub technical_indicators: Option<TechnicalIndicatorValues>,
    pub prediction_reason: Option<String>,
    pub key_factors: Option<Vec<String>>,
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
}

/// 训练结果
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metadata: ModelInfo,
    pub accuracy: f64,
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
    pub overall_price_accuracy: f64,
    pub overall_direction_accuracy: f64,
    pub average_prediction_error: f64,
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
    pub accuracy_rate: Option<f64>,
}

/// 专业预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPrediction {
    pub buy_points: Vec<BuySellPoint>,
    pub sell_points: Vec<BuySellPoint>,
    pub current_advice: String,
    pub risk_level: String,
}

/// 专业预测响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfessionalPredictionResponse {
    pub predictions: PredictionResponse,
    pub professional_analysis: ProfessionalPrediction,
}

