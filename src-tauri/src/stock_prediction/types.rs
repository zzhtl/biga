use serde::{Deserialize, Serialize};

// 重新导出utils模块中的趋势分析类型
pub use crate::stock_prediction::utils::{TrendState, TrendAnalysis, VolumePricePredictionStrategy};

// 向后兼容性类型别名
pub type DirectionPredictionStrategy = VolumePricePredictionStrategy;

// 简化的模型配置结构体
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub target_date: String,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    pub confidence: f64,
    pub trading_signal: Option<String>,
    pub signal_strength: Option<f64>,
    pub technical_indicators: Option<TechnicalIndicatorValues>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorValues {
    pub rsi: f64,
    pub macd_histogram: f64,
    pub kdj_j: f64,
    pub cci: f64,
    pub obv_trend: f64, // OBV相对于均值的比例
    // 新增MACD和KDJ信号字段
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub macd_golden_cross: bool,  // MACD金叉
    pub macd_death_cross: bool,   // MACD死叉
    pub kdj_golden_cross: bool,   // KDJ金叉
    pub kdj_death_cross: bool,    // KDJ死叉
    pub kdj_overbought: bool,     // KDJ超买
    pub kdj_oversold: bool,       // KDJ超卖
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub prediction_days: usize,
    pub use_candle: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metadata: ModelInfo,
    pub accuracy: f64,
}

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



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLog {
    pub epoch: usize,
    pub loss: f64,
    pub timestamp: String,
    pub message: Option<String>,
}

// 历史数据结构体
#[derive(Debug, Clone)]
pub struct HistoricalDataType {
    pub date: String,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: i64,
    pub change_percent: f64,
}

// 技术信号结构体
#[derive(Debug, Clone)]
pub struct TechnicalSignals {
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub macd_histogram: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub kdj_j: f64,
    pub rsi: f64,
    pub cci: f64,
    pub obv: f64,
    pub signal: TradingSignal,
    pub signal_strength: f64,
    pub buy_signals: i32,
    pub sell_signals: i32,
    // 新增：交叉信号标记
    pub macd_golden_cross: bool,     // MACD金叉（DIF上穿DEA）
    pub macd_death_cross: bool,      // MACD死叉（DIF下穿DEA）
    pub kdj_golden_cross: bool,      // KDJ金叉（K上穿D）
    pub kdj_death_cross: bool,       // KDJ死叉（K下穿D）
    pub kdj_overbought: bool,        // KDJ超买
    pub kdj_oversold: bool,          // KDJ超卖
    pub macd_zero_cross_up: bool,    // MACD上穿零轴
    pub macd_zero_cross_down: bool,  // MACD下穿零轴
}

// 交易信号枚举
#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

// 涨跌方向分类预测结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    pub target_date: String,
    pub predicted_direction: String, // "上涨", "下跌", "横盘"
    pub direction_confidence: f64,   // 方向预测置信度
    pub predicted_price: f64,        // 预测价格（仅供参考）
    pub predicted_change_percent: f64,
    pub confidence: f64,
}

// 涨跌方向分类枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,    // 上涨 > 0.5%
    Down,  // 下跌 < -0.5%
    Flat,  // 横盘 [-0.5%, 0.5%]
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
}

// 预测结果包装结构体，包含预测和最新真实数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    pub predictions: Vec<Prediction>,
    pub last_real_data: Option<LastRealData>,
}

// 最新真实数据结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LastRealData {
    pub date: String,
    pub price: f64,
    pub change_percent: f64,
}

// 历史波动特征结构体
#[derive(Debug)]
pub struct HistoricalVolatilityFeatures {
    pub avg_daily_change: f64,         // 平均日涨跌幅(绝对值)
    pub avg_up_change: f64,            // 平均上涨幅度
    pub avg_down_change: f64,          // 平均下跌幅度
    pub max_consecutive_up: usize,     // 最大连续上涨天数
    pub max_consecutive_down: usize,   // 最大连续下跌天数
    pub up_down_ratio: f64,            // 上涨/下跌天数比例
    pub volatility_pattern: String,    // 波动模式描述
} 