// ============================================================================
// 金融级股票预测系统模块
// ============================================================================

// 核心模块
pub mod types;
pub mod database;
pub mod utils;

// 金融预测引擎（新）
pub mod financial_prediction_engine;

// 风险管理（新）
pub mod risk_management;

// 简化的权重系统（新）
pub mod core_weights_simplified;

// 优化的技术指标（新）
pub mod technical_indicators_optimized;

// 简化的预测接口（新）
pub mod prediction_simplified;

// 保留的核心模块
pub mod prediction;
pub mod training;
pub mod evaluation;
pub mod model_management;

// 保留的分析模块
pub mod technical_analysis;
pub mod volume_analysis;
pub mod candlestick_patterns;
pub mod backtest;

// 向后兼容
pub mod candle_prediction;

// 以下模块保留但标记为将被替换
pub mod core_weights;  // 将被 core_weights_simplified 替代
pub mod technical_indicators;  // 将被 technical_indicators_optimized 替代
pub mod constants;  // 将被整合到 financial_prediction_engine
pub mod multi_timeframe_analysis;  // 将被整合
pub mod multi_factor_scoring;  // 将被整合

// 以下复杂模块暂时保留，后续删除
pub mod ensemble_learning;
pub mod advanced_features;
pub mod feature_optimization;
pub mod hyperparameter_optimization;