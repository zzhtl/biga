// 数据类型和结构定义
pub mod types;

// 数据库操作
pub mod database;

// 工具函数
pub mod utils;

// 技术指标计算
pub mod technical_indicators;

// 技术分析和信号生成
pub mod technical_analysis;

// 模型管理功能
pub mod model_management;

// 训练功能
pub mod training;

// 预测功能
pub mod prediction;

// 评估功能
pub mod evaluation;

// 主要的预测功能（保持向后兼容）
pub mod candle_prediction;

// 重新导出主要的公共类型和函数
pub use types::{
    ModelConfig, ModelInfo, Prediction, TechnicalIndicatorValues,
    TrainingRequest, PredictionRequest, TrainingResult, EvaluationResult,
    PredictionResponse, LastRealData
};

pub use model_management::{list_models, delete_model};
pub use training::train_candle_model;
pub use prediction::{predict_with_candle, predict_with_simple_strategy};
pub use candle_prediction::{evaluate_candle_model, retrain_candle_model}; 