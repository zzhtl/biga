//! 机器学习模型模块
//! 
//! 提供模型训练、预测、评估和管理功能

pub mod training;
pub mod inference;
pub mod management;
pub mod features;
pub mod network;
pub mod ml_inference;

pub const HORIZON_AWARE_MODEL_TYPE: &str = "candle_mlp_horizon";

pub use training::*;
pub use inference::*;
pub use management::*;
