//! 机器学习模型模块
//! 
//! 提供模型训练、预测、评估和管理功能

pub mod training;
pub mod inference;
pub mod management;

pub use training::*;
pub use inference::*;
pub use management::*;

