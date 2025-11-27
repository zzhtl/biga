//! 配置模块
//! 
//! 集中管理所有配置参数，包括：
//! - 预测权重配置
//! - 技术指标参数
//! - 系统常量

pub mod weights;
pub mod constants;

pub use weights::*;
pub use constants::*;

