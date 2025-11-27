//! 股票预测核心模块
//! 
//! 提供完整的股票预测功能，包括：
//! - 技术指标计算
//! - 趋势/量价/形态分析
//! - 多因子评分策略
//! - 机器学习模型

pub mod types;
pub mod indicators;
pub mod analysis;
pub mod strategy;
pub mod model;

// 重新导出常用类型
pub use types::*;
pub use indicators::{TechnicalIndicatorValues as IndicatorValues, TradingSignal, calculate_all_indicators};
pub use analysis::{TrendState, TrendAnalysis, VolumePriceSignal, PatternRecognition, SupportResistance};
pub use strategy::{MultiFactorScore, MultiTimeframeSignal};

