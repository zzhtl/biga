//! 系统常量配置

// =============================================================================
// 数据库配置
// =============================================================================

/// 批量插入大小
pub const BATCH_SIZE: usize = 500;

// =============================================================================
// 技术指标参数
// =============================================================================

/// RSI 周期
pub const RSI_PERIOD: usize = 14;
/// MACD 快线周期
pub const MACD_FAST_PERIOD: usize = 12;
/// MACD 慢线周期
pub const MACD_SLOW_PERIOD: usize = 26;
/// MACD 信号线周期
pub const MACD_SIGNAL_PERIOD: usize = 9;
/// KDJ 周期
pub const KDJ_PERIOD: usize = 9;
/// 布林带周期
pub const BOLLINGER_PERIOD: usize = 20;
/// 布林带标准差倍数
pub const BOLLINGER_STD_DEV: f64 = 2.0;
/// CCI 周期
pub const CCI_PERIOD: usize = 20;
/// ATR 周期
pub const ATR_PERIOD: usize = 14;

// =============================================================================
// 均线参数
// =============================================================================

/// MA5 周期
pub const MA5_PERIOD: usize = 5;
/// MA10 周期
pub const MA10_PERIOD: usize = 10;
/// MA20 周期
pub const MA20_PERIOD: usize = 20;
/// MA60 周期
pub const MA60_PERIOD: usize = 60;

// =============================================================================
// 预测配置
// =============================================================================

/// 最小历史数据天数
pub const MIN_HISTORICAL_DAYS: usize = 120;
/// 推荐历史数据天数
pub const RECOMMENDED_HISTORICAL_DAYS: usize = 180;
/// 最佳历史数据天数
pub const OPTIMAL_HISTORICAL_DAYS: usize = 250;

/// 默认波动率
pub const DEFAULT_VOLATILITY: f64 = 0.02;
/// 最大波动率限制
pub const MAX_VOLATILITY: f64 = 0.10;

/// A股涨停限制 (%)
pub const A_STOCK_LIMIT_UP: f64 = 10.0;
/// A股跌停限制 (%)
pub const A_STOCK_LIMIT_DOWN: f64 = -10.0;

// =============================================================================
// 信号阈值
// =============================================================================

/// RSI 超买阈值
pub const RSI_OVERBOUGHT: f64 = 70.0;
/// RSI 超卖阈值
pub const RSI_OVERSOLD: f64 = 30.0;
/// KDJ 超买阈值
pub const KDJ_OVERBOUGHT: f64 = 80.0;
/// KDJ 超卖阈值
pub const KDJ_OVERSOLD: f64 = 20.0;

// =============================================================================
// 模型训练参数
// =============================================================================

/// 默认隐藏层大小
pub const DEFAULT_HIDDEN_SIZE: usize = 128;
/// 默认 Dropout 率
pub const DEFAULT_DROPOUT: f64 = 0.2;
/// 最小 Dropout 率
pub const MIN_DROPOUT: f64 = 0.1;
/// 早停耐心值
pub const EARLY_STOPPING_PATIENCE: usize = 15;
/// 最小改进阈值
pub const MIN_IMPROVEMENT_DELTA: f64 = 0.0001;
/// 学习率衰减因子
pub const LR_DECAY_FACTOR: f64 = 0.95;
/// 学习率衰减周期
pub const LR_DECAY_EPOCHS: usize = 20;
/// L2 正则化系数
pub const L2_LAMBDA: f64 = 0.0001;

