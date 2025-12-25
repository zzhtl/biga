//! 股票预测系统 - 权重配置
//! 
//! 本文件集中管理所有预测权重参数，按功能分组
//! 
//! 优化说明 (v2.0 - 华尔街量化策略优化):
//! - 增强趋势和动量因子在趋势市的权重
//! - 增强形态和支撑阻力因子在震荡市的权重
//! - 添加背离检测相关权重
//! - 优化信号确认阈值

// =============================================================================
// 一、预测基础权重 (优化版)
// =============================================================================

/// 趋势因子占比（增强）
pub const PREDICTION_TREND_RATIO: f64 = 0.28;
/// 技术指标占比
pub const PREDICTION_TECHNICAL_RATIO: f64 = 0.32;
/// 均线量能占比
pub const PREDICTION_MA_VOLUME_RATIO: f64 = 0.25;
/// 市场波动占比（增强）
pub const PREDICTION_MARKET_FLUCTUATION_RATIO: f64 = 0.08;
/// 基础模型占比
pub const PREDICTION_BASE_MODEL_RATIO: f64 = 0.07;

// =============================================================================
// 二、多因子评分影响系数 (优化版)
// =============================================================================

/// 强烈看涨影响系数 (评分>78分) - 增强
pub const MULTI_FACTOR_STRONG_BULLISH_IMPACT: f64 = 0.0022;
/// 看涨影响系数 (评分60-78分) - 增强
pub const MULTI_FACTOR_BULLISH_IMPACT: f64 = 0.0012;
/// 看跌影响系数 (评分22-40分) - 增强
pub const MULTI_FACTOR_BEARISH_IMPACT: f64 = 0.0012;
/// 强烈看跌影响系数 (评分<22分) - 增强
pub const MULTI_FACTOR_STRONG_BEARISH_IMPACT: f64 = 0.0022;
/// 中性区间基准偏置（轻微看涨偏向，符合长期市场特征）
pub const MULTI_FACTOR_NEUTRAL_BIAS: f64 = 0.0002;

// =============================================================================
// 三、趋势一致性增强系数
// =============================================================================

/// 强一致性趋势偏置值
pub const STRONG_ALIGNMENT_BIAS: f64 = 0.004;
/// 强一致性反向预测抑制系数
pub const STRONG_ALIGNMENT_OPPOSITE_SUPPRESS: f64 = 0.35;
/// 中等一致性趋势偏置值
pub const MEDIUM_ALIGNMENT_BIAS: f64 = 0.002;
/// 中等一致性反向预测抑制系数
pub const MEDIUM_ALIGNMENT_OPPOSITE_SUPPRESS: f64 = 0.50;
/// 弱一致性趋势偏置值
pub const WEAK_ALIGNMENT_BIAS: f64 = 0.001;
/// 弱一致性反向预测抑制系数
pub const WEAK_ALIGNMENT_OPPOSITE_SUPPRESS: f64 = 0.60;
/// 最弱一致性反向预测抑制系数
pub const MINIMAL_ALIGNMENT_OPPOSITE_SUPPRESS: f64 = 0.75;
/// 最弱一致性趋势偏置值
pub const MINIMAL_ALIGNMENT_BIAS: f64 = 0.0005;
/// 中性趋势金叉死叉偏置
pub const NEUTRAL_CROSS_BIAS: f64 = 0.0015;

// =============================================================================
// 四、支撑压力位影响系数
// =============================================================================

/// 接近强压力位衰减系数
pub const NEAR_STRONG_RESISTANCE_DECAY: f64 = 0.50;
/// 接近压力位衰减系数
pub const NEAR_RESISTANCE_DECAY: f64 = 0.70;
/// 接近强支撑位增强系数
pub const NEAR_STRONG_SUPPORT_PROTECTION: f64 = 0.50;
/// 接近支撑位增强系数
pub const NEAR_SUPPORT_PROTECTION: f64 = 0.70;
/// 突破压力位加速系数
pub const BREAKOUT_ACCELERATION: f64 = 1.30;
/// 跌破支撑位加速系数
pub const BREAKDOWN_ACCELERATION: f64 = 1.30;

// =============================================================================
// 五、波动率与市场情绪调节系数
// =============================================================================

/// 高波动率抑制系数
pub const HIGH_VOLATILITY_SUPPRESS: f64 = 0.80;
/// 极高波动率抑制系数
pub const EXTREME_VOLATILITY_SUPPRESS: f64 = 0.60;
/// 低波动率增强系数
pub const LOW_VOLATILITY_ENHANCE: f64 = 1.15;
/// 极度恐慌增强系数
pub const EXTREME_FEAR_CONTRARIAN_BOOST: f64 = 1.25;
/// 极度贪婪抑制系数
pub const EXTREME_GREED_SUPPRESS: f64 = 0.75;
/// 恐慌期反转加成
pub const PANIC_REVERSAL_BONUS: f64 = 0.005;
/// 过热期回调加成
pub const OVERHEATED_CORRECTION_BONUS: f64 = 0.005;

// =============================================================================
// 六、方向投票权重
// =============================================================================

/// 强趋势投票分
pub const STRONG_TREND_VOTE_SCORE: f64 = 2.0;
/// 普通趋势投票分
pub const NORMAL_TREND_VOTE_SCORE: f64 = 1.0;
/// MACD金叉死叉投票分
pub const MACD_CROSS_VOTE_SCORE: f64 = 1.2;
/// MACD柱状图投票分
pub const MACD_HISTOGRAM_VOTE_SCORE: f64 = 0.6;
/// MACD零轴穿越投票分
pub const MACD_ZERO_CROSS_VOTE_SCORE: f64 = 0.8;
/// KDJ金叉死叉投票分
pub const KDJ_CROSS_VOTE_SCORE: f64 = 0.8;
/// KDJ超买超卖投票分
pub const KDJ_EXTREME_VOTE_SCORE: f64 = 0.6;
/// RSI强烈超买超卖投票分
pub const RSI_EXTREME_VOTE_SCORE: f64 = 0.8;
/// RSI一般偏离投票分
pub const RSI_MODERATE_VOTE_SCORE: f64 = 0.3;
/// 方向投票 - 强确认阈值
pub const DIRECTION_STRONG_CONFIRM_THRESHOLD: f64 = 5.0;
/// 方向投票 - 弱确认阈值
pub const DIRECTION_WEAK_CONFIRM_THRESHOLD: f64 = 3.0;
/// 方向高确认增强系数
pub const DIRECTION_STRONG_CONFIRM_BOOST: f64 = 1.20;
/// 方向弱确认抑制系数
pub const DIRECTION_WEAK_CONFIRM_SUPPRESS: f64 = 0.85;

// =============================================================================
// 七、多因子综合评分权重 (优化版 - 华尔街量化策略)
// =============================================================================

/// 趋势因子权重（核心因子，增强）
pub const TREND_FACTOR_WEIGHT: f64 = 0.24;
/// 量价因子权重（增强，量价配合很重要）
pub const VOLUME_PRICE_FACTOR_WEIGHT: f64 = 0.20;
/// 多周期共振因子权重
pub const MULTI_TIMEFRAME_FACTOR_WEIGHT: f64 = 0.14;
/// 动量因子权重（增强）
pub const MOMENTUM_FACTOR_WEIGHT: f64 = 0.16;
/// K线形态因子权重
pub const PATTERN_FACTOR_WEIGHT: f64 = 0.10;
/// 支撑压力因子权重
pub const SUPPORT_RESISTANCE_FACTOR_WEIGHT: f64 = 0.08;
/// 市场情绪因子权重
pub const SENTIMENT_FACTOR_WEIGHT: f64 = 0.05;
/// 波动率因子权重
pub const VOLATILITY_FACTOR_WEIGHT: f64 = 0.03;

// =============================================================================
// 新增：背离检测权重
// =============================================================================

/// RSI背离权重
pub const RSI_DIVERGENCE_WEIGHT: f64 = 0.35;
/// MACD背离权重
pub const MACD_DIVERGENCE_WEIGHT: f64 = 0.40;
/// OBV背离权重
pub const OBV_DIVERGENCE_WEIGHT: f64 = 0.25;

// =============================================================================
// 新增：信号确认阈值
// =============================================================================

/// 强确认所需信号数
pub const STRONG_CONFIRMATION_SIGNALS: i32 = 4;
/// 中等确认所需信号数
pub const MODERATE_CONFIRMATION_SIGNALS: i32 = 3;
/// 弱确认所需信号数
pub const WEAK_CONFIRMATION_SIGNALS: i32 = 2;

// =============================================================================
// 新增：市场状态自适应系数
// =============================================================================

/// 趋势市趋势因子放大系数
pub const TREND_MARKET_TREND_AMPLIFIER: f64 = 1.4;
/// 震荡市支撑阻力放大系数
pub const RANGE_MARKET_SR_AMPLIFIER: f64 = 1.5;
/// 转折点情绪因子放大系数
pub const TURNING_POINT_SENTIMENT_AMPLIFIER: f64 = 1.4;

// =============================================================================
// 八、技术指标影响权重
// =============================================================================

/// 强趋势技术指标一致权重
pub const TECH_STRONG_ALIGNED_WEIGHT: f64 = 0.035;
/// 强趋势技术指标冲突权重
pub const TECH_STRONG_CONFLICT_WEIGHT: f64 = 0.005;
/// 强趋势技术指标中性权重
pub const TECH_STRONG_NEUTRAL_WEIGHT: f64 = 0.015;
/// 中性趋势技术指标交叉权重
pub const TECH_NEUTRAL_CROSS_WEIGHT: f64 = 0.025;
/// 中性趋势技术指标一般权重
pub const TECH_NEUTRAL_WEIGHT: f64 = 0.012;

// =============================================================================
// 九、趋势衰减系数
// =============================================================================

/// 技术指标衰减基数
pub const TECH_DECAY_BASE: f64 = 0.92;
/// 均线量能衰减基数
pub const MA_VOLUME_DECAY_BASE: f64 = 0.96;
/// 强趋势一致衰减基数
pub const STRONG_ALIGNED_DECAY_BASE: f64 = 0.99;
/// 强趋势冲突衰减基数
pub const STRONG_CONFLICT_DECAY_BASE: f64 = 0.97;
/// 普通趋势衰减基数
pub const NORMAL_TREND_DECAY_BASE: f64 = 0.95;
/// 中性趋势衰减基数
pub const NEUTRAL_TREND_DECAY_BASE: f64 = 0.90;
/// 简单预测趋势衰减基数
pub const SIMPLE_TREND_DECAY_BASE: f64 = 0.90;

// =============================================================================
// 十、均值回归参数
// =============================================================================

/// MA20回归权重
pub const MA20_REVERSION_WEIGHT: f64 = 0.6;
/// MA60回归权重
pub const MA60_REVERSION_WEIGHT: f64 = 0.4;
/// 强均值回归力量
pub const STRONG_MEAN_REVERSION_FORCE: f64 = 0.3;
/// 温和均值回归力量
pub const MODERATE_MEAN_REVERSION_FORCE: f64 = 0.2;
/// 趋势成分权重
pub const TREND_COMPONENT_WEIGHT: f64 = 0.7;

// =============================================================================
// 十一、量价预测权重
// =============================================================================

/// 方向准确率权重
pub const DIRECTION_ACCURACY_WEIGHT: f64 = 0.7;
/// 价格准确率权重
pub const PRICE_ACCURACY_WEIGHT: f64 = 0.3;
/// 强信号基础置信度
pub const STRONG_SIGNAL_BASE_CONFIDENCE: f64 = 0.7;
/// 弱信号基础置信度
pub const WEAK_SIGNAL_BASE_CONFIDENCE: f64 = 0.55;
/// 信号差异置信度加成
pub const SIGNAL_DIFF_CONFIDENCE_BOOST: f64 = 0.05;
/// 信号差异置信度加成（弱信号）
pub const WEAK_SIGNAL_DIFF_CONFIDENCE_BOOST: f64 = 0.03;

