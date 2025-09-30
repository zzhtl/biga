/// 股票预测系统 - 技术参数配置
/// 
/// 本文件管理技术指标计算周期、阈值、评分参数等技术配置
/// 这些参数不直接影响预测权重，而是影响各指标的计算和判断逻辑
/// 
/// 注意：核心预测权重已移至 core_weights.rs 文件集中管理
/// 
/// 调整建议：
/// - 阈值调整建议每次±0.05，避免过度敏感
/// - 周期参数调整需要综合考虑市场特性
/// - 调整后需要进行回测验证效果

// ============================================================================
// 一、技术指标计算周期 (Technical Indicator Parameters)
// ============================================================================

/// MACD快线周期 (12)
pub const MACD_FAST_PERIOD: usize = 12;

/// MACD慢线周期 (26)
pub const MACD_SLOW_PERIOD: usize = 26;

/// MACD信号线周期 (9)
pub const MACD_SIGNAL_PERIOD: usize = 9;

/// KDJ K值周期 (9)
pub const KDJ_K_PERIOD: usize = 9;

/// KDJ D值周期 (3)
pub const KDJ_D_PERIOD: usize = 3;

/// KDJ J值周期 (3)
pub const KDJ_J_PERIOD: usize = 3;

/// RSI周期 (14)
pub const RSI_PERIOD: usize = 14;

/// 布林带周期 (20)
pub const BOLLINGER_PERIOD: usize = 20;

/// 布林带标准差倍数 (2.0)
pub const BOLLINGER_STD_DEV: f64 = 2.0;

/// ATR周期 (14)
pub const ATR_PERIOD: usize = 14;

/// DMI周期 (14)
pub const DMI_PERIOD: usize = 14;

// ============================================================================
// 二、趋势因子评分参数 (Trend Factor Scoring Parameters)
// ============================================================================

/// 均线多头排列加分 (25分)
pub const BULLISH_MA_ALIGNMENT_SCORE: f64 = 25.0;

/// 均线空头排列减分 (25分)
pub const BEARISH_MA_ALIGNMENT_SCORE: f64 = 25.0;

/// 价格位于均线之上加分 (12分)
pub const PRICE_ABOVE_MA_SCORE: f64 = 12.0;

/// 价格位于均线之下减分 (12分)
pub const PRICE_BELOW_MA_SCORE: f64 = 12.0;

/// 短期均线向上发散加分 (8分)
pub const MA_UPWARD_DIVERGENCE_SCORE: f64 = 8.0;

/// 短期均线发散阈值 (2%)
pub const MA_DIVERGENCE_THRESHOLD: f64 = 0.02;

// ============================================================================
// 三、量价因子评分参数 (Volume-Price Factor Scoring Parameters)
// ============================================================================

/// 量价配合加分 (15分)
pub const VOLUME_PRICE_SYNC_SCORE: f64 = 15.0;

/// OBV上升趋势加分 (12分)
pub const OBV_UPTREND_SCORE: f64 = 12.0;

/// 强烈吸筹信号加分 (15分)
pub const STRONG_ACCUMULATION_SCORE: f64 = 15.0;

/// 吸筹信号阈值 (70分)
pub const STRONG_ACCUMULATION_THRESHOLD: f64 = 70.0;

/// 一般吸筹信号加分 (8分)
pub const NORMAL_ACCUMULATION_SCORE: f64 = 8.0;

/// 一般吸筹信号阈值 (50分)
pub const NORMAL_ACCUMULATION_THRESHOLD: f64 = 50.0;

/// 成交量放大加分 (8分)
pub const VOLUME_SURGE_SCORE: f64 = 8.0;

/// 成交量萎缩减分 (5分)
pub const VOLUME_SHRINK_SCORE: f64 = 5.0;

// ============================================================================
// 四、动量因子评分参数 (Momentum Factor Scoring Parameters)
// ============================================================================

/// RSI超卖加分 (20分)
pub const RSI_OVERSOLD_SCORE: f64 = 20.0;

/// RSI超买减分 (20分)
pub const RSI_OVERBOUGHT_SCORE: f64 = 20.0;

/// RSI超卖阈值 (30)
pub const RSI_OVERSOLD_THRESHOLD: f64 = 30.0;

/// RSI超买阈值 (70)
pub const RSI_OVERBOUGHT_THRESHOLD: f64 = 70.0;

/// MACD双线上穿零轴加分 (20分)
pub const MACD_BULLISH_ABOVE_ZERO_SCORE: f64 = 20.0;

/// MACD金叉加分 (15分)
pub const MACD_GOLDEN_CROSS_SCORE: f64 = 15.0;

/// MACD双线下穿零轴减分 (20分)
pub const MACD_BEARISH_BELOW_ZERO_SCORE: f64 = 20.0;

/// MACD死叉减分 (15分)
pub const MACD_DEATH_CROSS_SCORE: f64 = 15.0;

/// MACD红柱加分 (10分)
pub const MACD_POSITIVE_BAR_SCORE: f64 = 10.0;

/// MACD绿柱减分 (10分)
pub const MACD_NEGATIVE_BAR_SCORE: f64 = 10.0;

// ============================================================================
// 五、支撑压力位评分参数 (Support/Resistance Scoring Parameters)
// ============================================================================

/// 接近强支撑加分 (25分)
pub const NEAR_STRONG_SUPPORT_SCORE: f64 = 25.0;

/// 靠近支撑区域加分 (15分)
pub const NEAR_SUPPORT_SCORE: f64 = 15.0;

/// 接近强压力减分 (25分)
pub const NEAR_STRONG_RESISTANCE_SCORE: f64 = 25.0;

/// 靠近压力区域减分 (15分)
pub const NEAR_RESISTANCE_SCORE: f64 = 15.0;

/// 上方空间充足加分 (10分)
pub const SUFFICIENT_UPSIDE_SCORE: f64 = 10.0;

/// 位于区间底部加分 (10分)
pub const BOTTOM_POSITION_SCORE: f64 = 10.0;

/// 位于区间顶部减分 (10分)
pub const TOP_POSITION_SCORE: f64 = 10.0;

/// 强支撑/压力距离阈值 (2%)
pub const STRONG_LEVEL_DISTANCE_THRESHOLD: f64 = 0.02;

/// 一般支撑/压力距离阈值 (5%)
pub const NORMAL_LEVEL_DISTANCE_THRESHOLD: f64 = 0.05;

/// 上方空间充足阈值 (10%)
pub const UPSIDE_SUFFICIENT_THRESHOLD: f64 = 0.10;

/// 区间底部位置阈值 (0.3)
pub const BOTTOM_ZONE_THRESHOLD: f64 = 0.3;

/// 区间顶部位置阈值 (0.7)
pub const TOP_ZONE_THRESHOLD: f64 = 0.7;

// ============================================================================
// 六、多周期共振评分参数 (Multi-Timeframe Scoring Parameters)
// ============================================================================

/// 共振级别单级加分 (12分)
pub const RESONANCE_LEVEL_SCORE: f64 = 12.0;

/// 多周期多头共振加分 (15分)
pub const MULTI_BULLISH_RESONANCE_SCORE: f64 = 15.0;

/// 多周期空头共振减分 (15分)
pub const MULTI_BEARISH_RESONANCE_SCORE: f64 = 15.0;

/// 信号质量影响系数 (0.3)
pub const SIGNAL_QUALITY_IMPACT: f64 = 0.3;

// ============================================================================
// 七、波动率评分参数 (Volatility Scoring Parameters)
// ============================================================================

/// 极低波动加分 (20分)
pub const VERY_LOW_VOLATILITY_SCORE: f64 = 20.0;

/// 低波动加分 (10分)
pub const LOW_VOLATILITY_SCORE: f64 = 10.0;

/// 高波动减分 (15分)
pub const HIGH_VOLATILITY_SCORE: f64 = 15.0;

/// 极高波动减分 (25分)
pub const VERY_HIGH_VOLATILITY_SCORE: f64 = 25.0;

/// 极低波动阈值 (1%)
pub const VERY_LOW_VOLATILITY_THRESHOLD: f64 = 0.01;

/// 低波动阈值 (1.5%)
pub const LOW_VOLATILITY_THRESHOLD: f64 = 0.015;

/// 中等波动阈值 (3%)
pub const MEDIUM_VOLATILITY_THRESHOLD: f64 = 0.03;

/// 高波动阈值 (5%)
pub const HIGH_VOLATILITY_THRESHOLD: f64 = 0.05;

// ============================================================================
// 八、智能权重调整参数 (Adaptive Weight Adjustment Parameters)
// ============================================================================

/// 强趋势权重倍数 (1.15)
pub const STRONG_TREND_MULTIPLIER: f64 = 1.15;

/// 中等趋势权重倍数 (1.08)
pub const MEDIUM_TREND_MULTIPLIER: f64 = 1.08;

/// 弱趋势权重倍数 (0.92)
pub const WEAK_TREND_MULTIPLIER: f64 = 0.92;

/// ADX强趋势阈值 (40)
pub const STRONG_TREND_ADX_THRESHOLD: f64 = 40.0;

/// ADX中等趋势阈值 (25)
pub const MEDIUM_TREND_ADX_THRESHOLD: f64 = 25.0;

/// 高波动权重惩罚系数 (0.90)
pub const HIGH_VOLATILITY_PENALTY: f64 = 0.90;

/// 中高波动权重惩罚系数 (0.95)
pub const MEDIUM_HIGH_VOLATILITY_PENALTY: f64 = 0.95;

/// 单因子最大权重限制 (30%)
pub const MAX_SINGLE_FACTOR_WEIGHT: f64 = 0.30;

/// 单因子最小权重限制 (2%)
pub const MIN_SINGLE_FACTOR_WEIGHT: f64 = 0.02;

// ============================================================================
// 九、信号质量评级阈值 (Signal Quality Thresholds)
// ============================================================================

/// 优秀信号阈值 (85分)
pub const EXCELLENT_SIGNAL_THRESHOLD: f64 = 85.0;

/// 良好信号阈值 (70分)
pub const GOOD_SIGNAL_THRESHOLD: f64 = 70.0;

/// 一般信号阈值 (55分)
pub const FAIR_SIGNAL_THRESHOLD: f64 = 55.0;

/// 较差信号阈值 (40分)
pub const POOR_SIGNAL_THRESHOLD: f64 = 40.0;

// ============================================================================
// 十、操作建议阈值 (Trading Advice Thresholds)
// ============================================================================

/// 强烈买入阈值 (75分)
pub const STRONG_BUY_THRESHOLD: f64 = 75.0;

/// 可以买入阈值 (65分)
pub const BUY_THRESHOLD: f64 = 65.0;

/// 轻仓试探阈值 (55分)
pub const LIGHT_BUY_THRESHOLD: f64 = 55.0;

/// 观望阈值 (45分)
pub const HOLD_THRESHOLD: f64 = 45.0;

/// 考虑减仓阈值 (35分)
pub const CONSIDER_SELL_THRESHOLD: f64 = 35.0;

// ============================================================================
// 十一、K线形态参数 (Candlestick Pattern Parameters)
// ============================================================================

/// K线形态影响系数 (0.5)
pub const PATTERN_IMPACT_FACTOR: f64 = 0.5;

/// 中性形态减分 (5分)
pub const NEUTRAL_PATTERN_PENALTY: f64 = 5.0;

// ============================================================================
// 十二、预测调整系数 (Prediction Adjustment Factors)
// ============================================================================

/// 趋势一致性加成系数 (1.2)
pub const TREND_ALIGNMENT_MULTIPLIER: f64 = 1.2;

/// 趋势冲突抑制系数 (0.7)
pub const TREND_CONFLICT_MULTIPLIER: f64 = 0.7;

/// 强支撑附近看涨加成 (1.3)
pub const NEAR_SUPPORT_BULLISH_MULTIPLIER: f64 = 1.3;

/// 强压力附近看跌加成 (1.3)
pub const NEAR_RESISTANCE_BEARISH_MULTIPLIER: f64 = 1.3;

/// 波动率调整基准 (0.02)
pub const VOLATILITY_BASE: f64 = 0.02;

/// 最大日涨跌幅限制 (10%)
pub const MAX_DAILY_CHANGE: f64 = 0.10;

// ============================================================================
// 十三、风险控制参数 (Risk Control Parameters)
// ============================================================================

/// 止损距离系数 (0.05)
pub const STOP_LOSS_DISTANCE: f64 = 0.05;

/// 第一止盈目标系数 (0.08)
pub const FIRST_TAKE_PROFIT: f64 = 0.08;

/// 第二止盈目标系数 (0.15)
pub const SECOND_TAKE_PROFIT: f64 = 0.15;

/// 最小风险收益比 (1.5)
pub const MIN_RISK_REWARD_RATIO: f64 = 1.5;

// ============================================================================
// 十四、模型训练参数 (Model Training Parameters)
// ============================================================================

/// 默认学习率 (0.001)
pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

/// 默认Dropout率 (0.1)
pub const DEFAULT_DROPOUT: f64 = 0.1;

/// 默认训练轮数 (100)
pub const DEFAULT_EPOCHS: usize = 100;

/// 默认批次大小 (32)
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// 默认训练集比例 (0.8)
pub const DEFAULT_TRAIN_TEST_SPLIT: f64 = 0.8;

// ============================================================================
// 十五、预测算法基础参数 (Prediction Algorithm Base Parameters)
// ============================================================================

/// 基础模型预测权重 (2%)
pub const BASE_MODEL_WEIGHT: f64 = 0.02;

/// 趋势强度映射系数 (0.012)
pub const TREND_STRENGTH_MAPPING: f64 = 0.012;

/// 趋势偏置权重系数 (0.5)
pub const TREND_BIAS_WEIGHT: f64 = 0.5;

// ============================================================================
// 十六、均线与量能偏置系数 (MA and Volume Bias Factors)
// ============================================================================

/// 价格位于MA5上方/下方偏置 (0.4)
pub const PRICE_VS_MA5_BIAS: f64 = 0.4;

/// MA5相对MA10位置偏置 (0.3)
pub const MA5_VS_MA10_BIAS: f64 = 0.3;

/// MA10相对MA20位置偏置 (0.3)
pub const MA10_VS_MA20_BIAS: f64 = 0.3;

/// MA5斜率偏置 (0.2)
pub const MA5_SLOPE_BIAS: f64 = 0.2;

/// MA10斜率偏置 (0.15)
pub const MA10_SLOPE_BIAS: f64 = 0.15;

/// MA20斜率偏置 (0.1)
pub const MA20_SLOPE_BIAS: f64 = 0.1;

/// 均线偏置映射系数 (1%)
pub const MA_BIAS_MAPPING: f64 = 0.01;

/// 高量比偏置 (0.8%)
pub const HIGH_VOLUME_RATIO_BIAS: f64 = 0.008;

/// 中等量比偏置 (0.4%)
pub const MEDIUM_VOLUME_RATIO_BIAS: f64 = 0.004;

/// 量比阈值 - 高量 (1.5)
pub const HIGH_VOLUME_RATIO_THRESHOLD: f64 = 1.5;

/// 量比阈值 - 中量 (1.2)
pub const MEDIUM_VOLUME_RATIO_THRESHOLD: f64 = 1.2;

/// 量比阈值 - 低量 (0.8)
pub const LOW_VOLUME_RATIO_THRESHOLD: f64 = 0.8;

/// 量比阈值 - 极低量 (0.6)
pub const VERY_LOW_VOLUME_RATIO_THRESHOLD: f64 = 0.6;

// ============================================================================
// 十八、波动率调整系数 (Volatility Adjustment Factors)
// ============================================================================

/// 强趋势一致波动率系数 (0.5)
pub const STRONG_ALIGNED_VOLATILITY_FACTOR: f64 = 0.5;

/// 强趋势冲突波动率系数 (0.7)
pub const STRONG_CONFLICT_VOLATILITY_FACTOR: f64 = 0.7;

/// 普通趋势波动率系数 (0.9)
pub const NORMAL_TREND_VOLATILITY_FACTOR: f64 = 0.9;

/// 中性趋势波动率系数 (1.1)
pub const NEUTRAL_TREND_VOLATILITY_FACTOR: f64 = 1.1;

/// 简单预测波动率基数 (0.25)
pub const SIMPLE_PREDICTION_BASE_VOLATILITY: f64 = 0.25;

/// 黄金分割波动周期系数 (0.618)
pub const GOLDEN_RATIO_OSCILLATION: f64 = 0.618;

/// 波动幅度归一化系数 (0.5)
pub const VOLATILITY_NORMALIZATION: f64 = 0.5;

/// 震荡市波动增强系数 (1.0)
pub const OSCILLATION_MARKET_VOLATILITY: f64 = 1.0;

/// 趋势市波动减弱系数 (0.5)
pub const TREND_MARKET_VOLATILITY: f64 = 0.5;

// ============================================================================
// 十九、均值回归阈值 (Mean Reversion Thresholds)
// ============================================================================

/// 强均值回归阈值 (5%)
pub const STRONG_MEAN_REVERSION_THRESHOLD: f64 = 0.05;

/// 温和均值回归阈值 (3%)
pub const MODERATE_MEAN_REVERSION_THRESHOLD: f64 = 0.03;

// ============================================================================
// 二十、压力支撑位影响参数 (Support/Resistance Impact Parameters)
// ============================================================================

/// 接近压力位距离阈值 (3%)
pub const NEAR_RESISTANCE_DISTANCE: f64 = 0.03;

/// 接近支撑位距离阈值 (3%)
pub const NEAR_SUPPORT_DISTANCE: f64 = 0.03;

/// 接近压力位趋势减弱系数 (30%)
pub const NEAR_RESISTANCE_WEAKEN: f64 = 0.3;

/// 接近支撑位趋势减弱系数 (30%)
pub const NEAR_SUPPORT_WEAKEN: f64 = 0.3;

/// RSI强超买趋势减弱系数 (40%)
pub const RSI_STRONG_OVERBOUGHT_WEAKEN: f64 = 0.4;

/// RSI中等超买趋势减弱系数 (70%)
pub const RSI_MODERATE_OVERBOUGHT_WEAKEN: f64 = 0.7;

/// RSI超买阈值 - 中等 (65)
pub const RSI_MODERATE_OVERBOUGHT: f64 = 65.0;

/// RSI强超卖趋势减弱系数 (40%)
pub const RSI_STRONG_OVERSOLD_WEAKEN: f64 = 0.4;

/// RSI中等超卖趋势减弱系数 (70%)
pub const RSI_MODERATE_OVERSOLD_WEAKEN: f64 = 0.7;

/// RSI超卖阈值 - 中等 (35)
pub const RSI_MODERATE_OVERSOLD: f64 = 35.0;

// ============================================================================
// 二十一、买卖点信号参数 (Buy/Sell Signal Parameters)
// ============================================================================

/// 买入点基础信号强度 (70分)
pub const BUY_POINT_BASE_STRENGTH: f64 = 70.0;

/// 信号质量影响系数 (0.2)
pub const SIGNAL_QUALITY_STRENGTH_IMPACT: f64 = 0.2;

/// 买入点基础置信度 (0.75)
pub const BUY_POINT_BASE_CONFIDENCE: f64 = 0.75;

/// 共振级别置信度加成 (0.05/级)
pub const RESONANCE_CONFIDENCE_BOOST: f64 = 0.05;

/// 接近压力位判定比例 (99%-102%)
pub const NEAR_RESISTANCE_RANGE_LOW: f64 = 0.99;
pub const NEAR_RESISTANCE_RANGE_HIGH: f64 = 1.02;

/// 预测基础置信度 (70%)
pub const PREDICTION_BASE_CONFIDENCE: f64 = 0.70;

/// 预测置信度下限 (40%)
pub const PREDICTION_MIN_CONFIDENCE: f64 = 0.40;

/// 预测置信度上限 (85%)
pub const PREDICTION_MAX_CONFIDENCE: f64 = 0.85;

/// 信号质量置信度影响 (0.003)
pub const SIGNAL_QUALITY_CONFIDENCE_IMPACT: f64 = 0.003;

// ============================================================================
// 使用说明
// ============================================================================
//
// 本文件包含技术参数配置，这些参数不直接影响预测权重，
// 而是影响技术指标的计算、阈值判断和评分逻辑。
//
// 核心预测权重已移至 core_weights.rs 文件管理，
// 包括：
// - 多因子评分权重
// - 预测算法核心权重
// - 趋势一致性增强系数
// - 方向投票权重
// - 技术指标影响权重
// - 趋势衰减系数
// - 均值回归参数
// - 量价预测权重
//
// 调整本文件参数时，重点关注：
// 1. 技术指标计算周期是否合理
// 2. 阈值设置是否符合市场特性
// 3. 评分参数是否能正确反映信号强度
// 4. 风险控制参数是否合理 