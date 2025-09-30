use candle_core::{Device, Tensor};
use candle_nn::{Module, VarMap};
use chrono;
use crate::stock_prediction::types::{
    ModelConfig, Prediction, TechnicalIndicatorValues, PredictionRequest, 
    PredictionResponse, LastRealData, TradingSignal
};
use crate::stock_prediction::model_management::{list_models, get_model_file_path};
use crate::stock_prediction::database::get_recent_market_data;
use crate::stock_prediction::utils::{
    get_next_trading_day, clamp_daily_change, calculate_historical_volatility,
    calculate_recent_trend, calculate_support_resistance, analyze_historical_volatility_pattern,
    print_last_real_vs_prediction, analyze_stock_trend, TrendState, predict_with_volume_price, 
    calculate_volume_price_change
};
use crate::stock_prediction::technical_analysis::analyze_technical_signals;
use crate::stock_prediction::technical_indicators::{
    get_feature_required_days, 
    calculate_feature_value, 
    calculate_rsi,
    calculate_macd_full
};
use crate::stock_prediction::multi_timeframe_analysis::{
    StockData, convert_to_weekly, convert_to_monthly, calculate_macd_signal, calculate_kdj_signal
};
use crate::stock_prediction::volume_analysis;
use crate::stock_prediction::candlestick_patterns;
use crate::stock_prediction::multi_factor_scoring;
use crate::stock_prediction::core_weights::*; // 核心预测权重
use crate::stock_prediction::constants::*;    // 技术参数配置

// ==================== 金融级预测策略系统 ====================

/// 买卖点信号
/// 金融术语规范：
/// - 止损位：价格跌到此位置时卖出止损（止损位 < 当前价）
/// - 止盈位：价格涨到此位置时卖出获利（止盈位 > 当前价）
/// - 无论买入点还是卖出点，止损位永远 < 当前价 < 止盈位
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BuySellPoint {
    pub point_type: String,           // "买入点" 或 "卖出点"
    pub signal_strength: f64,         // 信号强度 0-100
    pub price_level: f64,             // 建议操作价格
    pub stop_loss: f64,               // 止损位（跌到此价位卖出止损）
    pub take_profit: Vec<f64>,        // 止盈位（涨到此价位卖出获利，可多个目标）
    pub risk_reward_ratio: f64,       // 风险收益比 = 潜在收益/潜在风险
    pub reasons: Vec<String>,         // 信号产生原因
    pub confidence: f64,              // 信号置信度 0-1
    pub accuracy_rate: Option<f64>,   // 历史准确率（如有回测数据）
}

/// 支撑压力位
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SupportResistance {
    pub support_levels: Vec<f64>,     // 支撑位（从强到弱）
    pub resistance_levels: Vec<f64>,  // 压力位（从强到弱）
    pub current_position: String,     // 当前位置描述
}

/// 多周期共振分析结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiTimeframeSignal {
    pub daily_trend: String,          // 日线趋势
    pub weekly_trend: String,         // 周线趋势
    pub monthly_trend: String,        // 月线趋势
    pub resonance_level: i32,         // 共振级别 0-3
    pub resonance_direction: String,  // 共振方向
    pub signal_quality: f64,          // 信号质量 0-100
}

/// 量价背离分析
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VolumePriceDivergence {
    pub has_bullish_divergence: bool, // 底背离（看涨）
    pub has_bearish_divergence: bool, // 顶背离（看跌）
    pub divergence_strength: f64,     // 背离强度
    pub warning_message: String,      // 预警信息
}

/// 金融级预测结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProfessionalPrediction {
    pub buy_points: Vec<BuySellPoint>,
    pub sell_points: Vec<BuySellPoint>,
    pub support_resistance: SupportResistance,
    pub multi_timeframe: MultiTimeframeSignal,
    pub divergence: VolumePriceDivergence,
    pub current_advice: String,
    pub risk_level: String,
    pub candle_patterns: Vec<candlestick_patterns::PatternRecognition>, // K线形态
    pub volume_analysis: VolumeAnalysisInfo,  // 量价分析结果
    pub multi_factor_score: multi_factor_scoring::MultiFactorScore,  // 多因子综合评分
}

/// 量价分析信息（用于序列化）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VolumeAnalysisInfo {
    pub volume_trend: String,        // 量能趋势
    pub volume_price_sync: bool,     // 量价配合
    pub accumulation_signal: f64,    // 吸筹信号强度
    pub obv_trend: String,          // OBV趋势
}

// ==================== 核心策略函数 ====================

/// 计算支撑压力位
fn calculate_support_resistance_levels(
    prices: &[f64], 
    highs: &[f64], 
    lows: &[f64], 
    current_price: f64
) -> SupportResistance {
    let mut support_levels = Vec::new();
    let mut resistance_levels = Vec::new();
    
    if prices.len() < 20 {
        return SupportResistance {
            support_levels,
            resistance_levels,
            current_position: "数据不足".to_string(),
        };
    }
    
    let n = prices.len();
    
    // 1. 计算均线支撑/压力
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    let ma60 = calc_ma(60);
    
    // 2. 历史高低点（60日内）
    let lookback = n.min(60);
    let recent_high = highs[n-lookback..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let recent_low = lows[n-lookback..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    // 3. 斐波那契回撤位
    let fib_range = recent_high - recent_low;
    let fib_382 = recent_high - fib_range * 0.382;
    let fib_500 = recent_high - fib_range * 0.500;
    let fib_618 = recent_high - fib_range * 0.618;
    
    // 分类支撑和压力
    let mut all_levels = vec![
        ma5, ma10, ma20, ma60,
        recent_high, recent_low,
        fib_382, fib_500, fib_618,
    ];
    
    // 去重并排序
    all_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_levels.dedup_by(|a, b| (*a - *b).abs() < current_price * 0.01);
    
    for &level in &all_levels {
        if level < current_price && level > current_price * 0.85 {
            support_levels.push(level);
        } else if level > current_price && level < current_price * 1.15 {
            resistance_levels.push(level);
        }
    }
    
    // 按距离当前价格排序
    support_levels.sort_by(|a, b| (current_price - a).partial_cmp(&(current_price - b)).unwrap());
    resistance_levels.sort_by(|a, b| (a - current_price).partial_cmp(&(b - current_price)).unwrap());
    
    // 限制数量
    support_levels.truncate(5);
    resistance_levels.truncate(5);
    
    let current_position = if !support_levels.is_empty() && !resistance_levels.is_empty() {
        let to_support = ((current_price - support_levels[0]) / current_price * 100.0).abs();
        let to_resistance = ((resistance_levels[0] - current_price) / current_price * 100.0).abs();
        
        if to_support < 2.0 {
            "接近关键支撑".to_string()
        } else if to_resistance < 2.0 {
            "接近关键压力".to_string()
        } else if to_support < to_resistance {
            format!("中性偏下，距支撑{:.2}%", to_support)
        } else {
            format!("中性偏上，距压力{:.2}%", to_resistance)
        }
    } else {
        "中性区域".to_string()
    };
    
    SupportResistance {
        support_levels,
        resistance_levels,
        current_position,
    }
}

/// 多周期共振分析
fn analyze_multi_timeframe_resonance(
    daily_data: &[StockData],
) -> MultiTimeframeSignal {
    let weekly_data = convert_to_weekly(daily_data);
    let monthly_data = convert_to_monthly(daily_data);
    
    // 计算各周期MACD和趋势
    let daily_macd = calculate_macd_signal(daily_data, 12, 26, 9);
    let weekly_macd = calculate_macd_signal(&weekly_data, 12, 26, 9);
    let monthly_macd = calculate_macd_signal(&monthly_data, 12, 26, 9);
    
    let trend_from_macd = |signals: &[crate::stock_prediction::multi_timeframe_analysis::MacdSignal]| -> (String, i32) {
        if let Some(last) = signals.last() {
            if last.is_golden_cross {
                ("多头".to_string(), 1)
            } else if last.is_death_cross {
                ("空头".to_string(), -1)
            } else if last.histogram > 0.0 {
                ("偏多".to_string(), 1)
            } else {
                ("偏空".to_string(), -1)
            }
        } else {
            ("中性".to_string(), 0)
        }
    };
    
    let (daily_trend, daily_score) = trend_from_macd(&daily_macd);
    let (weekly_trend, weekly_score) = trend_from_macd(&weekly_macd);
    let (monthly_trend, monthly_score) = trend_from_macd(&monthly_macd);
    
    // 计算共振
    let resonance_score = daily_score + weekly_score + monthly_score;
    let (resonance_level, resonance_direction) = match resonance_score {
        3 => (3, "强烈多头共振".to_string()),
        2 => (2, "多头共振".to_string()),
        1 => (1, "偏多".to_string()),
        -1 => (1, "偏空".to_string()),
        -2 => (2, "空头共振".to_string()),
        -3 => (3, "强烈空头共振".to_string()),
        _ => (0, "无明显共振".to_string()),
    };
    
    // 信号质量评分
    let signal_quality = match resonance_level {
        3 => 95.0,
        2 => 80.0,
        1 => 60.0,
        _ => 40.0,
    };
    
    MultiTimeframeSignal {
        daily_trend,
        weekly_trend,
        monthly_trend,
        resonance_level,
        resonance_direction,
        signal_quality,
    }
}

/// 量价背离分析
fn analyze_volume_price_divergence(
    prices: &[f64],
    volumes: &[i64],
    highs: &[f64],
    lows: &[f64],
) -> VolumePriceDivergence {
    if prices.len() < 20 {
        return VolumePriceDivergence {
            has_bullish_divergence: false,
            has_bearish_divergence: false,
            divergence_strength: 0.0,
            warning_message: "数据不足".to_string(),
        };
    }
    
    let n = prices.len();
    let lookback = 20.min(n);
    
    // 寻找价格的高点和低点
    let mut price_peaks = Vec::new();
    let mut price_troughs = Vec::new();
    
    for i in 1..lookback-1 {
        let idx = n - lookback + i;
        if highs[idx] > highs[idx-1] && highs[idx] > highs[idx+1] {
            price_peaks.push((idx, highs[idx]));
        }
        if lows[idx] < lows[idx-1] && lows[idx] < lows[idx+1] {
            price_troughs.push((idx, lows[idx]));
        }
    }
    
    let mut has_bullish_divergence = false;
    let mut has_bearish_divergence = false;
    let mut divergence_strength = 0.0;
    let mut warning_message = "无明显背离".to_string();
    
    // 底背离检测（价格新低，指标未新低）
    if price_troughs.len() >= 2 {
        let last_trough = price_troughs[price_troughs.len()-1];
        let prev_trough = price_troughs[price_troughs.len()-2];
        
        if last_trough.1 < prev_trough.1 {
            // 价格创新低，检查成交量是否萎缩
            let last_vol = volumes[last_trough.0];
            let prev_vol = volumes[prev_trough.0];
            
            if last_vol < prev_vol {
                has_bullish_divergence = true;
                divergence_strength = (prev_vol as f64 - last_vol as f64) / prev_vol as f64;
                warning_message = "检测到底背离信号，可能即将反弹".to_string();
            }
        }
    }
    
    // 顶背离检测（价格新高，指标未新高）
    if price_peaks.len() >= 2 {
        let last_peak = price_peaks[price_peaks.len()-1];
        let prev_peak = price_peaks[price_peaks.len()-2];
        
        if last_peak.1 > prev_peak.1 {
            // 价格创新高，检查成交量是否萎缩
            let last_vol = volumes[last_peak.0];
            let prev_vol = volumes[prev_peak.0];
            
            if last_vol < prev_vol {
                has_bearish_divergence = true;
                divergence_strength = (prev_vol as f64 - last_vol as f64) / prev_vol as f64;
                warning_message = "检测到顶背离信号，注意回调风险".to_string();
            }
        }
    }
    
    VolumePriceDivergence {
        has_bullish_divergence,
        has_bearish_divergence,
        divergence_strength,
        warning_message,
    }
}

/// 识别买入点
fn identify_buy_points(
    prices: &[f64],
    volumes: &[i64],
    _highs: &[f64],
    _lows: &[f64],
    current_price: f64,
    support_resistance: &SupportResistance,
    multi_timeframe: &MultiTimeframeSignal,
    divergence: &VolumePriceDivergence,
) -> Vec<BuySellPoint> {
    let mut buy_points = Vec::new();
    
    if prices.len() < 20 {
        return buy_points;
    }
    
    let n = prices.len();
    
    // 计算技术指标
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    
    // 买点1：多周期共振 + 均线多头排列
    if multi_timeframe.resonance_level >= 2 
       && multi_timeframe.resonance_direction.contains("多头")
       && ma5 > ma10 && ma10 > ma20 {
        let mut reasons = vec![
            "多周期共振向上".to_string(),
            "均线呈多头排列".to_string(),
        ];
        
        let nearest_support = support_resistance.support_levels.first().copied().unwrap_or(current_price * 0.95);
        let nearest_resistance = support_resistance.resistance_levels.first().copied().unwrap_or(current_price * 1.10);
        
        let stop_loss = nearest_support * 0.97;
        let take_profit1 = nearest_resistance;
        let take_profit2 = current_price + (nearest_resistance - current_price) * 1.5;
        
        let risk = current_price - stop_loss;
        let reward = take_profit1 - current_price;
        let risk_reward_ratio = if risk > 0.0 { reward / risk } else { 0.0 };
        
        if divergence.has_bullish_divergence {
            reasons.push("底部背离确认".to_string());
        }
        
        let signal_strength = BUY_POINT_BASE_STRENGTH + multi_timeframe.signal_quality * SIGNAL_QUALITY_STRENGTH_IMPACT;
        
        buy_points.push(BuySellPoint {
            point_type: "买入点".to_string(),
            signal_strength,
            price_level: current_price,
            stop_loss,
            take_profit: vec![take_profit1, take_profit2],
            risk_reward_ratio,
            reasons,
            confidence: BUY_POINT_BASE_CONFIDENCE + multi_timeframe.resonance_level as f64 * RESONANCE_CONFIDENCE_BOOST,
            accuracy_rate: None,  // 待回测统计
        });
    }
    
    // 买点2：突破压力位 + 放量
    if let Some(&first_resistance) = support_resistance.resistance_levels.first() {
        if current_price > first_resistance * NEAR_RESISTANCE_RANGE_LOW && current_price < first_resistance * NEAR_RESISTANCE_RANGE_HIGH {
            // 检查是否放量
            if volumes.len() >= 5 {
                let recent_vol = volumes[n-1];
                let avg_vol = volumes[n-5..n-1].iter().map(|&v| v as f64).sum::<f64>() / 4.0;
                
                if recent_vol as f64 > avg_vol * 1.3 {
                    let stop_loss = first_resistance * 0.96;
                    let take_profit1 = current_price * 1.08;
                    
                    let risk = current_price - stop_loss;
                    let reward = take_profit1 - current_price;
                    let risk_reward_ratio = if risk > 0.0 { reward / risk } else { 0.0 };
                    
                    buy_points.push(BuySellPoint {
                        point_type: "突破买入点".to_string(),
                        signal_strength: 75.0,
                        price_level: current_price,
                        stop_loss,
                        take_profit: vec![take_profit1],
                        risk_reward_ratio,
                        reasons: vec![
                            "突破关键压力位".to_string(),
                            "放量确认突破有效".to_string(),
                        ],
                        confidence: 0.70,
                        accuracy_rate: None,  // 待回测统计
                    });
                }
            }
        }
    }
    
    // 买点3：回踩支撑位 + 缩量
    if let Some(&first_support) = support_resistance.support_levels.first() {
        if current_price < first_support * 1.02 && current_price > first_support * 0.98 {
            if volumes.len() >= 5 {
                let recent_vol = volumes[n-1];
                let avg_vol = volumes[n-5..n-1].iter().map(|&v| v as f64).sum::<f64>() / 4.0;
                
                if (recent_vol as f64) < avg_vol * 0.8 {
                    let stop_loss = first_support * 0.95;
                    let take_profit1 = current_price * 1.05;
                    
                    let risk = current_price - stop_loss;
                    let reward = take_profit1 - current_price;
                    let risk_reward_ratio = if risk > 0.0 { reward / risk } else { 0.0 };
                    
                    buy_points.push(BuySellPoint {
                        point_type: "回踩支撑买入点".to_string(),
                        signal_strength: 65.0,
                        price_level: current_price,
                        stop_loss,
                        take_profit: vec![take_profit1],
                        risk_reward_ratio,
                        reasons: vec![
                            "回踩关键支撑位".to_string(),
                            "缩量显示抛压减弱".to_string(),
                        ],
                        confidence: 0.65,
                        accuracy_rate: None,  // 待回测统计
                    });
                }
            }
        }
    }
    
    // 过滤掉风险收益比不佳的信号
    buy_points.retain(|bp| bp.risk_reward_ratio >= 1.5);
    
    // 按信号强度排序
    buy_points.sort_by(|a, b| b.signal_strength.partial_cmp(&a.signal_strength).unwrap());
    
    buy_points
}

/// 识别卖出点
fn identify_sell_points(
    prices: &[f64],
    volumes: &[i64],
    _highs: &[f64],
    _lows: &[f64],
    current_price: f64,
    support_resistance: &SupportResistance,
    multi_timeframe: &MultiTimeframeSignal,
    divergence: &VolumePriceDivergence,
) -> Vec<BuySellPoint> {
    let mut sell_points = Vec::new();
    
    if prices.len() < 20 {
        return sell_points;
    }
    
    let n = prices.len();
    
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    
    // 卖点1：多周期共振向下 + 均线空头排列
    // 建议持仓者现在卖出，如果不卖则设置防护止损止盈
    if multi_timeframe.resonance_level >= 2 
       && multi_timeframe.resonance_direction.contains("空头")
       && ma5 < ma10 && ma10 < ma20 {
        let mut reasons = vec![
            "多周期共振向下".to_string(),
            "均线呈空头排列".to_string(),
        ];
        
        if divergence.has_bearish_divergence {
            reasons.push("顶部背离确认".to_string());
        }
        
        let signal_strength = 75.0 + multi_timeframe.signal_quality * 0.2;
        
        // 止损位：跌3%止损（防止进一步下跌损失）
        let stop_loss = current_price * 0.97;
        // 止盈位：如果判断错误反弹，涨3-5%止盈
        let take_profit = vec![current_price * 1.03, current_price * 1.05];
        
        // 风险收益比：如果不卖出，向下风险vs向上机会
        let downside_risk = current_price * 0.10;  // 预期下跌10%的风险
        let upside_potential = current_price * 0.03;  // 反弹3%的可能
        let risk_reward_ratio = downside_risk / upside_potential;
        
        sell_points.push(BuySellPoint {
            point_type: "卖出点".to_string(),
            signal_strength,
            price_level: current_price,
            stop_loss,
            take_profit,
            risk_reward_ratio,
            reasons,
            confidence: BUY_POINT_BASE_CONFIDENCE + multi_timeframe.resonance_level as f64 * RESONANCE_CONFIDENCE_BOOST,
            accuracy_rate: None,  // 待回测统计
        });
    }
    
    // 卖点2：跌破关键支撑
    // 破位信号，建议立即止损出局
    if let Some(&first_support) = support_resistance.support_levels.first() {
        if current_price < first_support * NEAR_RESISTANCE_RANGE_LOW {
            if volumes.len() >= 5 {
                let recent_vol = volumes[n-1];
                let avg_vol = volumes[n-5..n-1].iter().map(|&v| v as f64).sum::<f64>() / 4.0;
                
                if recent_vol as f64 > avg_vol * 1.2 {
                    // 止损位：再跌5%必须走（已经破位，快速止损）
                    let stop_loss = current_price * 0.95;
                    // 止盈位：反弹2-3%离场（破位后很难快速修复）
                    let take_profit = vec![current_price * 1.02, current_price * 1.03];
                    
                    // 风险收益比：破位后继续下跌风险大
                    let downside_risk = current_price * 0.10;  // 破位后可能再跌10%
                    let upside_potential = current_price * 0.02;  // 反弹空间有限
                    let risk_reward_ratio = downside_risk / upside_potential;
                    
                    sell_points.push(BuySellPoint {
                        point_type: "破位卖出点".to_string(),
                        signal_strength: 85.0,  // 破位信号强度很高
                        price_level: current_price,
                        stop_loss,
                        take_profit,
                        risk_reward_ratio,
                        reasons: vec![
                            "跌破关键支撑位".to_string(),
                            "放量下跌确认破位".to_string(),
                        ],
                        confidence: 0.80,  // 破位信号可信度高
                        accuracy_rate: None,
                    });
                }
            }
        }
    }
    
    // 卖点3：触及压力位 + 顶背离
    // 高位风险信号，建议止盈离场
    if let Some(&first_resistance) = support_resistance.resistance_levels.first() {
        if current_price > first_resistance * 0.98 && divergence.has_bearish_divergence {
            // 止损位：跌5%止损（高位回落）
            let stop_loss = current_price * 0.95;
            // 止盈位：如果突破压力位，涨2%离场
            let take_profit = vec![current_price * 1.02];
            
            // 风险收益比：高位风险大于机会
            let downside_risk = current_price * 0.08;  // 高位回落风险8%
            let upside_potential = current_price * 0.02;  // 突破后空间有限
            let risk_reward_ratio = downside_risk / upside_potential;
            
            sell_points.push(BuySellPoint {
                point_type: "高位卖出点".to_string(),
                signal_strength: 75.0,  // 提高强度
                price_level: current_price,
                stop_loss,
                take_profit,
                risk_reward_ratio,
                reasons: vec![
                    "触及关键压力位".to_string(),
                    "顶部背离预警".to_string(),
                    "建议止盈离场".to_string(),
                ],
                confidence: 0.70,
                accuracy_rate: None,
            });
        }
    }
    
    sell_points.sort_by(|a, b| b.signal_strength.partial_cmp(&a.signal_strength).unwrap());
    
    sell_points
}

// 简化的模型创建函数（与training.rs中的相同，用于加载模型）
fn create_model(config: &ModelConfig, device: &Device) -> Result<(VarMap, Box<dyn Module + Send + Sync>), candle_core::Error> {
    // 创建一个简单的线性回归模型
    let varmap = VarMap::new();
    
    struct LinearRegression {
        linear: candle_nn::Linear,
    }
    
    impl LinearRegression {
        fn new(in_size: usize, out_size: usize, vb: candle_nn::VarBuilder) -> Result<Self, candle_core::Error> {
            let linear = candle_nn::linear(in_size, out_size, vb)?;
            Ok(Self { linear })
        }
    }
    
    unsafe impl Send for LinearRegression {}
    unsafe impl Sync for LinearRegression {}
    
    impl Module for LinearRegression {
        fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
            self.linear.forward(xs)
        }
    }
    
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = LinearRegression::new(config.input_size, config.output_size, vb)?;
    
    let model: Box<dyn Module + Send + Sync> = Box::new(model);
    
    Ok((varmap, model))
}

// 本地特征标准化（与训练阶段逻辑一致：按列计算 mean/std，再标准化）
fn normalize_features_local(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() { return Vec::new(); }
    let cols = features[0].len();
    let rows = features.len();
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];

    for c in 0..cols {
        let mut sum = 0.0;
        for r in 0..rows { sum += features[r][c]; }
        let mean = sum / rows as f64;
        means[c] = mean;
        let mut var_sum = 0.0;
        for r in 0..rows {
            let diff = features[r][c] - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / rows as f64).sqrt().max(1e-8);
        stds[c] = std;
    }

    let mut normalized = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            normalized[r][c] = (features[r][c] - means[c]) / stds[c];
        }
    }
    normalized
}

// 股票预测函数 - 基于趋势分析的改进版本
pub async fn predict_with_candle(request: PredictionRequest) -> std::result::Result<PredictionResponse, String> {
    let model_list = list_models(&request.stock_code);
    
    if model_list.is_empty() {
        return Err("没有找到可用的模型".to_string());
    }
    
    // 获取模型元数据
    let metadata = if let Some(model_name) = &request.model_name {
        model_list.iter()
            .find(|m| m.name == *model_name)
            .ok_or_else(|| format!("找不到名为 {model_name} 的模型"))?
            .clone()
    } else {
        // 如果没有指定模型名称，使用最新的模型
        model_list[0].clone()
    };
    
    // 加载模型
    let device = Device::Cpu;
    
    let config = ModelConfig {
        model_type: metadata.model_type.clone(),
        input_size: metadata.features.len(),
        hidden_size: 64,
        output_size: 1,
        dropout: 0.0,
        learning_rate: 0.001,
        n_layers: 2,
        n_heads: 4,
        max_seq_len: 60,
    };
    
    let mut varmap = VarMap::new();
    
    let (_, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {e}"))?;
    
    let model_path = get_model_file_path(&metadata.id);
    varmap.load(&model_path).map_err(|e| format!("模型加载失败: {e}"))?;
    
    // 获取最近的真实市场数据
    let (current_price, current_change_percent, dates, mut prices, mut volumes, mut highs, mut lows) = get_recent_market_data(&request.stock_code, 60).await
        .map_err(|e| format!("获取市场数据失败: {e}"))?;
    
    if prices.len() < 20 {
        return Err("历史数据不足，无法进行预测，需要至少20天数据".to_string());
    }
    
    // === 新增：趋势分析 ===
    let trend_analysis = analyze_stock_trend(&prices, &highs, &lows, &volumes);
    
    println!("🔮 基于趋势分析进行预测:");
    println!("   📈 当前趋势: {:?}", trend_analysis.overall_trend);
    println!("   🎯 趋势强度: {:.2}", trend_analysis.trend_strength);
    println!("   🔒 趋势置信度: {:.0}%", trend_analysis.trend_confidence * 100.0);
    println!("   ⚖️  预测偏向倍数: {:.2}", trend_analysis.bias_multiplier);
    
    // === 新增：与训练一致的特征计算与标准化 ===
    // 计算所需的最小历史窗口
    let required_days = metadata
        .features
        .iter()
        .map(|f| get_feature_required_days(f))
        .max()
        .unwrap_or(20);
    let lookback_window = required_days.max(30).min(prices.len().saturating_sub(1));
    let start_idx = lookback_window;
    let end_idx = prices.len() - 1;

    if start_idx > end_idx {
        return Err("历史数据不足以生成特征".to_string());
    }

    // 基于最近一段窗口，构建特征矩阵（用于标准化）
    let mut features_matrix: Vec<Vec<f64>> = Vec::with_capacity(end_idx - start_idx + 1);
    for i in start_idx..=end_idx {
        let mut feature_vector = Vec::with_capacity(metadata.features.len());
        for feature_name in &metadata.features {
            let value = calculate_feature_value(
                feature_name,
                &prices,
                &volumes,
                i,
                lookback_window,
                Some(&highs),
                Some(&lows),
            ).map_err(|e| format!("计算特征 '{feature_name}' 失败: {e}"))?;
            feature_vector.push(value);
        }
        features_matrix.push(feature_vector);
    }

    // 标准化，并选取最新一行作为推理输入
    let normalized_matrix = normalize_features_local(&features_matrix);
    let last_normalized_row = normalized_matrix
        .last()
        .cloned()
        .ok_or_else(|| "标准化特征为空".to_string())?;

    // 创建输入张量
    let features_f32: Vec<f32> = last_normalized_row.iter().map(|&x| x as f32).collect();
    let input_tensor = Tensor::from_slice(&features_f32, &[1, metadata.features.len()], &device)
        .map_err(|e| format!("创建输入张量失败: {e}"))?;
    
    // 进行预测（基础模型输出变化率）
    let output = model.forward(&input_tensor)
        .map_err(|e| format!("预测失败: {e}"))?;
    
    let raw_change_rate = match output.dims() {
        [_] => {
            output.to_vec1::<f32>().map_err(|e| format!("获取预测结果失败: {e}"))?[0] as f64
        },
        [_, n] => {
            if *n == 1 {
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {e}"))?[0][0] as f64
            } else {
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {e}"))?[0][0] as f64
            }
        },
        _ => {
            return Err(format!("预测输出维度不支持: {:?}", output.dims()));
        }
    };
    
    // 计算历史数据特征
    let historical_volatility = calculate_historical_volatility(&prices);
    let _recent_trend = calculate_recent_trend(&prices);
    let support_resistance = calculate_support_resistance_levels(&prices, &highs, &lows, current_price);
    let _volatility_features = analyze_historical_volatility_pattern(&prices, 30);
    let mut technical_signals = analyze_technical_signals(&prices, &highs, &lows, &volumes);
    
    // 计算多因子评分（用于预测调整）
    use crate::stock_prediction::technical_indicators::{
        calculate_market_sentiment, calculate_atr, calculate_dmi_adx
    };
    
    let n = prices.len();
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    let ma60 = calc_ma(60);
    
    // 计算RSI
    let rsi = if n >= 14 {
        let recent_prices = &prices[n-14..];
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 { gains += change; } else { losses += -change; }
        }
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        if avg_loss == 0.0 { 100.0 } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    } else { 50.0 };
    
    // 计算MACD
    let ema12 = calc_ma(12);
    let ema26 = calc_ma(26);
    let macd_dif = ema12 - ema26;
    let macd_dea = calc_ma(9);
    
    // 计算市场情绪和波动率
    let market_sentiment = calculate_market_sentiment(&prices, &volumes, &highs, &lows);
    let atr = calculate_atr(&highs, &lows, &prices, 14);
    let volatility_pct = (atr / current_price) * 100.0;
    let (_di_plus, _di_minus, adx) = calculate_dmi_adx(&highs, &lows, &prices, 14);
    
    // 计算各因子得分
    use crate::stock_prediction::multi_factor_scoring;
    use crate::stock_prediction::candlestick_patterns;
    use crate::stock_prediction::volume_analysis;
    
    let candles: Vec<candlestick_patterns::Candle> = prices.iter().zip(highs.iter()).zip(lows.iter()).zip(volumes.iter())
        .map(|(((& close, &high), &low), &volume)| {
            candlestick_patterns::Candle {
                open: close,
                high,
                low,
                close,
                volume,
            }
        }).collect();
    let candle_patterns = candlestick_patterns::identify_all_patterns(&candles);
    let volume_analysis_result = volume_analysis::analyze_volume_price_enhanced(&prices, &volumes, &highs, &lows);
    
    let trend_factor = multi_factor_scoring::score_trend_factor(ma5, ma10, ma20, ma60, current_price);
    let obv_last = volume_analysis_result.obv.last().copied().unwrap_or(0.0);
    let obv_trend_str = if obv_last > 0.0 { "上升".to_string() } else { "下降".to_string() };
    let volume_factor = multi_factor_scoring::score_volume_factor(
        &volume_analysis_result.volume_trend,
        volume_analysis_result.volume_price_sync,
        volume_analysis_result.accumulation_signal,
        &obv_trend_str,
    );
    let pattern_factor = multi_factor_scoring::score_pattern_factor(&candle_patterns);
    let momentum_factor = multi_factor_scoring::score_momentum_factor(rsi, macd_dif, macd_dea);
    let sr_factor = multi_factor_scoring::score_support_resistance_factor(
        current_price,
        &support_resistance.support_levels,
        &support_resistance.resistance_levels,
    );
    
    // 构建多周期数据
    let mut daily_data: Vec<StockData> = Vec::with_capacity(prices.len());
    for (i, date) in dates.iter().enumerate() {
        daily_data.push(StockData {
            symbol: request.stock_code.clone(),
            date: date.clone(),
            open: prices[i],
            high: highs.get(i).copied().unwrap_or(prices[i]),
            low: lows.get(i).copied().unwrap_or(prices[i]),
            close: prices[i],
            volume: volumes.get(i).copied().unwrap_or(0) as f64,
        });
    }
    
    let multi_timeframe = analyze_multi_timeframe_resonance(&daily_data);
    
    let mtf_factor = multi_factor_scoring::score_multi_timeframe_factor(
        multi_timeframe.resonance_level,
        &multi_timeframe.resonance_direction,
        multi_timeframe.signal_quality,
    );
    let sentiment_factor = multi_factor_scoring::score_sentiment_factor(
        market_sentiment.sentiment_score,
        market_sentiment.fear_greed_index,
        &market_sentiment.market_phase,
    );
    let volatility_factor = multi_factor_scoring::score_volatility_factor(atr, current_price);
    
    let mut factors = vec![
        trend_factor, volume_factor, pattern_factor, momentum_factor,
        sr_factor, mtf_factor, sentiment_factor, volatility_factor,
    ];
    
    // 智能权重调整
    multi_factor_scoring::adjust_factor_weights(&mut factors, &market_sentiment.market_phase, volatility_pct, adx);
    let multi_factor_score = multi_factor_scoring::calculate_multi_factor_score(factors);
    
    // 生成预测
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(dates.last().unwrap_or(&"2023-01-01".to_string()), "%Y-%m-%d")
        .unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    for day in 1..=request.prediction_days {
        // 确保预测日期为A股交易日
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // === 改进的预测算法：基于趋势分析 + 均线/量能融合 ===
        
        // 1. 基础模型预测（权重降低）
        let base_model_prediction = raw_change_rate * BASE_MODEL_WEIGHT;
        
        // 2. 趋势主导因子（大幅提高权重）
        let trend_bias = trend_analysis.trend_strength * TREND_STRENGTH_MAPPING;
        let trend_factor = trend_bias * trend_analysis.bias_multiplier * TREND_BIAS_WEIGHT;
        
        // 3. 技术指标确认（与趋势配合）：使用配置常量
        let tech_decay = TECH_DECAY_BASE.powi(day as i32);
        let technical_impact = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * TECH_STRONG_ALIGNED_WEIGHT * tech_decay
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * TECH_STRONG_CONFLICT_WEIGHT * tech_decay
                } else {
                    technical_signals.signal_strength * TECH_STRONG_NEUTRAL_WEIGHT * tech_decay
                }
            },
            TrendState::StrongBearish | TrendState::Bearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * TECH_STRONG_ALIGNED_WEIGHT * tech_decay
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * TECH_STRONG_CONFLICT_WEIGHT * tech_decay
                } else {
                    technical_signals.signal_strength * TECH_STRONG_NEUTRAL_WEIGHT * tech_decay
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross 
                    || technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * TECH_NEUTRAL_CROSS_WEIGHT * tech_decay
                } else {
                    technical_signals.signal_strength * TECH_NEUTRAL_WEIGHT * tech_decay
                }
            }
        };
        
        // 4. 均线与量能偏置（新增）：MA5/10/20 与量比
        let mut ma_bias: f64 = 0.0;
        let mut vol_bias: f64 = 0.0;
        if prices.len() >= 21 && volumes.len() >= 21 {
            let n = prices.len();
            let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
            let ma5 = avg(&prices[n-5..n]);
            let ma10 = avg(&prices[n-10..n]);
            let ma20 = avg(&prices[n-20..n]);
            let price = last_price;

            // 均线位置与多空排列：使用配置常量
            if price > ma5 { ma_bias += PRICE_VS_MA5_BIAS; } else { ma_bias -= PRICE_VS_MA5_BIAS; }
            if ma5 > ma10 { ma_bias += MA5_VS_MA10_BIAS; } else { ma_bias -= MA5_VS_MA10_BIAS; }
            if ma10 > ma20 { ma_bias += MA10_VS_MA20_BIAS; } else { ma_bias -= MA10_VS_MA20_BIAS; }

            // 均线斜率：使用配置常量
            let prev_ma5 = avg(&prices[n-6..n-1]);
            let prev_ma10 = avg(&prices[n-11..n-1]);
            let prev_ma20 = avg(&prices[n-21..n-1]);
            if ma5 > prev_ma5 { ma_bias += MA5_SLOPE_BIAS; } else { ma_bias -= MA5_SLOPE_BIAS; }
            if ma10 > prev_ma10 { ma_bias += MA10_SLOPE_BIAS; } else { ma_bias -= MA10_SLOPE_BIAS; }
            if ma20 > prev_ma20 { ma_bias += MA20_SLOPE_BIAS; } else { ma_bias -= MA20_SLOPE_BIAS; }
            ma_bias = ma_bias.clamp(-2.0, 2.0) * MA_BIAS_MAPPING;

            // 量能偏置：5日/20日量比：使用配置常量
            let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
            let v5 = avgv(&volumes[n-5..n]);
            let v20 = avgv(&volumes[n-20..n]);
            let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
            if vr > HIGH_VOLUME_RATIO_THRESHOLD { vol_bias += HIGH_VOLUME_RATIO_BIAS; }
            else if vr > MEDIUM_VOLUME_RATIO_THRESHOLD { vol_bias += MEDIUM_VOLUME_RATIO_BIAS; }
            if vr < VERY_LOW_VOLUME_RATIO_THRESHOLD { vol_bias -= HIGH_VOLUME_RATIO_BIAS; }
            else if vr < LOW_VOLUME_RATIO_THRESHOLD { vol_bias -= MEDIUM_VOLUME_RATIO_BIAS; }
        }
        let ma_vol_decay = MA_VOLUME_DECAY_BASE.powi(day as i32);
        
        // 5. 波动率调整（根据趋势一致性调整）：使用配置常量
        let volatility_factor = historical_volatility.clamp(0.01, 0.08);
        let trend_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                if (technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBullish) ||
                   (technical_signals.macd_death_cross || technical_signals.kdj_death_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBearish) {
                    STRONG_ALIGNED_DECAY_BASE.powi(day as i32)
                } else {
                    STRONG_CONFLICT_DECAY_BASE.powi(day as i32)
                }
            },
            TrendState::Bullish | TrendState::Bearish => NORMAL_TREND_DECAY_BASE.powi(day as i32),
            TrendState::Neutral => NEUTRAL_TREND_DECAY_BASE.powi(day as i32),
        };
        
        // 6. 随机扰动（轻微减小幅度，避免噪声将方向推向上涨）：使用配置常量
        let noise_amplitude = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross || 
                    technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    volatility_factor * STRONG_ALIGNED_VOLATILITY_FACTOR
                } else {
                    volatility_factor * STRONG_CONFLICT_VOLATILITY_FACTOR
                }
            },
            TrendState::Bullish | TrendState::Bearish => volatility_factor * NORMAL_TREND_VOLATILITY_FACTOR,
            TrendState::Neutral => volatility_factor * NEUTRAL_TREND_VOLATILITY_FACTOR,
        };
        
        // 使用确定性的市场波动（基于历史波动率和预测天数）：使用配置常量
        // 金融逻辑：市场总有波动，但波动是确定性的（基于历史数据）
        let market_fluctuation = {
            // 基于预测天数的确定性波动因子
            let day_factor = (day as f64 * GOLDEN_RATIO_OSCILLATION).sin() * VOLATILITY_NORMALIZATION + VOLATILITY_NORMALIZATION;
            noise_amplitude * (day_factor - VOLATILITY_NORMALIZATION) * 2.0
        };
        
        // 7. 综合预测变化率：使用配置常量
        let mut predicted_change_rate = base_model_prediction * PREDICTION_BASE_MODEL_RATIO
            + trend_factor * trend_decay * PREDICTION_TREND_RATIO
            + technical_impact * PREDICTION_TECHNICAL_RATIO
            + (ma_bias + vol_bias) * ma_vol_decay * PREDICTION_MA_VOLUME_RATIO
            + market_fluctuation * PREDICTION_MARKET_FLUCTUATION_RATIO;
        
        // 8. 趋势一致性增强（特别重视日线金叉死叉）：使用配置常量
        match trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= STRONG_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate += STRONG_ALIGNMENT_BIAS;
                } else {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= MEDIUM_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate += MEDIUM_ALIGNMENT_BIAS;
                }
            },
            TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= WEAK_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate += WEAK_ALIGNMENT_BIAS;
                } else {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= MINIMAL_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate += MINIMAL_ALIGNMENT_BIAS;
                }
            },
            TrendState::StrongBearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= STRONG_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate -= STRONG_ALIGNMENT_BIAS;
                } else {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= MEDIUM_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate -= MEDIUM_ALIGNMENT_BIAS;
                }
            },
            TrendState::Bearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= WEAK_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate -= WEAK_ALIGNMENT_BIAS;
                } else {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= MINIMAL_ALIGNMENT_OPPOSITE_SUPPRESS; }
                    predicted_change_rate -= MINIMAL_ALIGNMENT_BIAS;
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    predicted_change_rate += NEUTRAL_CROSS_BIAS;
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    predicted_change_rate -= NEUTRAL_CROSS_BIAS;
                }
            }
        };
        
        // 9. 支撑压力位调整（新增）- 根据价格位置调整预测强度
        let price_for_check = last_price * (1.0 + predicted_change_rate);
        
        // 检查接近压力位（上涨预测时）
        if predicted_change_rate > 0.0 && !support_resistance.resistance_levels.is_empty() {
            let mut resistance_adjustment: f64 = 1.0;
            
            for &resistance in &support_resistance.resistance_levels {
                let distance_pct = (resistance / price_for_check - 1.0).abs();
                
                if distance_pct < 0.02 {  // 接近强压力位 (<2%)
                    resistance_adjustment = f64::min(resistance_adjustment, NEAR_STRONG_RESISTANCE_DECAY);
                    break;
                } else if distance_pct < 0.05 {  // 接近普通压力位 (2%-5%)
                    resistance_adjustment = f64::min(resistance_adjustment, NEAR_RESISTANCE_DECAY);
                }
            }
            
            // 检查是否突破压力位
            if prices.len() >= 2 {
                let prev_price = prices[prices.len() - 1];
                if let Some(&first_resistance) = support_resistance.resistance_levels.first() {
                    if prev_price < first_resistance && price_for_check > first_resistance {
                        // 突破压力位，检查成交量
                        if volumes.len() >= 5 {
                            let n = volumes.len();
                            let recent_volume = volumes[n-1] as f64;
                            let avg_volume = volumes[n.saturating_sub(5)..n].iter().map(|&v| v as f64).sum::<f64>() / 5.0;
                            if recent_volume > avg_volume * 1.2 {  // 放量突破
                                resistance_adjustment = BREAKOUT_ACCELERATION;
                            }
                        }
                    }
                }
            }
            
            predicted_change_rate *= resistance_adjustment;
        }
        
        // 检查接近支撑位（下跌预测时）
        if predicted_change_rate < 0.0 && !support_resistance.support_levels.is_empty() {
            let mut support_adjustment: f64 = 1.0;
            
            for &support in &support_resistance.support_levels {
                let distance_pct = (price_for_check / support - 1.0).abs();
                
                if distance_pct < 0.02 {  // 接近强支撑位 (<2%)
                    support_adjustment = f64::min(support_adjustment, NEAR_STRONG_SUPPORT_PROTECTION);
                    break;
                } else if distance_pct < 0.05 {  // 接近普通支撑位 (2%-5%)
                    support_adjustment = f64::min(support_adjustment, NEAR_SUPPORT_PROTECTION);
                }
            }
            
            // 检查是否跌破支撑位
            if prices.len() >= 2 {
                let prev_price = prices[prices.len() - 1];
                if let Some(&first_support) = support_resistance.support_levels.first() {
                    if prev_price > first_support && price_for_check < first_support {
                        // 跌破支撑位，检查成交量
                        if volumes.len() >= 5 {
                            let n = volumes.len();
                            let recent_volume = volumes[n-1] as f64;
                            let avg_volume = volumes[n.saturating_sub(5)..n].iter().map(|&v| v as f64).sum::<f64>() / 5.0;
                            if recent_volume > avg_volume * 1.2 {  // 放量跌破
                                predicted_change_rate *= BREAKDOWN_ACCELERATION;
                            } else {
                                predicted_change_rate *= support_adjustment;
                            }
                        } else {
                            predicted_change_rate *= support_adjustment;
                        }
                    } else {
                        predicted_change_rate *= support_adjustment;
                    }
                } else {
                    predicted_change_rate *= support_adjustment;
                }
            } else {
                predicted_change_rate *= support_adjustment;
            }
        }
        
        // 10. 多因子评分调整（新增）- 根据综合评分直接影响涨跌幅
        let score = multi_factor_score.total_score;
        if score > 75.0 {
            // 强烈看涨 (评分>75)
            predicted_change_rate += (score - 75.0) * MULTI_FACTOR_STRONG_BULLISH_IMPACT;
        } else if score >= 60.0 {
            // 看涨 (评分60-75)
            predicted_change_rate += (score - 60.0) * MULTI_FACTOR_BULLISH_IMPACT;
        } else if score >= 40.0 {
            // 中性 (评分40-60)
            predicted_change_rate += MULTI_FACTOR_NEUTRAL_BIAS;
        } else if score >= 25.0 {
            // 看跌 (评分25-40)
            predicted_change_rate -= (40.0 - score) * MULTI_FACTOR_BEARISH_IMPACT;
        } else {
            // 强烈看跌 (评分<25)
            predicted_change_rate -= (0.005 + (25.0 - score) * MULTI_FACTOR_STRONG_BEARISH_IMPACT);
        }
        
        // 11. 波动率调节（新增）- 根据市场波动率调整预测置信度
        if volatility_pct > 8.0 {
            // 极高波动率
            predicted_change_rate *= EXTREME_VOLATILITY_SUPPRESS;
        } else if volatility_pct > 5.0 {
            // 高波动率
            predicted_change_rate *= HIGH_VOLATILITY_SUPPRESS;
        } else if volatility_pct < 1.5 {
            // 低波动率
            predicted_change_rate *= LOW_VOLATILITY_ENHANCE;
        }
        
        // 12. 市场情绪调节（新增）- 根据恐惧贪婪指数调整
        if market_sentiment.fear_greed_index < 20.0 && predicted_change_rate > 0.0 {
            // 极度恐慌时的反向买入信号增强
            predicted_change_rate *= EXTREME_FEAR_CONTRARIAN_BOOST;
            if market_sentiment.market_phase.contains("恐慌") {
                predicted_change_rate += PANIC_REVERSAL_BONUS;
            }
        } else if market_sentiment.fear_greed_index > 80.0 && predicted_change_rate > 0.0 {
            // 极度贪婪时的上涨预测抑制
            predicted_change_rate *= EXTREME_GREED_SUPPRESS;
        }
        
        if market_sentiment.market_phase.contains("过热") && predicted_change_rate < 0.0 {
            // 过热期的回调机会加成
            predicted_change_rate -= OVERHEATED_CORRECTION_BONUS;
        }

        // === 新增：方向投票（优先保证涨/跌判断的合理性） ===
        let (direction_prob_up, _direction_score) = {
            let mut score: f64 = 0.0;

            // 趋势权重：使用配置常量
            score += match trend_analysis.overall_trend {
                TrendState::StrongBullish => STRONG_TREND_VOTE_SCORE,
                TrendState::Bullish => NORMAL_TREND_VOTE_SCORE,
                TrendState::Neutral => 0.0,
                TrendState::Bearish => -NORMAL_TREND_VOTE_SCORE,
                TrendState::StrongBearish => -STRONG_TREND_VOTE_SCORE,
            };

            // MACD 权重（当下日线）：使用配置常量
            if technical_signals.macd_golden_cross { score += MACD_CROSS_VOTE_SCORE; }
            if technical_signals.macd_death_cross { score -= MACD_CROSS_VOTE_SCORE; }
            if technical_signals.macd_histogram > 0.0 { score += MACD_HISTOGRAM_VOTE_SCORE; } else { score -= MACD_HISTOGRAM_VOTE_SCORE; }
            if technical_signals.macd_zero_cross_up { score += MACD_ZERO_CROSS_VOTE_SCORE; }
            if technical_signals.macd_zero_cross_down { score -= MACD_ZERO_CROSS_VOTE_SCORE; }

            // KDJ 权重（当下日线）：使用配置常量
            if technical_signals.kdj_golden_cross { score += KDJ_CROSS_VOTE_SCORE; }
            if technical_signals.kdj_death_cross { score -= KDJ_CROSS_VOTE_SCORE; }
            if technical_signals.kdj_j > 80.0 { score -= KDJ_EXTREME_VOTE_SCORE; }
            if technical_signals.kdj_j < 20.0 { score += KDJ_EXTREME_VOTE_SCORE; }

            // RSI 权重：使用配置常量
            if technical_signals.rsi > 70.0 { score -= RSI_EXTREME_VOTE_SCORE; }
            else if technical_signals.rsi > 55.0 { score -= RSI_MODERATE_VOTE_SCORE; }
            else if technical_signals.rsi < 30.0 { score += RSI_EXTREME_VOTE_SCORE; }
            else if technical_signals.rsi < 45.0 { score += RSI_MODERATE_VOTE_SCORE; }

            // 均线排列与斜率
            if prices.len() >= 21 {
                let n = prices.len();
                let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
                let ma5 = avg(&prices[n-5..n]);
                let ma10 = avg(&prices[n-10..n]);
                let ma20 = avg(&prices[n-20..n]);
                if ma5 > ma10 && ma10 > ma20 { score += 1.0; }
                if ma5 < ma10 && ma10 < ma20 { score -= 1.0; }
                let prev_ma5 = avg(&prices[n-6..n-1]);
                if ma5 > prev_ma5 { score += 0.3; } else { score -= 0.3; }
            }

            // 量比与 OBV
            if volumes.len() >= 20 {
                let n = volumes.len();
                let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
                let v5 = avgv(&volumes[n-5..n]);
                let v20 = avgv(&volumes[n-20..n]);
                let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
                if vr > 1.2 { score += 0.3; }
                if vr < 0.8 { score -= 0.3; }
            }
            if technical_signals.obv > 0.0 { score += 0.2; } else { score -= 0.2; }

            // 多周期信号融合（日/周/月）
            // 构造日线数据
            let mut daily_data: Vec<StockData> = Vec::with_capacity(prices.len());
            for (i, date) in dates.iter().enumerate() {
                // 缺少开盘价，这里用收盘价近似
                let open_approx = prices[i];
                daily_data.push(StockData {
                    symbol: request.stock_code.clone(),
                    date: date.clone(),
                    open: open_approx,
                    high: highs.get(i).copied().unwrap_or(prices[i]),
                    low: lows.get(i).copied().unwrap_or(prices[i]),
                    close: prices[i],
                    volume: volumes.get(i).copied().unwrap_or(0) as f64,
                });
            }
            let weekly_data = convert_to_weekly(&daily_data);
            let monthly_data = convert_to_monthly(&daily_data);

            let daily_macd = calculate_macd_signal(&daily_data, 12, 26, 9);
            let weekly_macd = calculate_macd_signal(&weekly_data, 12, 26, 9);
            let monthly_macd = calculate_macd_signal(&monthly_data, 12, 26, 9);
            let daily_kdj = calculate_kdj_signal(&daily_data, 9, 3, 3);
            let weekly_kdj = calculate_kdj_signal(&weekly_data, 9, 3, 3);
            let monthly_kdj = calculate_kdj_signal(&monthly_data, 9, 3, 3);

            if let (Some(dm), Some(wm), Some(mm)) = (daily_macd.last(), weekly_macd.last(), monthly_macd.last()) {
                if dm.is_golden_cross { score += 0.5; } else if dm.is_death_cross { score -= 0.5; }
                if wm.is_golden_cross { score += 0.8; } else if wm.is_death_cross { score -= 0.8; }
                if mm.is_golden_cross { score += 1.2; } else if mm.is_death_cross { score -= 1.2; }
                if dm.histogram > 0.0 { score += 0.2; } else { score -= 0.2; }
                if wm.histogram > 0.0 { score += 0.35; } else { score -= 0.35; }
                if mm.histogram > 0.0 { score += 0.5; } else { score -= 0.5; }
            }
            if let (Some(dk), Some(wk), Some(mk)) = (daily_kdj.last(), weekly_kdj.last(), monthly_kdj.last()) {
                if dk.is_golden_cross { score += 0.3; } else if dk.is_death_cross { score -= 0.3; }
                if wk.is_golden_cross { score += 0.5; } else if wk.is_death_cross { score -= 0.5; }
                if mk.is_golden_cross { score += 0.8; } else if mk.is_death_cross { score -= 0.8; }
                if dk.j > 80.0 { score -= 0.2; }
                if dk.j < 20.0 { score += 0.2; }
            }

            // 20日突破信号（Donchian 简化）
            if highs.len() >= 21 && lows.len() >= 21 {
                let n = highs.len();
                let max20 = highs[n-21..n-1].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min20 = lows[n-21..n-1].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let last_close = prices[n-1];
                if last_close > max20 { score += 0.8; }
                if last_close < min20 { score -= 0.8; }
            }

            let k = 0.9_f64; // 温和的放大系数
            let prob_up = 1.0 / (1.0 + (-k * score).exp());
            (prob_up, score)
        };

        // 根据方向概率调整预测方向，并设定保守幅度（更关注方向正确性）
        // 收敛阈值，避免总偏向上涨或下跌
        if direction_prob_up >= 0.60 && predicted_change_rate < 0.0 {
            predicted_change_rate = predicted_change_rate.abs();
        }
        if direction_prob_up <= 0.40 && predicted_change_rate > 0.0 {
            predicted_change_rate = -predicted_change_rate.abs();
        }
        // 使用基于波动率与趋势置信的幅度调整
        // 金融逻辑：保守预测，但保留趋势信息
        let dir_mag = (volatility_factor * (0.6 + 0.4 * trend_analysis.trend_confidence) * (0.98_f64.powi((day as i32) - 1)))
            .clamp(0.003, 0.06);
        
        // 确保有合理的变化率（金融逻辑：股价不会完全不动）
        if predicted_change_rate.abs() < 0.001 {
            // 变化率太小时，使用dir_mag作为基准
            predicted_change_rate = if direction_prob_up >= 0.5 { dir_mag } else { -dir_mag };
        } else if predicted_change_rate.abs() > dir_mag * 2.0 {
            // 变化率过大时，限制为2倍dir_mag（保守预测）
            predicted_change_rate = predicted_change_rate.signum() * dir_mag * 2.0;
        } else {
            // 变化率适中，保留原值但确保有最小幅度
            let min_mag = dir_mag * 0.3;
            if predicted_change_rate.abs() < min_mag {
                predicted_change_rate = predicted_change_rate.signum() * min_mag;
            }
        }

        // 9. 应用A股涨跌停限制
        let change_percent = clamp_daily_change(predicted_change_rate * 100.0);
        let clamped_change_rate = change_percent / 100.0;
        let predicted_price = last_price * (1.0 + clamped_change_rate);
        
        // 10. 置信度（轻降上限，避免总是高置信度买入）
        let base_confidence = (metadata.accuracy + 0.25).min(0.75);
        let trend_confidence_boost = trend_analysis.trend_confidence * 0.2;
        let volatility_impact = 1.0 - (volatility_factor * 6.0).min(0.3);
        let prediction_magnitude = 1.0 - (change_percent.abs() / 12.0).min(0.25);
        let time_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => 0.97_f64.powi(day as i32),
            _ => 0.94_f64.powi(day as i32),
        };
        
        // MA排列契合及量比对置信度的贡献
        let mut confidence_extra = 0.0;
        if prices.len() >= 21 && volumes.len() >= 21 {
            let n = prices.len();
            let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
            let ma5 = avg(&prices[n-5..n]);
            let ma10 = avg(&prices[n-10..n]);
            let ma20 = avg(&prices[n-20..n]);
            if ma5 > ma10 && ma10 > ma20 { confidence_extra += 0.03; }
            if ma5 < ma10 && ma10 < ma20 { confidence_extra += 0.03; }
            let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
            let v5 = avgv(&volumes[n-5..n]);
            let v20 = avgv(&volumes[n-20..n]);
            let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
            if vr > 1.2 { confidence_extra += 0.02; }
        }
        
        let confidence = (base_confidence 
            * volatility_impact 
            * prediction_magnitude 
            * time_decay 
            + trend_confidence_boost
            + confidence_extra)
            .clamp(0.35, 0.92);
        
        // 11. 交易信号（结合趋势状态和日线技术指标）
        let trading_signal_str = match &trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "强烈买入"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "强烈买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有"
                } else {
                    "买入"
                }
            },
            TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有"
                } else {
                    "买入"
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出"
                } else {
                    match &technical_signals.signal {
                        TradingSignal::StrongBuy => "买入",
                        TradingSignal::Buy => "买入",
                        TradingSignal::Hold => "持有",
                        TradingSignal::Sell => "卖出", 
                        TradingSignal::StrongSell => "卖出",
                    }
                }
            },
            TrendState::Bearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有"
                } else {
                    "卖出"
                }
            },
            TrendState::StrongBearish => {
                if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "强烈卖出"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "强烈卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有"
                } else {
                    "卖出"
                }
            },
        };
        
        let technical_indicators = TechnicalIndicatorValues {
            rsi: technical_signals.rsi,
            macd_histogram: technical_signals.macd_histogram,
            kdj_j: technical_signals.kdj_j,
            cci: technical_signals.cci,
            obv_trend: if technical_signals.obv > 0.0 { 1.0 } else { -1.0 },
            macd_dif: technical_signals.macd_dif,
            macd_dea: technical_signals.macd_dea,
            kdj_k: technical_signals.kdj_k,
            kdj_d: technical_signals.kdj_d,
            macd_golden_cross: technical_signals.macd_golden_cross,
            macd_death_cross: technical_signals.macd_death_cross,
            kdj_golden_cross: technical_signals.kdj_golden_cross,
            kdj_death_cross: technical_signals.kdj_death_cross,
            kdj_overbought: technical_signals.kdj_overbought,
            kdj_oversold: technical_signals.kdj_oversold,
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(trading_signal_str.to_string()),
            signal_strength: Some(trend_analysis.trend_strength),
            technical_indicators: Some(technical_indicators),
            prediction_reason: None,  // 主预测函数暂不生成理由
            key_factors: None,
        });
        
        last_price = predicted_price;
        
        // 更新价格序列以便下一天预测
        if day < request.prediction_days {
            prices.push(predicted_price);
            highs.push(predicted_price * 1.005);
            lows.push(predicted_price * 0.995);
            
            if let Some(&last_volume) = volumes.last() {
                // 使用确定性的成交量变化（基于趋势方向）
                let volume_change = match trend_analysis.overall_trend {
                    TrendState::StrongBullish => 1.08,  // 强势上涨：成交量明显增加
                    TrendState::Bullish => 1.05,        // 上涨趋势：成交量略增
                    TrendState::Neutral => 1.0,         // 震荡：成交量持平
                    TrendState::Bearish => 0.95,        // 下跌趋势：成交量略减
                    TrendState::StrongBearish => 0.92,  // 强势下跌：成交量明显减少
                };
                volumes.push((last_volume as f64 * volume_change) as i64);
            }
            
            technical_signals = analyze_technical_signals(&prices, &highs, &lows, &volumes);
        }
    }
    
    // 构建最新真实数据
    let last_real_data = if !dates.is_empty() {
        Some(LastRealData {
            date: dates.last().unwrap().clone(),
            price: current_price,
            change_percent: current_change_percent,
        })
    } else {
        None
    };
    
    // 打印对比信息
    if !predictions.is_empty() {
        print_last_real_vs_prediction(&dates, &prices, &technical_signals, &predictions[0], current_change_percent);
    }
    
    println!("\n🎯 预测完成! 基于{}趋势进行了{}天预测", 
             match trend_analysis.overall_trend {
                 TrendState::StrongBullish => "强烈上涨",
                 TrendState::Bullish => "上涨",
                 TrendState::Neutral => "震荡",
                 TrendState::Bearish => "下跌",
                 TrendState::StrongBearish => "强烈下跌",
             },
             request.prediction_days);
    
    Ok(PredictionResponse {
        predictions,
        last_real_data,
    })
}

// 基于量价关系的预测函数 - 专注于核心要素
pub async fn predict_with_simple_strategy(request: PredictionRequest) -> std::result::Result<PredictionResponse, String> {
    // 获取最近的真实市场数据
    let (current_price, current_change_percent, dates, prices, volumes, highs, lows) = get_recent_market_data(&request.stock_code, 60).await
        .map_err(|e| format!("获取市场数据失败: {e}"))?;
    
    if prices.len() < 10 {
        return Err("历史数据不足，无法进行预测，需要至少10天数据".to_string());
    }
    
    println!("🎯 使用量价关系策略进行预测:");
    println!("   📊 历史数据: {}天", prices.len());
    println!("   💰 当前价格: {current_price:.2}元");
    
    // 使用量价关系预测策略
    let volume_price_strategy = predict_with_volume_price(&prices, &highs, &lows, &volumes, current_price);
    
    // 计算历史波动率
    let historical_volatility = calculate_historical_volatility(&prices);
    
    // 生成预测
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(dates.last().unwrap_or(&"2023-01-01".to_string()), "%Y-%m-%d")
        .unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    // 为每一天生成预测
    for day in 1..=request.prediction_days {
        // 确保预测日期为A股交易日
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 使用量价关系计算涨跌幅
        let (predicted_change_percent, confidence) = calculate_volume_price_change(&volume_price_strategy, historical_volatility, day as i32);
        let predicted_price = last_price * (1.0 + predicted_change_percent / 100.0);
        
        // 交易信号直接来自量价策略
        let trading_signal = &volume_price_strategy.volume_price_signal;
        
        // 创建技术指标值（基于量价关系）
        let technical_indicators = TechnicalIndicatorValues {
            rsi: 50.0,
            macd_histogram: 0.0,
            kdj_j: 50.0,
            cci: 0.0,
            obv_trend: 0.0,
            macd_dif: 0.0,
            macd_dea: 0.0,
            kdj_k: 50.0,
            kdj_d: 50.0,
            macd_golden_cross: volume_price_strategy.predicted_direction == "上涨",
            macd_death_cross: volume_price_strategy.predicted_direction == "下跌",
            kdj_golden_cross: volume_price_strategy.predicted_direction == "上涨",
            kdj_death_cross: volume_price_strategy.predicted_direction == "下跌",
            kdj_overbought: false,
            kdj_oversold: false,
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent,
            confidence,
            trading_signal: Some(trading_signal.clone()),
            signal_strength: Some(volume_price_strategy.direction_confidence),
            technical_indicators: Some(technical_indicators),
            prediction_reason: None,  // 量价策略暂不生成理由
            key_factors: None,
        });
        
        last_price = predicted_price;
    }
    
    // 构建最新真实数据
    let last_real_data = if !dates.is_empty() {
        Some(LastRealData {
            date: dates.last().unwrap().clone(),
            price: current_price,
            change_percent: current_change_percent,
        })
    } else {
        None
    };
    
    println!("\n✅ 量价关系预测完成!");
    println!("   🎯 预测方向: {} (置信度: {:.0}%)", 
             volume_price_strategy.predicted_direction, 
             volume_price_strategy.direction_confidence * 100.0);
    println!("   📊 价格趋势: {}", volume_price_strategy.price_trend);
    println!("   📊 成交量趋势: {}", volume_price_strategy.volume_trend);
    println!("   💡 量价信号: {}", volume_price_strategy.volume_price_signal);
    println!("   📋 关键因素: {}", volume_price_strategy.key_factors.join(", "));
    println!("   📈 预测天数: {}天", request.prediction_days);
    
    Ok(PredictionResponse {
        predictions,
        last_real_data,
    })
} 

// ==================== 金融级预测主函数 ====================

/// 金融级预测策略 - 提供买卖点和专业分析
pub async fn predict_with_professional_strategy(
    request: PredictionRequest
) -> std::result::Result<(PredictionResponse, ProfessionalPrediction), String> {
    println!("\n🎯 ========== 金融级策略分析 ==========");
    
    // 获取最近的真实市场数据
    let (current_price, current_change_percent, dates, prices, volumes, highs, lows) = 
        get_recent_market_data(&request.stock_code, 120).await
        .map_err(|e| format!("获取市场数据失败: {e}"))?;
    
    if prices.len() < 60 {
        return Err("历史数据不足，需要至少60天数据进行专业分析".to_string());
    }
    
    println!("📊 数据加载完成: {}天历史数据", prices.len());
    println!("💰 当前价格: {:.2}元 ({:+.2}%)\n", current_price, current_change_percent);
    
    // 1. 计算支撑压力位
    let support_resistance = calculate_support_resistance_levels(
        &prices, &highs, &lows, current_price
    );
    
    println!("📍 ========== 支撑压力位分析 ==========");
    println!("   当前位置: {}", support_resistance.current_position);
    if !support_resistance.support_levels.is_empty() {
        println!("   🟢 关键支撑位:");
        for (i, &level) in support_resistance.support_levels.iter().enumerate() {
            let distance = (current_price - level) / current_price * 100.0;
            println!("      {}. {:.2}元 (距离-{:.2}%)", i+1, level, distance);
        }
    }
    if !support_resistance.resistance_levels.is_empty() {
        println!("   🔴 关键压力位:");
        for (i, &level) in support_resistance.resistance_levels.iter().enumerate() {
            let distance = (level - current_price) / current_price * 100.0;
            println!("      {}. {:.2}元 (距离+{:.2}%)", i+1, level, distance);
        }
    }
    
    // 2. 构建多周期数据
    let mut daily_data: Vec<StockData> = Vec::with_capacity(prices.len());
    for (i, date) in dates.iter().enumerate() {
        daily_data.push(StockData {
            symbol: request.stock_code.clone(),
            date: date.clone(),
            open: prices[i],
            high: highs.get(i).copied().unwrap_or(prices[i]),
            low: lows.get(i).copied().unwrap_or(prices[i]),
            close: prices[i],
            volume: volumes.get(i).copied().unwrap_or(0) as f64,
        });
    }
    
    // 3. 多周期共振分析
    let multi_timeframe = analyze_multi_timeframe_resonance(&daily_data);
    
    println!("\n🔄 ========== 多周期共振分析 ==========");
    println!("   📈 日线趋势: {}", multi_timeframe.daily_trend);
    println!("   📊 周线趋势: {}", multi_timeframe.weekly_trend);
    println!("   📉 月线趋势: {}", multi_timeframe.monthly_trend);
    println!("   ⚡ 共振级别: {} ({})", 
             multi_timeframe.resonance_level,
             multi_timeframe.resonance_direction);
    println!("   ✨ 信号质量: {:.0}分", multi_timeframe.signal_quality);
    
    // 4. 量价背离分析
    let divergence = analyze_volume_price_divergence(&prices, &volumes, &highs, &lows);
    
    println!("\n⚠️  ========== 量价背离分析 ==========");
    if divergence.has_bullish_divergence {
        println!("   🟢 检测到底背离 (强度: {:.0}%)", divergence.divergence_strength * 100.0);
    }
    if divergence.has_bearish_divergence {
        println!("   🔴 检测到顶背离 (强度: {:.0}%)", divergence.divergence_strength * 100.0);
    }
    println!("   💡 {}", divergence.warning_message);
    
    // 5. 识别买卖点
    let buy_points = identify_buy_points(
        &prices, &volumes, &highs, &lows, 
        current_price, &support_resistance, 
        &multi_timeframe, &divergence
    );
    
    let sell_points = identify_sell_points(
        &prices, &volumes, &highs, &lows,
        current_price, &support_resistance,
        &multi_timeframe, &divergence
    );
    
    println!("\n💎 ========== 买卖点信号 ==========");
    if !buy_points.is_empty() {
        println!("   🟢 买入信号 ({} 个):", buy_points.len());
        for (i, bp) in buy_points.iter().enumerate() {
            println!("      {}. {} (信号强度: {:.0}分)", i+1, bp.point_type, bp.signal_strength);
            println!("         建议价格: {:.2}元", bp.price_level);
            println!("         止损位: {:.2}元 ({:.2}%)", 
                     bp.stop_loss, 
                     (bp.stop_loss - bp.price_level) / bp.price_level * 100.0);
            println!("         止盈位: {}", 
                     bp.take_profit.iter()
                       .map(|&p| format!("{:.2}元({:+.2}%)", p, (p - bp.price_level) / bp.price_level * 100.0))
                       .collect::<Vec<_>>()
                       .join(", "));
            println!("         风险收益比: 1:{:.2}", bp.risk_reward_ratio);
            println!("         置信度: {:.0}%", bp.confidence * 100.0);
            println!("         理由: {}", bp.reasons.join("; "));
        }
    } else {
        println!("   🟡 暂无明确买入信号");
    }
    
    if !sell_points.is_empty() {
        println!("   🔴 卖出信号 ({} 个):", sell_points.len());
        for (i, sp) in sell_points.iter().enumerate() {
            println!("      {}. {} (信号强度: {:.0}分)", i+1, sp.point_type, sp.signal_strength);
            println!("         建议价格: {:.2}元", sp.price_level);
            println!("         止损位: {:.2}元 ({:+.2}%)", 
                     sp.stop_loss,
                     (sp.stop_loss - sp.price_level) / sp.price_level * 100.0);
            println!("         目标位: {}", 
                     sp.take_profit.iter()
                       .map(|&p| format!("{:.2}元({:.2}%)", p, (p - sp.price_level) / sp.price_level * 100.0))
                       .collect::<Vec<_>>()
                       .join(", "));
            println!("         置信度: {:.0}%", sp.confidence * 100.0);
            println!("         理由: {}", sp.reasons.join("; "));
        }
    } else {
        println!("   🟡 暂无明确卖出信号");
    }
    
    // 6. K线形态识别
    let candles: Vec<candlestick_patterns::Candle> = daily_data.iter().map(|d| {
        candlestick_patterns::Candle {
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
            volume: d.volume as i64,
        }
    }).collect();
    
    let candle_patterns = candlestick_patterns::identify_all_patterns(&candles);
    
    println!("\n📊 ========== K线形态识别（增强版） ==========");
    if !candle_patterns.is_empty() {
        for pattern in &candle_patterns {
            let direction_str = match pattern.direction {
                candlestick_patterns::Direction::Bullish => "🟢 看涨",
                candlestick_patterns::Direction::Bearish => "🔴 看跌",
                candlestick_patterns::Direction::Neutral => "🟡 中性",
            };
            let confirm_str = if pattern.confirmed { "✅已确认" } else { "⏳待确认" };
            println!("   {} - {} (强度: {:.0}%, 可靠性: {:.0}%) {}", 
                     direction_str,
                     pattern.description,
                     pattern.strength * 100.0,
                     pattern.reliability * 100.0,
                     confirm_str);
            println!("      位置: {} | 出现在第{}根K线", 
                     pattern.location_type,
                     pattern.position + 1);
        }
    } else {
        println!("   未检测到明显的K线形态信号");
    }
    
    // 7. 量价关系深度分析（增强版）
    let volume_analysis_raw = volume_analysis::analyze_volume_price_enhanced(&prices, &volumes, &highs, &lows);
    
    println!("\n📈 ========== 量价关系分析（增强版） ==========");
    println!("   量能趋势: {}", volume_analysis_raw.volume_trend);
    println!("   量价配合: {}", if volume_analysis_raw.volume_price_sync { "✅ 良好" } else { "⚠️ 背离" });
    println!("   吸筹信号: {:.0}分", volume_analysis_raw.accumulation_signal);
    println!("   🔥 VR量价比率: {:.1} {}", 
             volume_analysis_raw.vr_ratio,
             if volume_analysis_raw.vr_ratio > 180.0 { "(强势)" }
             else if volume_analysis_raw.vr_ratio > 120.0 { "(适中)" }
             else if volume_analysis_raw.vr_ratio > 80.0 { "(弱势)" }
             else { "(超弱)" });
    println!("   💰 MFI资金流向: {:.1}", volume_analysis_raw.mfi);
    println!("   📊 成交量形态: {}", volume_analysis_raw.volume_pattern);
    println!("   💵 资金趋势: {}", volume_analysis_raw.money_flow_trend);
    
    if volume_analysis_raw.accumulation_signal > 60.0 {
        println!("   💡 检测到主力吸筹信号！");
    }
    
    if volume_analysis_raw.mfi > 80.0 {
        println!("   ⚡ MFI超买预警！资金流入过热");
    } else if volume_analysis_raw.mfi < 20.0 {
        println!("   ⚡ MFI超卖！资金流出严重");
    }
    
    if !volume_analysis_raw.abnormal_volume_days.is_empty() {
        println!("   ⚡ 异常放量: 最近{}天有{}次异常放量", 
                 volume_analysis_raw.volume_ratio.len(),
                 volume_analysis_raw.abnormal_volume_days.len());
    }
    
    // OBV趋势判断
    let obv_trend = if volume_analysis_raw.obv.len() >= 10 {
        let recent_obv = &volume_analysis_raw.obv[volume_analysis_raw.obv.len()-10..];
        if recent_obv.last().unwrap() > recent_obv.first().unwrap() {
            "上升趋势".to_string()
        } else {
            "下降趋势".to_string()
        }
    } else {
        "数据不足".to_string()
    };
    
    let volume_analysis = VolumeAnalysisInfo {
        volume_trend: volume_analysis_raw.volume_trend.clone(),
        volume_price_sync: volume_analysis_raw.volume_price_sync,
        accumulation_signal: volume_analysis_raw.accumulation_signal,
        obv_trend,
    };
    
    // 8. 生成当前操作建议
    let (current_advice, risk_level) = generate_trading_advice(
        &buy_points,
        &sell_points,
        &multi_timeframe,
        &support_resistance,
        &divergence,
        current_price,
    );
    
    println!("\n📋 ========== 操作建议 ==========");
    println!("   {}", current_advice);
    println!("   风险等级: {}", risk_level);
    
    // 7. 生成未来价格预测（基于趋势延续）
    let predictions = generate_price_predictions(
        &request,
        &prices,
        &highs,
        &lows,
        &volumes,
        &dates,
        current_price,
        &multi_timeframe,
        &support_resistance,
    ).await?;
    
    let last_real_data = Some(LastRealData {
        date: dates.last().unwrap().clone(),
        price: current_price,
        change_percent: current_change_percent,
    });
    
    // 9. 多因子综合评分 (增强版 - 加入市场情绪和波动率)
    println!("\n🎯 ========== 多因子综合评分 (增强版) ==========");
    
    // 计算需要的技术指标
    let n = prices.len();
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    let ma60 = calc_ma(60);
    
    // 计算RSI（简化版）
    let rsi = if n >= 14 {
        let recent_prices = &prices[n-14..];
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += -change;
            }
        }
        
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        
        if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    } else {
        50.0
    };
    
    // 计算MACD（简化版）
    let ema12 = calc_ma(12);
    let ema26 = calc_ma(26);
    let macd_dif = ema12 - ema26;
    let macd_dea = calc_ma(9); // 简化，实际应该是DIF的EMA
    
    // 新增：计算市场情绪指标
    use crate::stock_prediction::technical_indicators::{
        calculate_market_sentiment, calculate_atr, calculate_dmi_adx
    };
    
    let market_sentiment = calculate_market_sentiment(&prices, &volumes, &highs, &lows);
    println!("   📊 市场情绪: {} (恐惧贪婪指数: {:.0})", 
             market_sentiment.sentiment_level,
             market_sentiment.fear_greed_index);
    println!("   📍 市场阶段: {}", market_sentiment.market_phase);
    
    // 新增：计算ATR波动率
    let atr = calculate_atr(&highs, &lows, &prices, 14);
    let volatility_pct = (atr / current_price) * 100.0;
    println!("   📈 ATR波动率: {:.2}% ({})", 
             volatility_pct,
             if volatility_pct > 3.0 { "高波动" } 
             else if volatility_pct < 1.5 { "低波动" } 
             else { "正常波动" });
    
    // 新增：计算ADX趋势强度
    let (di_plus, di_minus, adx) = calculate_dmi_adx(&highs, &lows, &prices, 14);
    println!("   💪 ADX趋势强度: {:.1} ({})", 
             adx,
             if adx > 40.0 { "强趋势" }
             else if adx > 25.0 { "中等趋势" }
             else { "弱趋势/震荡" });
    println!("   ➕ DI+: {:.1}  ➖ DI-: {:.1}", di_plus, di_minus);
    
    // 计算各因子得分
    let trend_factor = multi_factor_scoring::score_trend_factor(
        ma5, ma10, ma20, ma60, current_price
    );
    
    let volume_factor = multi_factor_scoring::score_volume_factor(
        &volume_analysis.volume_trend,
        volume_analysis.volume_price_sync,
        volume_analysis.accumulation_signal,
        &volume_analysis.obv_trend,
    );
    
    let pattern_factor = multi_factor_scoring::score_pattern_factor(&candle_patterns);
    
    let momentum_factor = multi_factor_scoring::score_momentum_factor(
        rsi, macd_dif, macd_dea
    );
    
    let sr_factor = multi_factor_scoring::score_support_resistance_factor(
        current_price,
        &support_resistance.support_levels,
        &support_resistance.resistance_levels,
    );
    
    let mtf_factor = multi_factor_scoring::score_multi_timeframe_factor(
        multi_timeframe.resonance_level,
        &multi_timeframe.resonance_direction,
        multi_timeframe.signal_quality,
    );
    
    // 新增因子
    let sentiment_factor = multi_factor_scoring::score_sentiment_factor(
        market_sentiment.sentiment_score,
        market_sentiment.fear_greed_index,
        &market_sentiment.market_phase,
    );
    
    let volatility_factor = multi_factor_scoring::score_volatility_factor(
        atr,
        current_price,
    );
    
    let mut factors = vec![
        trend_factor,
        volume_factor,
        pattern_factor,
        momentum_factor,
        sr_factor,
        mtf_factor,
        sentiment_factor,
        volatility_factor,
    ];
    
    // 新增：智能权重调整
    println!("\n   🔧 智能权重调整中...");
    multi_factor_scoring::adjust_factor_weights(
        &mut factors,
        &market_sentiment.market_phase,
        volatility_pct,
        adx,
    );
    
    let multi_factor_score = multi_factor_scoring::calculate_multi_factor_score(factors);
    
    // 打印评分结果
    println!("   📊 综合评分: {:.1}分 ({})", 
             multi_factor_score.total_score,
             multi_factor_score.signal_quality.to_string());
    println!("   💡 操作建议: {}", multi_factor_score.operation_suggestion);
    println!("\n   各因子得分:");
    for factor in &multi_factor_score.factors {
        let status_icon = match factor.status {
            multi_factor_scoring::FactorStatus::VeryBullish => "🔥",
            multi_factor_scoring::FactorStatus::Bullish => "📈",
            multi_factor_scoring::FactorStatus::Neutral => "➡️",
            multi_factor_scoring::FactorStatus::Bearish => "📉",
            multi_factor_scoring::FactorStatus::VeryBearish => "❄️",
        };
        println!("      {} {} {:.1}分 (权重{:.0}%) - {}", 
                 status_icon,
                 factor.name,
                 factor.score,
                 factor.weight * 100.0,
                 factor.description);
    }
    
    let professional_prediction = ProfessionalPrediction {
        buy_points,
        sell_points,
        support_resistance,
        multi_timeframe,
        divergence,
        current_advice,
        risk_level,
        candle_patterns,
        volume_analysis,
        multi_factor_score,
    };
    
    println!("\n✅ 金融级策略分析完成！\n");
    
    Ok((
        PredictionResponse {
            predictions,
            last_real_data,
        },
        professional_prediction,
    ))
}

/// 生成交易建议
fn generate_trading_advice(
    buy_points: &[BuySellPoint],
    sell_points: &[BuySellPoint],
    multi_timeframe: &MultiTimeframeSignal,
    support_resistance: &SupportResistance,
    divergence: &VolumePriceDivergence,
    _current_price: f64,
) -> (String, String) {
    let mut advice_parts = Vec::new();
    
    // 基于买卖点信号计算初始risk_score
    let mut risk_score = if !buy_points.is_empty() && sell_points.is_empty() {
        let best_buy = &buy_points[0];
        advice_parts.push(format!(
            "💚 建议{}，目标价{:.2}元，止损{:.2}元",
            best_buy.point_type,
            best_buy.take_profit[0],
            best_buy.stop_loss
        ));
        4 // 买入信号，风险较低
    } else if !sell_points.is_empty() && buy_points.is_empty() {
        let best_sell = &sell_points[0];
        advice_parts.push(format!(
            "❤️ 建议{}，目标价{:.2}元，止损{:.2}元",
            best_sell.point_type,
            best_sell.take_profit[0],
            best_sell.stop_loss
        ));
        7 // 卖出信号，风险较高
    } else if !buy_points.is_empty() && !sell_points.is_empty() {
        let buy_strength = buy_points[0].signal_strength;
        let sell_strength = sell_points[0].signal_strength;
        let score = if buy_strength > sell_strength {
            advice_parts.push("💛 信号矛盾，但买入信号更强，建议谨慎买入或观望".to_string());
            5
        } else {
            advice_parts.push("💛 信号矛盾，但卖出信号更强，建议减仓或观望".to_string());
            6
        };
        score
    } else {
        advice_parts.push("💙 当前无明确买卖信号，建议观望".to_string());
        5
    };
    
    // 多周期共振建议
    if multi_timeframe.resonance_level >= 2 {
        if multi_timeframe.resonance_direction.contains("多头") {
            advice_parts.push("多周期共振向上，趋势向好".to_string());
            risk_score -= 1;
        } else if multi_timeframe.resonance_direction.contains("空头") {
            advice_parts.push("多周期共振向下，注意风险".to_string());
            risk_score += 1;
        }
    }
    
    // 支撑压力位建议
    if support_resistance.current_position.contains("接近关键支撑") {
        advice_parts.push("价格接近支撑位，可关注反弹机会".to_string());
        risk_score -= 1;
    } else if support_resistance.current_position.contains("接近关键压力") {
        advice_parts.push("价格接近压力位，注意回调风险".to_string());
        risk_score += 1;
    }
    
    // 背离预警
    if divergence.has_bullish_divergence {
        advice_parts.push("底部背离，可能即将反弹".to_string());
        risk_score -= 1;
    }
    if divergence.has_bearish_divergence {
        advice_parts.push("顶部背离，警惕回调".to_string());
        risk_score += 2;
    }
    
    let risk_level = match risk_score.clamp(1, 10) {
        1..=3 => "低风险 ✅".to_string(),
        4..=6 => "中等风险 ⚠️".to_string(),
        7..=8 => "较高风险 🔶".to_string(),
        _ => "高风险 ⛔".to_string(),
    };
    
    (advice_parts.join("；"), risk_level)
}

/// 生成价格预测
/// 生成预测理由和关键因素
fn generate_prediction_reason(
    predicted_price: f64,
    _current_price: f64,
    change_percent: f64,
    day: usize,
    support_resistance: &SupportResistance,
    multi_timeframe: &MultiTimeframeSignal,
    trend_strength: f64,
    rsi: f64,
    macd_histogram: f64,
) -> (String, Vec<String>) {
    let mut reasons = Vec::new();
    let mut key_factors = Vec::new();
    
    // 1. 分析价格位置
    let near_resistance = support_resistance.resistance_levels.iter()
        .any(|&r| (predicted_price - r).abs() / r < 0.02);
    let near_support = support_resistance.support_levels.iter()
        .any(|&s| (predicted_price - s).abs() / s < 0.02);
    
    // 2. 分析趋势强度
    let trend_desc = if trend_strength > 0.010 {
        "强势上涨趋势"
    } else if trend_strength > 0.005 {
        "温和上涨趋势"
    } else if trend_strength > -0.005 {
        "震荡整理"
    } else if trend_strength > -0.010 {
        "温和下跌趋势"
    } else {
        "强势下跌趋势"
    };
    
    // 3. RSI状态分析
    let rsi_state = if rsi > 70.0 {
        "超买区域，存在回调压力"
    } else if rsi > 60.0 {
        "偏强区域，上涨动能充足"
    } else if rsi > 40.0 {
        "中性区域，多空平衡"
    } else if rsi > 30.0 {
        "偏弱区域，下跌动能较强"
    } else {
        "超卖区域，存在反弹动力"
    };
    
    // 4. MACD状态分析
    let macd_state = if macd_histogram > 0.5 {
        "MACD红柱放大，多头强势"
    } else if macd_histogram > 0.0 {
        "MACD红柱缩小，多头减弱"
    } else if macd_histogram > -0.5 {
        "MACD绿柱缩小，空头减弱"
    } else {
        "MACD绿柱放大，空头强势"
    };
    
    // 5. 生成主要理由
    if change_percent > 0.0 {
        // 上涨预测
        if near_resistance {
            reasons.push(format!("接近压力位{:.2}元，上涨空间受限", 
                support_resistance.resistance_levels.iter()
                    .find(|&&r| (predicted_price - r).abs() / r < 0.02)
                    .unwrap_or(&predicted_price)));
            key_factors.push("⚠️ 压力位约束".to_string());
        } else if rsi > 70.0 {
            reasons.push("RSI超买，短期可能回调".to_string());
            key_factors.push("⚠️ 技术指标超买".to_string());
        } else {
            reasons.push(format!("处于{}，", trend_desc));
            reasons.push(rsi_state.to_string());
            if multi_timeframe.resonance_level >= 2 {
                reasons.push(format!("多周期{}共振", multi_timeframe.resonance_direction));
                key_factors.push(format!("✅ {}级共振", multi_timeframe.resonance_level));
            }
        }
    } else if change_percent < 0.0 {
        // 下跌预测
        if near_support {
            reasons.push(format!("接近支撑位{:.2}元，下跌空间有限", 
                support_resistance.support_levels.iter()
                    .find(|&&s| (predicted_price - s).abs() / s < 0.02)
                    .unwrap_or(&predicted_price)));
            key_factors.push("✅ 支撑位保护".to_string());
        } else if rsi < 30.0 {
            reasons.push("RSI超卖，短期可能反弹".to_string());
            key_factors.push("✅ 技术指标超卖".to_string());
        } else {
            reasons.push(format!("处于{}，", trend_desc));
            reasons.push(rsi_state.to_string());
            if multi_timeframe.resonance_level >= 2 {
                reasons.push(format!("多周期{}共振", multi_timeframe.resonance_direction));
                key_factors.push(format!("⚠️ {}级共振下跌", multi_timeframe.resonance_level));
            }
        }
    } else {
        // 横盘预测
        reasons.push("多空力量平衡，震荡整理".to_string());
        key_factors.push("📊 震荡整理".to_string());
    }
    
    // 6. 添加MACD分析
    key_factors.push(macd_state.to_string());
    
    // 7. 远期预测衰减说明
    if day > 3 {
        reasons.push(format!("第{}日预测，不确定性增加", day));
        key_factors.push(format!("⏰ T+{} 预测衰减", day));
    }
    
    let final_reason = reasons.join("；");
    (final_reason, key_factors)
}

async fn generate_price_predictions(
    request: &PredictionRequest,
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
    dates: &[String],
    current_price: f64,
    multi_timeframe: &MultiTimeframeSignal,
    support_resistance: &SupportResistance,
) -> Result<Vec<Prediction>, String> {
    let mut predictions = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(
        dates.last().unwrap_or(&"2023-01-01".to_string()),
        "%Y-%m-%d"
    ).unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    // ========== 计算多因子评分和市场情绪（影响涨跌幅的核心因子）==========
    use crate::stock_prediction::technical_indicators::{
        calculate_market_sentiment, calculate_atr, calculate_dmi_adx
    };
    use crate::stock_prediction::candlestick_patterns;
    use crate::stock_prediction::volume_analysis;
    use crate::stock_prediction::multi_factor_scoring;
    
    let n = prices.len();
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20_for_scoring = calc_ma(20);
    let ma60_for_scoring = calc_ma(60);
    
    // RSI
    let rsi_for_scoring = if n >= 14 {
        let recent_prices = &prices[n-14..];
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 { gains += change; } else { losses += -change; }
        }
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        if avg_loss == 0.0 { 100.0 } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    } else { 50.0 };
    
    // MACD
    let ema12 = calc_ma(12);
    let ema26 = calc_ma(26);
    let macd_dif_for_scoring = ema12 - ema26;
    let macd_dea_for_scoring = calc_ma(9);
    
    // 市场情绪和波动率
    let market_sentiment = calculate_market_sentiment(prices, volumes, highs, lows);
    let atr_value = calculate_atr(highs, lows, prices, 14);
    let volatility_pct_global = (atr_value / current_price) * 100.0;
    let (_di_plus, _di_minus, adx_global) = calculate_dmi_adx(highs, lows, prices, 14);
    
    // 计算KDJ指标（仅用于验证，实际在循环中动态计算）
    use crate::stock_prediction::technical_indicators::calculate_kdj;
    let (_kdj_k_global, _kdj_d_global, _kdj_j_global) = calculate_kdj(highs, lows, prices, 9);
    
    // K线形态和量价分析
    let candles: Vec<candlestick_patterns::Candle> = prices.iter().zip(highs.iter()).zip(lows.iter()).zip(volumes.iter())
        .map(|(((& close, &high), &low), &volume)| {
            candlestick_patterns::Candle {
                open: close,
                high,
                low,
                close,
                volume,
            }
        }).collect();
    let candle_patterns = candlestick_patterns::identify_all_patterns(&candles);
    let volume_analysis_result = volume_analysis::analyze_volume_price_enhanced(prices, volumes, highs, lows);
    
    // 计算各因子得分
    let trend_factor = multi_factor_scoring::score_trend_factor(ma5, ma10, ma20_for_scoring, ma60_for_scoring, current_price);
    let obv_last = volume_analysis_result.obv.last().copied().unwrap_or(0.0);
    let obv_trend_str = if obv_last > 0.0 { "上升".to_string() } else { "下降".to_string() };
    let volume_factor = multi_factor_scoring::score_volume_factor(
        &volume_analysis_result.volume_trend,
        volume_analysis_result.volume_price_sync,
        volume_analysis_result.accumulation_signal,
        &obv_trend_str,
    );
    let pattern_factor = multi_factor_scoring::score_pattern_factor(&candle_patterns);
    let momentum_factor = multi_factor_scoring::score_momentum_factor(rsi_for_scoring, macd_dif_for_scoring, macd_dea_for_scoring);
    let sr_factor = multi_factor_scoring::score_support_resistance_factor(
        current_price,
        &support_resistance.support_levels,
        &support_resistance.resistance_levels,
    );
    let mtf_factor = multi_factor_scoring::score_multi_timeframe_factor(
        multi_timeframe.resonance_level,
        &multi_timeframe.resonance_direction,
        multi_timeframe.signal_quality,
    );
    let sentiment_factor = multi_factor_scoring::score_sentiment_factor(
        market_sentiment.sentiment_score,
        market_sentiment.fear_greed_index,
        &market_sentiment.market_phase,
    );
    let volatility_factor = multi_factor_scoring::score_volatility_factor(atr_value, current_price);
    
    let mut factors = vec![
        trend_factor, volume_factor, pattern_factor, momentum_factor,
        sr_factor, mtf_factor, sentiment_factor, volatility_factor,
    ];
    
    // 智能权重调整
    multi_factor_scoring::adjust_factor_weights(&mut factors, &market_sentiment.market_phase, volatility_pct_global, adx_global);
    let multi_factor_score = multi_factor_scoring::calculate_multi_factor_score(factors);
    
    println!("   📊 多因子评分: {:.1}/100 (影响涨跌幅)", multi_factor_score.total_score);
    // ========== 多因子计算完成 ==========
    
    // 基于共振方向确定趋势偏向
    let trend_bias: f64 = match multi_timeframe.resonance_level {
        3 => {
            if multi_timeframe.resonance_direction.contains("多头") { 0.015 }
            else if multi_timeframe.resonance_direction.contains("空头") { -0.015 }
            else { 0.0 }
        },
        2 => {
            if multi_timeframe.resonance_direction.contains("多头") { 0.010 }
            else if multi_timeframe.resonance_direction.contains("空头") { -0.010 }
            else { 0.0 }
        },
        1 => {
            if multi_timeframe.resonance_direction.contains("多") { 0.005 }
            else if multi_timeframe.resonance_direction.contains("空") { -0.005 }
            else { 0.0 }
        },
        _ => 0.0,
    };
    
    // 计算历史波动率（金融级别：必须基于实际市场波动）
    let volatility = calculate_historical_volatility(prices).clamp(0.015, 0.08);
    
    // 计算MA均线（用于均值回归）
    let ma20 = if prices.len() >= 20 {
        prices[prices.len()-20..].iter().sum::<f64>() / 20.0
    } else {
        current_price
    };
    
    let ma60 = if prices.len() >= 60 {
        prices[prices.len()-60..].iter().sum::<f64>() / 60.0
    } else {
        current_price
    };
    
    // 计算价格动量（最近5日相对前5日的变化）
    let momentum = if prices.len() >= 10 {
        let recent_avg = prices[prices.len()-5..].iter().sum::<f64>() / 5.0;
        let previous_avg = prices[prices.len()-10..prices.len()-5].iter().sum::<f64>() / 5.0;
        (recent_avg - previous_avg) / previous_avg
    } else {
        0.0
    };
    
    // 趋势强度（结合动量和共振，但限制最大强度）
    let initial_trend_strength = if trend_bias.abs() > 0.001 {
        trend_bias.clamp(-0.020, 0.020)  // 限制最大趋势强度±2%
    } else {
        // 无明显共振时，使用动量作为趋势判断
        (momentum * 0.5).clamp(-0.015, 0.015)
    };
    
    // 用于累积预测的向量（动态更新RSI/MACD/KDJ）
    let mut predicted_prices_for_calc = prices.to_vec();
    let mut predicted_highs_for_calc = highs.to_vec();
    let mut predicted_lows_for_calc = lows.to_vec();
    
    // 存储前一天的指标值，用于判断金叉死叉
    let mut prev_kdj_k: Option<f64> = None;
    let mut prev_kdj_d: Option<f64> = None;
    let mut prev_macd_dif: Option<f64> = None;
    let mut prev_macd_dea: Option<f64> = None;
    
    for day in 1..=request.prediction_days {
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 计算当前的RSI、MACD和KDJ（基于累积预测价格）
        let current_rsi = calculate_rsi(&predicted_prices_for_calc);
        let (_, _, macd_histogram) = calculate_macd_full(&predicted_prices_for_calc);
        let (current_kdj_k, current_kdj_d, current_kdj_j) = calculate_kdj(
            &predicted_highs_for_calc, 
            &predicted_lows_for_calc, 
            &predicted_prices_for_calc, 
            9
        );
        
        // 趋势衰减（金融逻辑：预测越远衰减越快）：使用配置常量
        // 更激进的衰减：第1天100%，第2天81%，第3天66%，第5天44%
        let trend_decay = SIMPLE_TREND_DECAY_BASE.powi(day as i32);
        
        // 均值回归力量（价格偏离均线越远，回归力量越强）：使用配置常量
        let price_deviation_from_ma20 = (last_price - ma20) / ma20;
        let price_deviation_from_ma60 = (last_price - ma60) / ma60;
        
        // 回归系数：偏离±5%以上时启动回归：使用配置常量
        let mean_reversion_force = {
            let deviation = price_deviation_from_ma20 * MA20_REVERSION_WEIGHT + price_deviation_from_ma60 * MA60_REVERSION_WEIGHT;
            if deviation.abs() > STRONG_MEAN_REVERSION_THRESHOLD {
                // 偏离>5%，强制回归
                -deviation * STRONG_MEAN_REVERSION_FORCE
            } else if deviation.abs() > MODERATE_MEAN_REVERSION_THRESHOLD {
                // 偏离>3%，温和回归
                -deviation * MODERATE_MEAN_REVERSION_FORCE
            } else {
                0.0  // 偏离<3%，不回归
            }
        };
        
        // 动态调整趋势强度（金融逻辑：根据技术指标和价位调整）
        let mut current_trend_strength = initial_trend_strength;
        
        // 检查是否接近压力位（上涨时）：使用配置常量
        if current_trend_strength > 0.0 {
            for &resistance in &support_resistance.resistance_levels {
                if (last_price - resistance).abs() / resistance < NEAR_RESISTANCE_DISTANCE {
                    // 接近压力位，减弱上涨趋势
                    current_trend_strength *= NEAR_RESISTANCE_WEAKEN;
                    break;
                }
            }
            // RSI超买，减弱上涨趋势：使用配置常量
            if current_rsi > 70.0 {
                current_trend_strength *= RSI_STRONG_OVERBOUGHT_WEAKEN;
            } else if current_rsi > RSI_MODERATE_OVERBOUGHT {
                current_trend_strength *= RSI_MODERATE_OVERBOUGHT_WEAKEN;
            }
        }
        
        // 检查是否接近支撑位（下跌时）：使用配置常量
        if current_trend_strength < 0.0 {
            for &support in &support_resistance.support_levels {
                if (last_price - support).abs() / support < NEAR_SUPPORT_DISTANCE {
                    // 接近支撑位，减弱下跌趋势
                    current_trend_strength *= NEAR_SUPPORT_WEAKEN;
                    break;
                }
            }
            // RSI超卖，减弱下跌趋势：使用配置常量
            if current_rsi < 30.0 {
                current_trend_strength *= RSI_STRONG_OVERSOLD_WEAKEN;
            } else if current_rsi < RSI_MODERATE_OVERSOLD {
                current_trend_strength *= RSI_MODERATE_OVERSOLD_WEAKEN;
            }
        }
        
        // 确定性波动调整（金融逻辑：双向波动，不放大趋势）：使用配置常量
        let base_volatility = volatility * SIMPLE_PREDICTION_BASE_VOLATILITY;
        
        // 双向波动因子：基于日期的确定性正负波动：使用配置常量
        let oscillation_factor = ((day as f64 * 1.618).sin() * VOLATILITY_NORMALIZATION + 
                                   (day as f64 * GOLDEN_RATIO_OSCILLATION).cos() * 0.3) * trend_decay;
        
        let volatility_adjustment = if current_trend_strength.abs() < 0.001 {
            // 震荡市：纯双向波动：使用配置常量
            base_volatility * oscillation_factor * OSCILLATION_MARKET_VOLATILITY
        } else {
            // 趋势市：小幅双向波动（不放大趋势，只增加波动性）：使用配置常量
            base_volatility * oscillation_factor * TREND_MARKET_VOLATILITY
        };
        
        // 综合变化率（金融逻辑：趋势 + 波动 + 均值回归）：使用配置常量
        // 关键修复：趋势衰减更快，加入均值回归
        let mut change_rate = current_trend_strength * trend_decay * TREND_COMPONENT_WEIGHT
                        + volatility_adjustment
                        + mean_reversion_force;
        
        // ========== 新增：多因子评分调整（第一层增强）==========
        let score = multi_factor_score.total_score;
        if score > 75.0 {
            // 强烈看涨 (评分>75)
            change_rate += (score - 75.0) * MULTI_FACTOR_STRONG_BULLISH_IMPACT;
        } else if score >= 60.0 {
            // 看涨 (评分60-75)
            change_rate += (score - 60.0) * MULTI_FACTOR_BULLISH_IMPACT;
        } else if score >= 40.0 {
            // 中性 (评分40-60)
            change_rate += MULTI_FACTOR_NEUTRAL_BIAS;
        } else if score >= 25.0 {
            // 看跌 (评分25-40)
            change_rate -= (40.0 - score) * MULTI_FACTOR_BEARISH_IMPACT;
        } else {
            // 强烈看跌 (评分<25)
            change_rate -= 0.005 + (25.0 - score) * MULTI_FACTOR_STRONG_BEARISH_IMPACT;
        }
        
        // ========== 新增：波动率调节（第二层调节）==========
        if volatility_pct_global > 8.0 {
            // 极高波动率
            change_rate *= EXTREME_VOLATILITY_SUPPRESS;
        } else if volatility_pct_global > 5.0 {
            // 高波动率
            change_rate *= HIGH_VOLATILITY_SUPPRESS;
        } else if volatility_pct_global < 1.5 {
            // 低波动率
            change_rate *= LOW_VOLATILITY_ENHANCE;
        }
        
        // ========== 新增：市场情绪调节（第三层增强）==========
        if market_sentiment.fear_greed_index < 20.0 && change_rate > 0.0 {
            // 极度恐慌时的反向买入信号增强
            change_rate *= EXTREME_FEAR_CONTRARIAN_BOOST;
            if market_sentiment.market_phase.contains("恐慌") {
                change_rate += PANIC_REVERSAL_BONUS;
            }
        } else if market_sentiment.fear_greed_index > 80.0 && change_rate > 0.0 {
            // 极度贪婪时的上涨预测抑制
            change_rate *= EXTREME_GREED_SUPPRESS;
        }
        
        if market_sentiment.market_phase.contains("过热") && change_rate < 0.0 {
            // 过热期的回调机会加成
            change_rate -= OVERHEATED_CORRECTION_BONUS;
        }
        // ========== 多因子调整完成 ==========
        
        // 确保变化率有最小值（金融逻辑：股价不会完全不动）
        let adjusted_change_rate = if change_rate.abs() < 0.001 {
            // 最小波动：±0.3%
            if day % 3 == 0 { 0.003 } 
            else if day % 3 == 1 { -0.002 }
            else { 0.001 }
        } else {
            change_rate
        };
        
        let change_percent = clamp_daily_change(adjusted_change_rate * 100.0);
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        // 置信度随时间递减：使用配置常量
        let confidence = (PREDICTION_BASE_CONFIDENCE * trend_decay + multi_timeframe.signal_quality * SIGNAL_QUALITY_CONFIDENCE_IMPACT).clamp(PREDICTION_MIN_CONFIDENCE, PREDICTION_MAX_CONFIDENCE);
        
        // 交易信号
        let trading_signal = if current_trend_strength > 0.008 {
            "买入"
        } else if current_trend_strength < -0.008 {
            "卖出"
        } else {
            "持有"
        }.to_string();
        
        // 生成预测理由和关键因素
        let (prediction_reason, key_factors) = generate_prediction_reason(
            predicted_price,
            current_price,
            change_percent,
            day,
            support_resistance,
            multi_timeframe,
            current_trend_strength,
            current_rsi,
            macd_histogram,
        );
        
        // 构建技术指标数据
        use crate::stock_prediction::types::TechnicalIndicatorValues;
        
        // 判断KDJ金叉死叉（需要比较当前和前一个值的穿越）
        let kdj_golden_cross = if let (Some(prev_k), Some(prev_d)) = (prev_kdj_k, prev_kdj_d) {
            // 金叉：前一时刻K<D，当前时刻K>D（从下往上穿）
            prev_k <= prev_d && current_kdj_k > current_kdj_d
        } else {
            false
        };
        
        let kdj_death_cross = if let (Some(prev_k), Some(prev_d)) = (prev_kdj_k, prev_kdj_d) {
            // 死叉：前一时刻K>D，当前时刻K<D（从上往下穿）
            prev_k >= prev_d && current_kdj_k < current_kdj_d
        } else {
            false
        };
        
        // 判断MACD金叉死叉（需要比较当前和前一个值的穿越）
        let macd_golden_cross = if let (Some(prev_dif), Some(prev_dea)) = (prev_macd_dif, prev_macd_dea) {
            // 金叉：前一时刻DIF<DEA，当前时刻DIF>DEA（从下往上穿）
            prev_dif <= prev_dea && macd_dif_for_scoring > macd_dea_for_scoring
        } else {
            false
        };
        
        let macd_death_cross = if let (Some(prev_dif), Some(prev_dea)) = (prev_macd_dif, prev_macd_dea) {
            // 死叉：前一时刻DIF>DEA，当前时刻DIF<DEA（从上往下穿）
            prev_dif >= prev_dea && macd_dif_for_scoring < macd_dea_for_scoring
        } else {
            false
        };
        
        let tech_indicators = TechnicalIndicatorValues {
            rsi: current_rsi,
            macd_histogram,
            macd_dif: macd_dif_for_scoring,
            macd_dea: macd_dea_for_scoring,
            kdj_j: current_kdj_j,
            kdj_k: current_kdj_k,
            kdj_d: current_kdj_d,
            cci: 0.0, // CCI指标后续可补充
            obv_trend: obv_last,
            macd_golden_cross,
            macd_death_cross,
            kdj_golden_cross,
            kdj_death_cross,
            kdj_overbought: current_kdj_k > 80.0 && current_kdj_d > 80.0,
            kdj_oversold: current_kdj_k < 20.0 && current_kdj_d < 20.0,
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(trading_signal),
            signal_strength: Some(multi_timeframe.signal_quality / 100.0),
            technical_indicators: Some(tech_indicators),
            prediction_reason: Some(prediction_reason),
            key_factors: Some(key_factors),
        });
        
        // 更新价格向量用于下一轮RSI/MACD/KDJ计算
        predicted_prices_for_calc.push(predicted_price);
        
        // 估算预测的high和low（基于波动率）
        let predicted_high = predicted_price * (1.0 + volatility * 0.5);
        let predicted_low = predicted_price * (1.0 - volatility * 0.5);
        predicted_highs_for_calc.push(predicted_high);
        predicted_lows_for_calc.push(predicted_low);
        
        // 更新前一天的指标值（用于下次判断金叉死叉）
        prev_kdj_k = Some(current_kdj_k);
        prev_kdj_d = Some(current_kdj_d);
        prev_macd_dif = Some(macd_dif_for_scoring);
        prev_macd_dea = Some(macd_dea_for_scoring);
        
        last_price = predicted_price;
    }
    
    Ok(predictions)
} 

/// 纯技术分析预测（不依赖模型）
/// 仅基于历史数据和技术指标进行金融级预测
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TechnicalOnlyRequest {
    pub stock_code: String,
    pub history_days: usize,      // 使用多少天历史数据（推荐120-250）
    pub prediction_days: usize,   // 预测未来多少天
}

/// 纯技术分析预测 - 不需要模型训练
pub async fn predict_with_technical_analysis_only(
    request: TechnicalOnlyRequest
) -> std::result::Result<(PredictionResponse, ProfessionalPrediction), String> {
    println!("\n🎯 ========== 纯技术分析预测（无模型）==========");
    println!("📊 使用 {} 天历史数据", request.history_days);
    println!("🔮 预测未来 {} 天", request.prediction_days);
    
    // 获取历史数据
    let (current_price, current_change_percent, dates, prices, volumes, highs, lows) = 
        get_recent_market_data(&request.stock_code, request.history_days).await
        .map_err(|e| format!("获取市场数据失败: {e}"))?;
    
    if prices.len() < 60 {
        return Err(format!("历史数据不足，需要至少60天数据，当前只有{}天", prices.len()));
    }
    
    println!("✅ 数据加载完成: {}天历史数据", prices.len());
    println!("💰 当前价格: {:.2}元 ({:+.2}%)\n", current_price, current_change_percent);
    
    // 1. 计算支撑压力位
    let support_resistance = calculate_support_resistance_levels(
        &prices, &highs, &lows, current_price
    );
    
    println!("📍 ========== 支撑压力位分析 ==========");
    println!("   当前位置: {}", support_resistance.current_position);
    if !support_resistance.support_levels.is_empty() {
        println!("   🟢 关键支撑位:");
        for (i, &level) in support_resistance.support_levels.iter().enumerate() {
            let distance = (current_price - level) / current_price * 100.0;
            println!("      {}. {:.2}元 (距离-{:.2}%)", i+1, level, distance);
        }
    }
    if !support_resistance.resistance_levels.is_empty() {
        println!("   🔴 关键压力位:");
        for (i, &level) in support_resistance.resistance_levels.iter().enumerate() {
            let distance = (level - current_price) / current_price * 100.0;
            println!("      {}. {:.2}元 (距离+{:.2}%)", i+1, level, distance);
        }
    }
    
    // 2. 构建多周期数据
    let mut daily_data: Vec<StockData> = Vec::with_capacity(prices.len());
    for (i, date) in dates.iter().enumerate() {
        daily_data.push(StockData {
            symbol: request.stock_code.clone(),
            date: date.clone(),
            open: prices[i],
            high: highs.get(i).copied().unwrap_or(prices[i]),
            low: lows.get(i).copied().unwrap_or(prices[i]),
            close: prices[i],
            volume: volumes.get(i).copied().unwrap_or(0) as f64,
        });
    }
    
    // 3. 多周期共振分析
    let multi_timeframe = analyze_multi_timeframe_resonance(&daily_data);
    
    println!("\n🔄 ========== 多周期共振分析 ==========");
    println!("   📈 日线趋势: {}", multi_timeframe.daily_trend);
    println!("   📊 周线趋势: {}", multi_timeframe.weekly_trend);
    println!("   📉 月线趋势: {}", multi_timeframe.monthly_trend);
    println!("   ⚡ 共振级别: {} ({})", 
             multi_timeframe.resonance_level,
             multi_timeframe.resonance_direction);
    println!("   ✨ 信号质量: {:.0}分", multi_timeframe.signal_quality);
    
    // 4. 量价背离分析
    let divergence = analyze_volume_price_divergence(&prices, &volumes, &highs, &lows);
    
    println!("\n⚠️  ========== 量价背离分析 ==========");
    if divergence.has_bullish_divergence {
        println!("   🟢 检测到底背离 (强度: {:.0}%)", divergence.divergence_strength * 100.0);
    }
    if divergence.has_bearish_divergence {
        println!("   🔴 检测到顶背离 (强度: {:.0}%)", divergence.divergence_strength * 100.0);
    }
    println!("   💡 {}", divergence.warning_message);
    
    // 5. 识别买卖点
    let buy_points = identify_buy_points(
        &prices, &volumes, &highs, &lows, 
        current_price, &support_resistance, 
        &multi_timeframe, &divergence
    );
    
    let sell_points = identify_sell_points(
        &prices, &volumes, &highs, &lows,
        current_price, &support_resistance,
        &multi_timeframe, &divergence
    );
    
    println!("\n💎 ========== 买卖点信号 ==========");
    if !buy_points.is_empty() {
        println!("   🟢 买入信号 ({} 个)", buy_points.len());
    } else {
        println!("   🟡 暂无明确买入信号");
    }
    
    if !sell_points.is_empty() {
        println!("   🔴 卖出信号 ({} 个)", sell_points.len());
    } else {
        println!("   🟡 暂无明确卖出信号");
    }
    
    // 6. K线形态识别
    let candles: Vec<candlestick_patterns::Candle> = daily_data.iter().map(|d| {
        candlestick_patterns::Candle {
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
            volume: d.volume as i64,
        }
    }).collect();
    
    let candle_patterns = candlestick_patterns::identify_all_patterns(&candles);
    
    println!("\n📊 ========== K线形态识别 ==========");
    if !candle_patterns.is_empty() {
        println!("   识别到 {} 个形态", candle_patterns.len());
    } else {
        println!("   未检测到明显的K线形态信号");
    }
    
    // 7. 量价关系深度分析
    let volume_analysis_raw = volume_analysis::analyze_volume_price_enhanced(&prices, &volumes, &highs, &lows);
    
    println!("\n📈 ========== 量价关系分析 ==========");
    println!("   量能趋势: {}", volume_analysis_raw.volume_trend);
    println!("   量价配合: {}", if volume_analysis_raw.volume_price_sync { "✅ 良好" } else { "⚠️ 背离" });
    println!("   吸筹信号: {:.0}分", volume_analysis_raw.accumulation_signal);
    
    let obv_trend = if volume_analysis_raw.obv.len() >= 10 {
        let recent_obv = &volume_analysis_raw.obv[volume_analysis_raw.obv.len()-10..];
        if recent_obv.last().unwrap() > recent_obv.first().unwrap() {
            "上升趋势".to_string()
        } else {
            "下降趋势".to_string()
        }
    } else {
        "数据不足".to_string()
    };
    
    let volume_analysis = VolumeAnalysisInfo {
        volume_trend: volume_analysis_raw.volume_trend.clone(),
        volume_price_sync: volume_analysis_raw.volume_price_sync,
        accumulation_signal: volume_analysis_raw.accumulation_signal,
        obv_trend,
    };
    
    // 8. 生成操作建议
    let (current_advice, risk_level) = generate_trading_advice(
        &buy_points,
        &sell_points,
        &multi_timeframe,
        &support_resistance,
        &divergence,
        current_price,
    );
    
    println!("\n📋 ========== 操作建议 ==========");
    println!("   {}", current_advice);
    println!("   风险等级: {}", risk_level);
    
    // 9. 多因子综合评分
    println!("\n🎯 ========== 多因子综合评分 ==========");
    
    use crate::stock_prediction::technical_indicators::{
        calculate_market_sentiment, calculate_atr, calculate_dmi_adx
    };
    
    let n = prices.len();
    let calc_ma = |window: usize| -> f64 {
        if n >= window {
            prices[n-window..].iter().sum::<f64>() / window as f64
        } else {
            current_price
        }
    };
    
    let ma5 = calc_ma(5);
    let ma10 = calc_ma(10);
    let ma20 = calc_ma(20);
    let ma60 = calc_ma(60);
    
    let rsi = if n >= 14 {
        let recent_prices = &prices[n-14..];
        let mut gains = 0.0;
        let mut losses = 0.0;
        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 { gains += change; } else { losses += -change; }
        }
        let avg_gain = gains / 14.0;
        let avg_loss = losses / 14.0;
        if avg_loss == 0.0 { 100.0 } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        }
    } else { 50.0 };
    
    let ema12 = calc_ma(12);
    let ema26 = calc_ma(26);
    let macd_dif = ema12 - ema26;
    let macd_dea = calc_ma(9);
    
    let market_sentiment = calculate_market_sentiment(&prices, &volumes, &highs, &lows);
    let atr = calculate_atr(&highs, &lows, &prices, 14);
    let volatility_pct = (atr / current_price) * 100.0;
    let (_di_plus, _di_minus, adx) = calculate_dmi_adx(&highs, &lows, &prices, 14);
    
    println!("   📊 市场情绪: {} (恐惧贪婪指数: {:.0})", 
             market_sentiment.sentiment_level,
             market_sentiment.fear_greed_index);
    println!("   📈 ATR波动率: {:.2}%", volatility_pct);
    println!("   💪 ADX趋势强度: {:.1}", adx);
    
    let trend_factor = multi_factor_scoring::score_trend_factor(ma5, ma10, ma20, ma60, current_price);
    
    let obv_last = volume_analysis_raw.obv.last().copied().unwrap_or(0.0);
    let obv_trend_str = if obv_last > 0.0 { "上升".to_string() } else { "下降".to_string() };
    let volume_factor = multi_factor_scoring::score_volume_factor(
        &volume_analysis_raw.volume_trend,
        volume_analysis_raw.volume_price_sync,
        volume_analysis_raw.accumulation_signal,
        &obv_trend_str,
    );
    
    let pattern_factor = multi_factor_scoring::score_pattern_factor(&candle_patterns);
    let momentum_factor = multi_factor_scoring::score_momentum_factor(rsi, macd_dif, macd_dea);
    let sr_factor = multi_factor_scoring::score_support_resistance_factor(
        current_price,
        &support_resistance.support_levels,
        &support_resistance.resistance_levels,
    );
    let mtf_factor = multi_factor_scoring::score_multi_timeframe_factor(
        multi_timeframe.resonance_level,
        &multi_timeframe.resonance_direction,
        multi_timeframe.signal_quality,
    );
    let sentiment_factor = multi_factor_scoring::score_sentiment_factor(
        market_sentiment.sentiment_score,
        market_sentiment.fear_greed_index,
        &market_sentiment.market_phase,
    );
    let volatility_factor = multi_factor_scoring::score_volatility_factor(atr, current_price);
    
    let mut factors = vec![
        trend_factor, volume_factor, pattern_factor, momentum_factor,
        sr_factor, mtf_factor, sentiment_factor, volatility_factor,
    ];
    
    multi_factor_scoring::adjust_factor_weights(&mut factors, &market_sentiment.market_phase, volatility_pct, adx);
    let multi_factor_score = multi_factor_scoring::calculate_multi_factor_score(factors);
    
    println!("   ⭐ 综合评分: {:.1}/100", multi_factor_score.total_score);
    println!("   📝 操作建议: {}", multi_factor_score.operation_suggestion);
    
    // 10. 生成未来价格预测（纯技术分析）
    let predictions = generate_price_predictions_technical_only(
        &prices,
        &highs,
        &lows,
        &volumes,
        &dates,
        current_price,
        &multi_timeframe,
        &support_resistance,
        &market_sentiment,
        &multi_factor_score,
        request.prediction_days,
        volatility_pct,
        adx,
    ).await?;
    
    let last_real_data = Some(LastRealData {
        date: dates.last().unwrap().clone(),
        price: current_price,
        change_percent: current_change_percent,
    });
    
    let professional_prediction = ProfessionalPrediction {
        buy_points,
        sell_points,
        support_resistance,
        multi_timeframe,
        divergence,
        current_advice,
        risk_level,
        candle_patterns,
        volume_analysis,
        multi_factor_score,
    };
    
    println!("\n✅ 纯技术分析预测完成！\n");
    
    Ok((
        PredictionResponse {
            predictions,
            last_real_data,
        },
        professional_prediction,
    ))
}

/// 纯技术分析的价格预测生成（不依赖模型）
async fn generate_price_predictions_technical_only(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    _volumes: &[i64],
    dates: &[String],
    current_price: f64,
    multi_timeframe: &MultiTimeframeSignal,
    _support_resistance: &SupportResistance,
    market_sentiment: &crate::stock_prediction::technical_indicators::MarketSentiment,
    multi_factor_score: &multi_factor_scoring::MultiFactorScore,
    prediction_days: usize,
    volatility_pct: f64,
    _adx: f64,
) -> Result<Vec<Prediction>, String> {
    use crate::stock_prediction::technical_indicators::calculate_kdj;
    
    let mut predictions = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(
        dates.last().unwrap_or(&"2023-01-01".to_string()),
        "%Y-%m-%d"
    ).unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    // 计算基础技术指标
    let n = prices.len();
    let ma20 = if n >= 20 {
        prices[n-20..].iter().sum::<f64>() / 20.0
    } else {
        current_price
    };
    
    let _ma60 = if n >= 60 {
        prices[n-60..].iter().sum::<f64>() / 60.0
    } else {
        current_price
    };
    
    let volatility = calculate_historical_volatility(prices).clamp(0.015, 0.08);
    
    // 基于多因子评分和趋势确定方向
    let score = multi_factor_score.total_score;
    let base_direction = if score > 60.0 {
        0.01  // 看涨
    } else if score < 40.0 {
        -0.01  // 看跌
    } else {
        0.0  // 中性
    };
    
    // 共振趋势加成
    let trend_bias = match multi_timeframe.resonance_level {
        3 => {
            if multi_timeframe.resonance_direction.contains("多头") { 0.015 }
            else if multi_timeframe.resonance_direction.contains("空头") { -0.015 }
            else { 0.0 }
        },
        2 => {
            if multi_timeframe.resonance_direction.contains("多头") { 0.010 }
            else if multi_timeframe.resonance_direction.contains("空头") { -0.010 }
            else { 0.0 }
        },
        1 => {
            if multi_timeframe.resonance_direction.contains("多") { 0.005 }
            else if multi_timeframe.resonance_direction.contains("空") { -0.005 }
            else { 0.0 }
        },
        _ => 0.0,
    };
    
    let mut predicted_prices_for_calc = prices.to_vec();
    let mut predicted_highs_for_calc = highs.to_vec();
    let mut predicted_lows_for_calc = lows.to_vec();
    
    let mut prev_kdj_k: Option<f64> = None;
    let mut prev_kdj_d: Option<f64> = None;
    
    let calc_ma_local = |window: usize, data: &[f64]| -> f64 {
        let len = data.len();
        if len >= window {
            data[len-window..].iter().sum::<f64>() / window as f64
        } else if !data.is_empty() {
            data[len-1]
        } else {
            current_price
        }
    };
    
    let macd_dif_base = calc_ma_local(12, prices) - calc_ma_local(26, prices);
    let macd_dea_base = calc_ma_local(9, prices);
    let mut prev_macd_dif = Some(macd_dif_base);
    let mut prev_macd_dea = Some(macd_dea_base);
    
    for day in 1..=prediction_days {
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 计算动态指标
        let current_rsi = calculate_rsi(&predicted_prices_for_calc);
        let (_, _, macd_histogram) = calculate_macd_full(&predicted_prices_for_calc);
        let (current_kdj_k, current_kdj_d, current_kdj_j) = calculate_kdj(
            &predicted_highs_for_calc,
            &predicted_lows_for_calc,
            &predicted_prices_for_calc,
            9
        );
        
        let macd_dif_current = calc_ma_local(12, &predicted_prices_for_calc) - calc_ma_local(26, &predicted_prices_for_calc);
        let macd_dea_current = calc_ma_local(9, &predicted_prices_for_calc);
        
        // 趋势衰减
        let trend_decay = 0.9_f64.powi(day as i32);
        
        // 综合趋势强度
        let mut trend_strength = (base_direction + trend_bias) * trend_decay;
        
        // 多因子评分影响
        if score > 75.0 {
            trend_strength += (score - 75.0) * 0.0002;
        } else if score < 25.0 {
            trend_strength -= (25.0 - score) * 0.0002;
        }
        
        // 市场情绪调节
        if market_sentiment.fear_greed_index < 20.0 && trend_strength > 0.0 {
            trend_strength *= 1.2;
        } else if market_sentiment.fear_greed_index > 80.0 && trend_strength > 0.0 {
            trend_strength *= 0.8;
        }
        
        // 波动率调节
        if volatility_pct > 8.0 {
            trend_strength *= 0.7;
        } else if volatility_pct < 1.5 {
            trend_strength *= 1.1;
        }
        
        // 均值回归
        let price_deviation_from_ma20 = (last_price - ma20) / ma20;
        let mean_reversion = if price_deviation_from_ma20.abs() > 0.05 {
            -price_deviation_from_ma20 * 0.1
        } else {
            0.0
        };
        
        // 随机波动
        let oscillation = ((day as f64 * 1.618).sin() * 0.3 + (day as f64 * 0.618).cos() * 0.2) * volatility * trend_decay;
        
        let change_rate = trend_strength + mean_reversion + oscillation;
        let change_percent = clamp_daily_change(change_rate * 100.0);
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        let confidence = (0.75 * trend_decay + multi_timeframe.signal_quality * 0.003).clamp(0.4, 0.85);
        
        let trading_signal = if trend_strength > 0.008 {
            "买入"
        } else if trend_strength < -0.008 {
            "卖出"
        } else {
            "持有"
        }.to_string();
        
        // 判断金叉死叉
        let kdj_golden_cross = if let (Some(prev_k), Some(prev_d)) = (prev_kdj_k, prev_kdj_d) {
            prev_k <= prev_d && current_kdj_k > current_kdj_d
        } else {
            false
        };
        
        let kdj_death_cross = if let (Some(prev_k), Some(prev_d)) = (prev_kdj_k, prev_kdj_d) {
            prev_k >= prev_d && current_kdj_k < current_kdj_d
        } else {
            false
        };
        
        let macd_golden_cross = if let (Some(prev_dif), Some(prev_dea)) = (prev_macd_dif, prev_macd_dea) {
            prev_dif <= prev_dea && macd_dif_current > macd_dea_current
        } else {
            false
        };
        
        let macd_death_cross = if let (Some(prev_dif), Some(prev_dea)) = (prev_macd_dif, prev_macd_dea) {
            prev_dif >= prev_dea && macd_dif_current < macd_dea_current
        } else {
            false
        };
        
        let obv_last = 0.0;
        let tech_indicators = TechnicalIndicatorValues {
            rsi: current_rsi,
            macd_histogram,
            macd_dif: macd_dif_current,
            macd_dea: macd_dea_current,
            kdj_j: current_kdj_j,
            kdj_k: current_kdj_k,
            kdj_d: current_kdj_d,
            cci: 0.0,
            obv_trend: obv_last,
            macd_golden_cross,
            macd_death_cross,
            kdj_golden_cross,
            kdj_death_cross,
            kdj_overbought: current_kdj_k > 80.0 && current_kdj_d > 80.0,
            kdj_oversold: current_kdj_k < 20.0 && current_kdj_d < 20.0,
        };
        
        let prediction_reason = format!(
            "基于{}天历史数据的纯技术分析。多因子评分{:.0}分，趋势方向{}，RSI:{:.1}",
            prices.len(),
            score,
            if trend_strength > 0.0 { "看涨" } else if trend_strength < 0.0 { "看跌" } else { "震荡" },
            current_rsi
        );
        
        let mut key_factors = vec![];
        if multi_timeframe.resonance_level >= 2 {
            key_factors.push(format!("{}周期共振", multi_timeframe.resonance_level));
        }
        if volatility_pct > 5.0 {
            key_factors.push("高波动".to_string());
        }
        if _adx > 40.0 {
            key_factors.push("强趋势".to_string());
        }
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(trading_signal),
            signal_strength: Some(multi_timeframe.signal_quality / 100.0),
            technical_indicators: Some(tech_indicators),
            prediction_reason: Some(prediction_reason),
            key_factors: Some(key_factors),
        });
        
        predicted_prices_for_calc.push(predicted_price);
        let predicted_high = predicted_price * (1.0 + volatility * 0.5);
        let predicted_low = predicted_price * (1.0 - volatility * 0.5);
        predicted_highs_for_calc.push(predicted_high);
        predicted_lows_for_calc.push(predicted_low);
        
        prev_kdj_k = Some(current_kdj_k);
        prev_kdj_d = Some(current_kdj_d);
        prev_macd_dif = Some(macd_dif_current);
        prev_macd_dea = Some(macd_dea_current);
        
        last_price = predicted_price;
    }
    
    Ok(predictions)
} 