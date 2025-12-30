//! 专业级背离检测模块
//! 
//! 背离是华尔街量化交易中最重要的反转信号之一
//! 
//! 背离类型：
//! - 常规背离：预示趋势反转
//! - 隐藏背离：预示趋势延续
//! - 三重背离：连续三次背离，极强信号
//! 
//! 检测指标：
//! - RSI背离
//! - MACD背离
//! - OBV背离（量价背离）
//! - Williams %R背离
//! - ROC背离

use serde::{Deserialize, Serialize};
use crate::prediction::indicators::{rsi, macd, williams, roc};

/// 背离类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceType {
    /// 常规看涨背离：价格创新低，指标未创新低
    RegularBullish,
    /// 常规看跌背离：价格创新高，指标未创新高
    RegularBearish,
    /// 隐藏看涨背离：价格未创新低，指标创新低（趋势延续）
    HiddenBullish,
    /// 隐藏看跌背离：价格未创新高，指标创新高（趋势延续）
    HiddenBearish,
}

impl DivergenceType {
    pub fn to_string(&self) -> String {
        match self {
            Self::RegularBullish => "常规底背离".to_string(),
            Self::RegularBearish => "常规顶背离".to_string(),
            Self::HiddenBullish => "隐藏底背离".to_string(),
            Self::HiddenBearish => "隐藏顶背离".to_string(),
        }
    }
    
    pub fn is_bullish(&self) -> bool {
        matches!(self, Self::RegularBullish | Self::HiddenBullish)
    }
    
    pub fn is_reversal_signal(&self) -> bool {
        matches!(self, Self::RegularBullish | Self::RegularBearish)
    }
}

/// 背离信号强度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DivergenceStrength {
    Strong,     // 强背离
    Moderate,   // 中等背离
    Weak,       // 弱背离
}

impl DivergenceStrength {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::Strong => 1.0,
            Self::Moderate => 0.7,
            Self::Weak => 0.4,
        }
    }
}

/// 单个背离信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceSignal {
    /// 背离类型
    pub divergence_type: DivergenceType,
    /// 指标名称
    pub indicator: String,
    /// 信号强度
    pub strength: DivergenceStrength,
    /// 置信度 (0-1)
    pub confidence: f64,
    /// 价格变化幅度
    pub price_change: f64,
    /// 指标变化幅度
    pub indicator_change: f64,
    /// 背离持续周期数
    pub duration_bars: usize,
    /// 描述
    pub description: String,
}

/// 综合背离分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergenceAnalysis {
    /// 是否检测到有效背离
    pub has_divergence: bool,
    /// RSI背离信号
    pub rsi_divergence: Option<DivergenceSignal>,
    /// MACD背离信号
    pub macd_divergence: Option<DivergenceSignal>,
    /// OBV背离信号（量价）
    pub obv_divergence: Option<DivergenceSignal>,
    /// Williams %R背离信号
    pub williams_divergence: Option<DivergenceSignal>,
    /// ROC背离信号
    pub roc_divergence: Option<DivergenceSignal>,
    /// 综合背离得分 (-1到1，正数看涨，负数看跌)
    pub composite_score: f64,
    /// 背离数量
    pub divergence_count: usize,
    /// 主要信号方向
    pub primary_direction: String,
    /// 综合置信度
    pub overall_confidence: f64,
    /// 建议动作
    pub suggested_action: String,
    /// 是否为三重背离（极强信号）
    pub is_triple_divergence: bool,
    /// 隐藏背离数量
    pub hidden_divergence_count: usize,
}

impl Default for DivergenceAnalysis {
    fn default() -> Self {
        Self {
            has_divergence: false,
            rsi_divergence: None,
            macd_divergence: None,
            obv_divergence: None,
            williams_divergence: None,
            roc_divergence: None,
            composite_score: 0.0,
            divergence_count: 0,
            primary_direction: "中性".to_string(),
            overall_confidence: 0.0,
            suggested_action: "观望".to_string(),
            is_triple_divergence: false,
            hidden_divergence_count: 0,
        }
    }
}

/// 综合背离分析
pub fn analyze_all_divergences(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
) -> DivergenceAnalysis {
    let len = prices.len();
    if len < 30 {
        return DivergenceAnalysis::default();
    }
    
    // 检测RSI背离（包括隐藏背离）
    let rsi_divergence = detect_rsi_divergence_enhanced(prices);
    
    // 检测MACD背离
    let macd_divergence = detect_macd_divergence(prices);
    
    // 检测OBV（量价）背离
    let obv_divergence = detect_obv_divergence(prices, volumes);
    
    // 检测Williams %R背离
    let williams_divergence = detect_williams_divergence(highs, lows, prices);
    
    // 检测ROC背离
    let roc_divergence = detect_roc_divergence(prices);
    
    // 汇总背离信号
    let mut bullish_score = 0.0;
    let mut bearish_score = 0.0;
    let mut divergence_count = 0;
    let mut total_confidence = 0.0;
    let mut hidden_divergence_count = 0;
    
    // 记录各个背离的方向用于检测三重背离
    let mut divergence_directions: Vec<bool> = Vec::new(); // true = bullish
    
    if let Some(ref div) = rsi_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 1.2;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 1.2;
        }
    }
    
    if let Some(ref div) = macd_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 1.0;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 1.0;
        }
    }
    
    if let Some(ref div) = obv_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.8;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.8;
        }
    }
    
    if let Some(ref div) = williams_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if !div.divergence_type.is_reversal_signal() {
            hidden_divergence_count += 1;
        }
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.7;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.7;
        }
    }
    
    if let Some(ref div) = roc_divergence {
        divergence_count += 1;
        total_confidence += div.confidence;
        divergence_directions.push(div.divergence_type.is_bullish());
        if div.divergence_type.is_bullish() {
            bullish_score += div.strength.to_score() * div.confidence * 0.6;
        } else {
            bearish_score += div.strength.to_score() * div.confidence * 0.6;
        }
    }
    
    // 检测三重背离（3个或以上同向背离）
    let bullish_div_count = divergence_directions.iter().filter(|&&d| d).count();
    let bearish_div_count = divergence_directions.iter().filter(|&&d| !d).count();
    let is_triple_divergence = bullish_div_count >= 3 || bearish_div_count >= 3;
    
    // 三重背离加成
    if is_triple_divergence {
        if bullish_div_count >= 3 {
            bullish_score *= 1.5;
        } else {
            bearish_score *= 1.5;
        }
    }
    
    // 计算综合得分
    let composite_score = if bullish_score > 0.0 || bearish_score > 0.0 {
        (bullish_score - bearish_score) / (bullish_score + bearish_score).max(1.0)
    } else {
        0.0
    };
    
    // 计算综合置信度
    let overall_confidence = if divergence_count > 0 {
        let base_conf = total_confidence / divergence_count as f64;
        // 三重背离提升置信度
        if is_triple_divergence {
            (base_conf * 1.2).min(0.95)
        } else {
            base_conf
        }
    } else {
        0.0
    };
    
    // 确定主要方向
    let primary_direction = if composite_score > 0.3 {
        "看涨".to_string()
    } else if composite_score < -0.3 {
        "看跌".to_string()
    } else {
        "中性".to_string()
    };
    
    // 生成建议
    let suggested_action = generate_divergence_action_enhanced(
        composite_score, 
        divergence_count, 
        overall_confidence,
        is_triple_divergence,
        hidden_divergence_count,
    );
    
    DivergenceAnalysis {
        has_divergence: divergence_count > 0,
        rsi_divergence,
        macd_divergence,
        obv_divergence,
        williams_divergence,
        roc_divergence,
        composite_score,
        divergence_count,
        primary_direction,
        overall_confidence,
        suggested_action,
        is_triple_divergence,
        hidden_divergence_count,
    }
}

/// 检测RSI背离
pub fn detect_rsi_divergence(prices: &[f64]) -> Option<DivergenceSignal> {
    let len = prices.len();
    if len < 30 {
        return None;
    }
    
    // 计算RSI序列
    let mut rsi_values = Vec::new();
    for i in 14..len {
        let rsi_val = rsi::calculate_rsi(&prices[i.saturating_sub(14)..=i]);
        rsi_values.push(rsi_val);
    }
    
    if rsi_values.len() < 15 {
        return None;
    }
    
    let rsi_len = rsi_values.len();
    let price_len = len;
    
    // 寻找价格和RSI的局部极值点
    let (price_lows, price_highs) = find_local_extremes(&prices[price_len.saturating_sub(25)..], 5);
    let (rsi_lows, rsi_highs) = find_local_extremes(&rsi_values[rsi_len.saturating_sub(25)..], 5);
    
    // 检测常规底背离：价格创新低，RSI未创新低
    if price_lows.len() >= 2 && rsi_lows.len() >= 2 {
        let latest_price_low = price_lows.last().unwrap();
        let prev_price_low = price_lows[price_lows.len() - 2];
        let latest_rsi_low = rsi_lows.last().unwrap();
        let prev_rsi_low = rsi_lows[rsi_lows.len() - 2];
        
        // 价格创新低
        if latest_price_low.1 < prev_price_low.1 {
            // RSI未创新低（形成底背离）
            if latest_rsi_low.1 > prev_rsi_low.1 {
                let price_change = (latest_price_low.1 - prev_price_low.1) / prev_price_low.1 * 100.0;
                let rsi_change = latest_rsi_low.1 - prev_rsi_low.1;
                
                let strength = determine_divergence_strength(price_change.abs(), rsi_change.abs());
                let confidence = calculate_divergence_confidence(
                    price_change.abs(), rsi_change.abs(), latest_rsi_low.1
                );
                
                return Some(DivergenceSignal {
                    divergence_type: DivergenceType::RegularBullish,
                    indicator: "RSI".to_string(),
                    strength,
                    confidence,
                    price_change,
                    indicator_change: rsi_change,
                    duration_bars: (latest_price_low.0 - prev_price_low.0).max(1),
                    description: format!(
                        "RSI底背离: 价格下跌{:.1}%但RSI上升{:.1}，预示可能反弹",
                        price_change.abs(), rsi_change
                    ),
                });
            }
        }
    }
    
    // 检测常规顶背离：价格创新高，RSI未创新高
    if price_highs.len() >= 2 && rsi_highs.len() >= 2 {
        let latest_price_high = price_highs.last().unwrap();
        let prev_price_high = price_highs[price_highs.len() - 2];
        let latest_rsi_high = rsi_highs.last().unwrap();
        let prev_rsi_high = rsi_highs[rsi_highs.len() - 2];
        
        // 价格创新高
        if latest_price_high.1 > prev_price_high.1 {
            // RSI未创新高（形成顶背离）
            if latest_rsi_high.1 < prev_rsi_high.1 {
                let price_change = (latest_price_high.1 - prev_price_high.1) / prev_price_high.1 * 100.0;
                let rsi_change = latest_rsi_high.1 - prev_rsi_high.1;
                
                let strength = determine_divergence_strength(price_change, rsi_change.abs());
                let confidence = calculate_divergence_confidence(
                    price_change, rsi_change.abs(), latest_rsi_high.1
                );
                
                return Some(DivergenceSignal {
                    divergence_type: DivergenceType::RegularBearish,
                    indicator: "RSI".to_string(),
                    strength,
                    confidence,
                    price_change,
                    indicator_change: rsi_change,
                    duration_bars: (latest_price_high.0 - prev_price_high.0).max(1),
                    description: format!(
                        "RSI顶背离: 价格上涨{:.1}%但RSI下降{:.1}，预示可能回调",
                        price_change, rsi_change.abs()
                    ),
                });
            }
        }
    }
    
    None
}

/// 检测MACD背离
pub fn detect_macd_divergence(prices: &[f64]) -> Option<DivergenceSignal> {
    let len = prices.len();
    if len < 35 {
        return None;
    }
    
    // 计算MACD柱状图序列
    let mut macd_hist_values = Vec::new();
    for i in 26..len {
        let (_, _, hist) = macd::calculate_macd_full(&prices[..=i]);
        macd_hist_values.push(hist);
    }
    
    if macd_hist_values.len() < 15 {
        return None;
    }
    
    let macd_len = macd_hist_values.len();
    let price_start = len - macd_len;
    
    // 寻找极值点
    let (price_lows, price_highs) = find_local_extremes(&prices[price_start..], 5);
    let (macd_lows, macd_highs) = find_local_extremes(&macd_hist_values, 5);
    
    // 检测底背离
    if price_lows.len() >= 2 && macd_lows.len() >= 2 {
        let latest_price = price_lows.last().unwrap();
        let prev_price = price_lows[price_lows.len() - 2];
        let latest_macd = macd_lows.last().unwrap();
        let prev_macd = macd_lows[macd_lows.len() - 2];
        
        if latest_price.1 < prev_price.1 && latest_macd.1 > prev_macd.1 {
            let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
            let macd_change = latest_macd.1 - prev_macd.1;
            
            let strength = determine_macd_divergence_strength(price_change.abs(), macd_change);
            let confidence = 0.6 + (macd_change.abs() * 100.0).min(0.3);
            
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBullish,
                indicator: "MACD".to_string(),
                strength,
                confidence: confidence.min(0.9),
                price_change,
                indicator_change: macd_change,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: format!(
                    "MACD底背离: 价格下跌{:.1}%但MACD柱状图收窄，动能减弱",
                    price_change.abs()
                ),
            });
        }
    }
    
    // 检测顶背离
    if price_highs.len() >= 2 && macd_highs.len() >= 2 {
        let latest_price = price_highs.last().unwrap();
        let prev_price = price_highs[price_highs.len() - 2];
        let latest_macd = macd_highs.last().unwrap();
        let prev_macd = macd_highs[macd_highs.len() - 2];
        
        if latest_price.1 > prev_price.1 && latest_macd.1 < prev_macd.1 {
            let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
            let macd_change = latest_macd.1 - prev_macd.1;
            
            let strength = determine_macd_divergence_strength(price_change, macd_change.abs());
            let confidence = 0.6 + (macd_change.abs() * 100.0).min(0.3);
            
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBearish,
                indicator: "MACD".to_string(),
                strength,
                confidence: confidence.min(0.9),
                price_change,
                indicator_change: macd_change,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: format!(
                    "MACD顶背离: 价格上涨{:.1}%但MACD柱状图萎缩，动能减弱",
                    price_change
                ),
            });
        }
    }
    
    None
}

/// 检测OBV（量价）背离
pub fn detect_obv_divergence(prices: &[f64], volumes: &[i64]) -> Option<DivergenceSignal> {
    let len = prices.len();
    if len < 20 || volumes.len() < 20 {
        return None;
    }
    
    // 计算OBV序列
    let mut obv_values = Vec::new();
    let mut cumulative_obv = 0i64;
    
    for i in 0..len {
        if i > 0 {
            if prices[i] > prices[i - 1] {
                cumulative_obv += volumes[i];
            } else if prices[i] < prices[i - 1] {
                cumulative_obv -= volumes[i];
            }
        }
        obv_values.push(cumulative_obv as f64);
    }
    
    let obv_len = obv_values.len();
    
    // 寻找极值点
    let (price_lows, price_highs) = find_local_extremes(&prices[len.saturating_sub(20)..], 4);
    let (obv_lows, obv_highs) = find_local_extremes(&obv_values[obv_len.saturating_sub(20)..], 4);
    
    // 检测底背离：价格创新低，OBV未创新低
    if price_lows.len() >= 2 && obv_lows.len() >= 2 {
        let latest_price = price_lows.last().unwrap();
        let prev_price = price_lows[price_lows.len() - 2];
        let latest_obv = obv_lows.last().unwrap();
        let prev_obv = obv_lows[obv_lows.len() - 2];
        
        if latest_price.1 < prev_price.1 && latest_obv.1 > prev_obv.1 {
            let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
            
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBullish,
                indicator: "OBV".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.65,
                price_change,
                indicator_change: 0.0,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "量价底背离: 价格创新低但成交量萎缩，抛压减轻".to_string(),
            });
        }
    }
    
    // 检测顶背离：价格创新高，OBV未创新高
    if price_highs.len() >= 2 && obv_highs.len() >= 2 {
        let latest_price = price_highs.last().unwrap();
        let prev_price = price_highs[price_highs.len() - 2];
        let latest_obv = obv_highs.last().unwrap();
        let prev_obv = obv_highs[obv_highs.len() - 2];
        
        if latest_price.1 > prev_price.1 && latest_obv.1 < prev_obv.1 {
            let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
            
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBearish,
                indicator: "OBV".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.65,
                price_change,
                indicator_change: 0.0,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "量价顶背离: 价格创新高但量能不足，上涨乏力".to_string(),
            });
        }
    }
    
    None
}

/// 寻找局部极值点
fn find_local_extremes(data: &[f64], window: usize) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    let mut lows = Vec::new();
    let mut highs = Vec::new();
    
    if data.len() < window * 2 + 1 {
        return (lows, highs);
    }
    
    for i in window..(data.len() - window) {
        let current = data[i];
        
        // 检查是否为局部最低点
        let is_low = data[i.saturating_sub(window)..i].iter().all(|&x| x >= current) &&
                     data[(i + 1)..=(i + window).min(data.len() - 1)].iter().all(|&x| x >= current);
        
        // 检查是否为局部最高点
        let is_high = data[i.saturating_sub(window)..i].iter().all(|&x| x <= current) &&
                      data[(i + 1)..=(i + window).min(data.len() - 1)].iter().all(|&x| x <= current);
        
        if is_low {
            lows.push((i, current));
        }
        if is_high {
            highs.push((i, current));
        }
    }
    
    (lows, highs)
}

/// 确定RSI背离强度
fn determine_divergence_strength(price_change: f64, rsi_change: f64) -> DivergenceStrength {
    let combined = price_change * 0.5 + rsi_change * 0.5;
    
    if combined > 5.0 {
        DivergenceStrength::Strong
    } else if combined > 2.5 {
        DivergenceStrength::Moderate
    } else {
        DivergenceStrength::Weak
    }
}

/// 确定MACD背离强度
fn determine_macd_divergence_strength(price_change: f64, macd_change: f64) -> DivergenceStrength {
    if price_change > 5.0 && macd_change > 0.01 {
        DivergenceStrength::Strong
    } else if price_change > 2.5 || macd_change > 0.005 {
        DivergenceStrength::Moderate
    } else {
        DivergenceStrength::Weak
    }
}

/// 计算背离置信度
fn calculate_divergence_confidence(price_change: f64, indicator_change: f64, indicator_value: f64) -> f64 {
    let mut confidence = 0.5;
    
    // 价格变化越大，置信度越高
    confidence += (price_change / 10.0).min(0.2);
    
    // 指标变化越大，置信度越高
    confidence += (indicator_change / 20.0).min(0.15);
    
    // RSI在极端区域时，置信度更高
    if indicator_value < 30.0 || indicator_value > 70.0 {
        confidence += 0.1;
    }
    
    confidence.clamp(0.3, 0.9)
}

/// 生成背离建议动作
#[allow(dead_code)]
fn generate_divergence_action(score: f64, count: usize, confidence: f64) -> String {
    if count == 0 {
        return "无明显背离信号，维持当前策略".to_string();
    }
    
    let strength = if count >= 3 { "强" } else if count >= 2 { "中等" } else { "弱" };
    
    if score > 0.5 && confidence > 0.65 {
        format!("{}多重底背离信号，考虑逢低布局", strength)
    } else if score > 0.3 && confidence > 0.55 {
        format!("{}底背离信号，关注反弹机会", strength)
    } else if score < -0.5 && confidence > 0.65 {
        format!("{}多重顶背离信号，考虑减仓或对冲", strength)
    } else if score < -0.3 && confidence > 0.55 {
        format!("{}顶背离信号，警惕回调风险", strength)
    } else {
        "背离信号强度不足，建议观望".to_string()
    }
}

/// 增强版背离建议
fn generate_divergence_action_enhanced(
    score: f64, 
    count: usize, 
    confidence: f64,
    is_triple: bool,
    hidden_count: usize,
) -> String {
    if count == 0 {
        return "无明显背离信号，维持当前策略".to_string();
    }
    
    // 三重背离特殊处理
    if is_triple {
        if score > 0.3 {
            return format!("⚠️ 三重底背离！极强反转信号，置信度{:.0}%，建议积极布局", confidence * 100.0);
        } else if score < -0.3 {
            return format!("⚠️ 三重顶背离！极强见顶信号，置信度{:.0}%，建议立即减仓", confidence * 100.0);
        }
    }
    
    let strength = if count >= 4 { "极强" } else if count >= 3 { "强" } else if count >= 2 { "中等" } else { "弱" };
    
    // 隐藏背离提示
    let hidden_note = if hidden_count > 0 {
        format!("(含{}个隐藏背离，趋势可能延续)", hidden_count)
    } else {
        String::new()
    };
    
    if score > 0.5 && confidence > 0.65 {
        format!("{}多重底背离信号{}，考虑逢低布局", strength, hidden_note)
    } else if score > 0.3 && confidence > 0.55 {
        format!("{}底背离信号{}，关注反弹机会", strength, hidden_note)
    } else if score < -0.5 && confidence > 0.65 {
        format!("{}多重顶背离信号{}，考虑减仓或对冲", strength, hidden_note)
    } else if score < -0.3 && confidence > 0.55 {
        format!("{}顶背离信号{}，警惕回调风险", strength, hidden_note)
    } else {
        "背离信号强度不足，建议观望".to_string()
    }
}

/// 增强版RSI背离检测（包括隐藏背离）
pub fn detect_rsi_divergence_enhanced(prices: &[f64]) -> Option<DivergenceSignal> {
    let len = prices.len();
    if len < 30 {
        return None;
    }
    
    // 计算RSI序列
    let mut rsi_values = Vec::new();
    for i in 14..len {
        let rsi_val = rsi::calculate_rsi(&prices[i.saturating_sub(14)..=i]);
        rsi_values.push(rsi_val);
    }
    
    if rsi_values.len() < 15 {
        return None;
    }
    
    let rsi_len = rsi_values.len();
    let price_len = len;
    
    // 寻找价格和RSI的局部极值点
    let (price_lows, price_highs) = find_local_extremes(&prices[price_len.saturating_sub(25)..], 5);
    let (rsi_lows, rsi_highs) = find_local_extremes(&rsi_values[rsi_len.saturating_sub(25)..], 5);
    
    // 检测常规底背离
    if let Some(signal) = check_regular_bullish_divergence(&price_lows, &rsi_lows, "RSI") {
        return Some(signal);
    }
    
    // 检测常规顶背离
    if let Some(signal) = check_regular_bearish_divergence(&price_highs, &rsi_highs, "RSI") {
        return Some(signal);
    }
    
    // 检测隐藏底背离：价格未创新低（更高的低点），但RSI创新低
    if let Some(signal) = check_hidden_bullish_divergence(&price_lows, &rsi_lows, "RSI") {
        return Some(signal);
    }
    
    // 检测隐藏顶背离：价格未创新高（更低的高点），但RSI创新高
    if let Some(signal) = check_hidden_bearish_divergence(&price_highs, &rsi_highs, "RSI") {
        return Some(signal);
    }
    
    None
}

/// 检测Williams %R背离
pub fn detect_williams_divergence(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) -> Option<DivergenceSignal> {
    let len = closes.len();
    if len < 30 || highs.len() < 30 || lows.len() < 30 {
        return None;
    }
    
    // 计算Williams %R序列
    let mut wr_values = Vec::new();
    for i in 14..len {
        let wr = williams::calculate_williams_r(
            &highs[i.saturating_sub(13)..=i],
            &lows[i.saturating_sub(13)..=i],
            &closes[i.saturating_sub(13)..=i],
            14,
        );
        wr_values.push(wr);
    }
    
    if wr_values.len() < 15 {
        return None;
    }
    
    let wr_len = wr_values.len();
    
    // 寻找极值点
    let (price_lows, price_highs) = find_local_extremes(&closes[len.saturating_sub(20)..], 4);
    let (wr_lows, wr_highs) = find_local_extremes(&wr_values[wr_len.saturating_sub(20)..], 4);
    
    // 注意：Williams %R是负值，越低（越接近-100）表示越超卖
    // 底背离：价格创新低，W%R没创新低（W%R更高，即更接近0）
    if price_lows.len() >= 2 && wr_lows.len() >= 2 {
        let latest_price = price_lows.last().unwrap();
        let prev_price = price_lows[price_lows.len() - 2];
        let latest_wr = wr_lows.last().unwrap();
        let prev_wr = wr_lows[wr_lows.len() - 2];
        
        // 价格创新低，W%R更高（没创新低）
        if latest_price.1 < prev_price.1 && latest_wr.1 > prev_wr.1 {
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBullish,
                indicator: "Williams%R".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.65,
                price_change: (latest_price.1 - prev_price.1) / prev_price.1 * 100.0,
                indicator_change: latest_wr.1 - prev_wr.1,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "Williams%R底背离：价格新低但动量减弱".to_string(),
            });
        }
    }
    
    // 顶背离检测
    if price_highs.len() >= 2 && wr_highs.len() >= 2 {
        let latest_price = price_highs.last().unwrap();
        let prev_price = price_highs[price_highs.len() - 2];
        let latest_wr = wr_highs.last().unwrap();
        let prev_wr = wr_highs[wr_highs.len() - 2];
        
        // 价格创新高，W%R更低（没创新高，因为W%R是负值）
        if latest_price.1 > prev_price.1 && latest_wr.1 < prev_wr.1 {
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBearish,
                indicator: "Williams%R".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.65,
                price_change: (latest_price.1 - prev_price.1) / prev_price.1 * 100.0,
                indicator_change: latest_wr.1 - prev_wr.1,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "Williams%R顶背离：价格新高但动量不足".to_string(),
            });
        }
    }
    
    None
}

/// 检测ROC背离
pub fn detect_roc_divergence(prices: &[f64]) -> Option<DivergenceSignal> {
    let len = prices.len();
    if len < 30 {
        return None;
    }
    
    // 计算ROC序列
    let roc_values = roc::calculate_roc_series(prices, 12);
    
    if roc_values.len() < 20 {
        return None;
    }
    
    let roc_len = roc_values.len();
    
    // 寻找极值点
    let (price_lows, price_highs) = find_local_extremes(&prices[len.saturating_sub(20)..], 4);
    let (roc_lows, roc_highs) = find_local_extremes(&roc_values[roc_len.saturating_sub(20)..], 4);
    
    // 底背离
    if price_lows.len() >= 2 && roc_lows.len() >= 2 {
        let latest_price = price_lows.last().unwrap();
        let prev_price = price_lows[price_lows.len() - 2];
        let latest_roc = roc_lows.last().unwrap();
        let prev_roc = roc_lows[roc_lows.len() - 2];
        
        if latest_price.1 < prev_price.1 && latest_roc.1 > prev_roc.1 {
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBullish,
                indicator: "ROC".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.60,
                price_change: (latest_price.1 - prev_price.1) / prev_price.1 * 100.0,
                indicator_change: latest_roc.1 - prev_roc.1,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "ROC底背离：价格新低但变动率收窄".to_string(),
            });
        }
    }
    
    // 顶背离
    if price_highs.len() >= 2 && roc_highs.len() >= 2 {
        let latest_price = price_highs.last().unwrap();
        let prev_price = price_highs[price_highs.len() - 2];
        let latest_roc = roc_highs.last().unwrap();
        let prev_roc = roc_highs[roc_highs.len() - 2];
        
        if latest_price.1 > prev_price.1 && latest_roc.1 < prev_roc.1 {
            return Some(DivergenceSignal {
                divergence_type: DivergenceType::RegularBearish,
                indicator: "ROC".to_string(),
                strength: DivergenceStrength::Moderate,
                confidence: 0.60,
                price_change: (latest_price.1 - prev_price.1) / prev_price.1 * 100.0,
                indicator_change: latest_roc.1 - prev_roc.1,
                duration_bars: (latest_price.0 - prev_price.0).max(1),
                description: "ROC顶背离：价格新高但变动率下降".to_string(),
            });
        }
    }
    
    None
}

/// 检测常规看涨背离
fn check_regular_bullish_divergence(
    price_lows: &[(usize, f64)],
    indicator_lows: &[(usize, f64)],
    indicator_name: &str,
) -> Option<DivergenceSignal> {
    if price_lows.len() < 2 || indicator_lows.len() < 2 {
        return None;
    }
    
    let latest_price = price_lows.last().unwrap();
    let prev_price = price_lows[price_lows.len() - 2];
    let latest_ind = indicator_lows.last().unwrap();
    let prev_ind = indicator_lows[indicator_lows.len() - 2];
    
    // 价格创新低，指标未创新低
    if latest_price.1 < prev_price.1 && latest_ind.1 > prev_ind.1 {
        let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
        let ind_change = latest_ind.1 - prev_ind.1;
        
        let strength = determine_divergence_strength(price_change.abs(), ind_change.abs());
        let confidence = calculate_divergence_confidence(price_change.abs(), ind_change.abs(), latest_ind.1);
        
        return Some(DivergenceSignal {
            divergence_type: DivergenceType::RegularBullish,
            indicator: indicator_name.to_string(),
            strength,
            confidence,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!("{}底背离: 价格下跌{:.1}%但指标上升{:.1}", 
                indicator_name, price_change.abs(), ind_change),
        });
    }
    
    None
}

/// 检测常规看跌背离
fn check_regular_bearish_divergence(
    price_highs: &[(usize, f64)],
    indicator_highs: &[(usize, f64)],
    indicator_name: &str,
) -> Option<DivergenceSignal> {
    if price_highs.len() < 2 || indicator_highs.len() < 2 {
        return None;
    }
    
    let latest_price = price_highs.last().unwrap();
    let prev_price = price_highs[price_highs.len() - 2];
    let latest_ind = indicator_highs.last().unwrap();
    let prev_ind = indicator_highs[indicator_highs.len() - 2];
    
    // 价格创新高，指标未创新高
    if latest_price.1 > prev_price.1 && latest_ind.1 < prev_ind.1 {
        let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
        let ind_change = latest_ind.1 - prev_ind.1;
        
        let strength = determine_divergence_strength(price_change, ind_change.abs());
        let confidence = calculate_divergence_confidence(price_change, ind_change.abs(), latest_ind.1);
        
        return Some(DivergenceSignal {
            divergence_type: DivergenceType::RegularBearish,
            indicator: indicator_name.to_string(),
            strength,
            confidence,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!("{}顶背离: 价格上涨{:.1}%但指标下降{:.1}", 
                indicator_name, price_change, ind_change.abs()),
        });
    }
    
    None
}

/// 检测隐藏看涨背离（趋势延续信号）
fn check_hidden_bullish_divergence(
    price_lows: &[(usize, f64)],
    indicator_lows: &[(usize, f64)],
    indicator_name: &str,
) -> Option<DivergenceSignal> {
    if price_lows.len() < 2 || indicator_lows.len() < 2 {
        return None;
    }
    
    let latest_price = price_lows.last().unwrap();
    let prev_price = price_lows[price_lows.len() - 2];
    let latest_ind = indicator_lows.last().unwrap();
    let prev_ind = indicator_lows[indicator_lows.len() - 2];
    
    // 隐藏底背离：价格更高的低点，指标更低的低点
    if latest_price.1 > prev_price.1 && latest_ind.1 < prev_ind.1 {
        let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
        let ind_change = latest_ind.1 - prev_ind.1;
        
        return Some(DivergenceSignal {
            divergence_type: DivergenceType::HiddenBullish,
            indicator: indicator_name.to_string(),
            strength: DivergenceStrength::Moderate,
            confidence: 0.60,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!("{}隐藏底背离: 上涨趋势可能延续", indicator_name),
        });
    }
    
    None
}

/// 检测隐藏看跌背离（趋势延续信号）
fn check_hidden_bearish_divergence(
    price_highs: &[(usize, f64)],
    indicator_highs: &[(usize, f64)],
    indicator_name: &str,
) -> Option<DivergenceSignal> {
    if price_highs.len() < 2 || indicator_highs.len() < 2 {
        return None;
    }
    
    let latest_price = price_highs.last().unwrap();
    let prev_price = price_highs[price_highs.len() - 2];
    let latest_ind = indicator_highs.last().unwrap();
    let prev_ind = indicator_highs[indicator_highs.len() - 2];
    
    // 隐藏顶背离：价格更低的高点，指标更高的高点
    if latest_price.1 < prev_price.1 && latest_ind.1 > prev_ind.1 {
        let price_change = (latest_price.1 - prev_price.1) / prev_price.1 * 100.0;
        let ind_change = latest_ind.1 - prev_ind.1;
        
        return Some(DivergenceSignal {
            divergence_type: DivergenceType::HiddenBearish,
            indicator: indicator_name.to_string(),
            strength: DivergenceStrength::Moderate,
            confidence: 0.60,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!("{}隐藏顶背离: 下跌趋势可能延续", indicator_name),
        });
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_find_extremes() {
        let data = vec![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5];
        let (lows, highs) = find_local_extremes(&data, 1);
        assert!(!lows.is_empty() || !highs.is_empty());
    }
}

