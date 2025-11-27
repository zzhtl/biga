//! K线形态分析模块

use serde::{Deserialize, Serialize};

/// K线形态类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    // 看涨形态
    Hammer,              // 锤子线
    InvertedHammer,      // 倒锤子
    BullishEngulfing,    // 看涨吞没
    MorningStar,         // 早晨之星
    ThreeWhiteSoldiers,  // 三只白兵
    
    // 看跌形态
    HangingMan,          // 吊颈线
    ShootingStar,        // 流星线
    BearishEngulfing,    // 看跌吞没
    EveningStar,         // 黄昏之星
    ThreeBlackCrows,     // 三只乌鸦
    
    // 中性形态
    Doji,                // 十字星
    SpinningTop,         // 纺锤线
}

impl PatternType {
    pub fn to_string(&self) -> String {
        match self {
            Self::Hammer => "锤子线".to_string(),
            Self::InvertedHammer => "倒锤子".to_string(),
            Self::BullishEngulfing => "看涨吞没".to_string(),
            Self::MorningStar => "早晨之星".to_string(),
            Self::ThreeWhiteSoldiers => "三只白兵".to_string(),
            Self::HangingMan => "吊颈线".to_string(),
            Self::ShootingStar => "流星线".to_string(),
            Self::BearishEngulfing => "看跌吞没".to_string(),
            Self::EveningStar => "黄昏之星".to_string(),
            Self::ThreeBlackCrows => "三只乌鸦".to_string(),
            Self::Doji => "十字星".to_string(),
            Self::SpinningTop => "纺锤线".to_string(),
        }
    }
    
    pub fn is_bullish(&self) -> bool {
        matches!(self, 
            Self::Hammer | Self::InvertedHammer | Self::BullishEngulfing |
            Self::MorningStar | Self::ThreeWhiteSoldiers
        )
    }
    
    pub fn is_bearish(&self) -> bool {
        matches!(self,
            Self::HangingMan | Self::ShootingStar | Self::BearishEngulfing |
            Self::EveningStar | Self::ThreeBlackCrows
        )
    }
}

/// 形态识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognition {
    pub pattern_type: String,
    pub is_bullish: bool,
    pub reliability: f64,
    pub description: String,
}

/// 识别K线形态
pub fn recognize_patterns(
    opens: &[f64],
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
) -> Vec<PatternRecognition> {
    let mut patterns = Vec::new();
    let len = opens.len();
    
    if len < 3 {
        return patterns;
    }
    
    // 检测单根K线形态
    if let Some(pattern) = detect_single_candle(&opens[len-1], &closes[len-1], &highs[len-1], &lows[len-1]) {
        patterns.push(pattern);
    }
    
    // 检测双根K线形态
    if len >= 2 {
        if let Some(pattern) = detect_double_candle(
            &opens[len-2..], &closes[len-2..], &highs[len-2..], &lows[len-2..]
        ) {
            patterns.push(pattern);
        }
    }
    
    // 检测三根K线形态
    if len >= 3 {
        if let Some(pattern) = detect_triple_candle(
            &opens[len-3..], &closes[len-3..], &highs[len-3..], &lows[len-3..]
        ) {
            patterns.push(pattern);
        }
    }
    
    patterns
}

/// 检测单根K线形态
fn detect_single_candle(open: &f64, close: &f64, high: &f64, low: &f64) -> Option<PatternRecognition> {
    let body = (close - open).abs();
    let total_range = high - low;
    
    if total_range == 0.0 {
        return None;
    }
    
    let body_ratio = body / total_range;
    let upper_shadow = high - open.max(*close);
    let lower_shadow = open.min(*close) - low;
    
    // 十字星：实体很小
    if body_ratio < 0.1 {
        return Some(PatternRecognition {
            pattern_type: PatternType::Doji.to_string(),
            is_bullish: false,
            reliability: 0.6,
            description: "十字星，市场犹豫不决".to_string(),
        });
    }
    
    // 锤子线：下影线长，上影线短，实体小
    if lower_shadow > body * 2.0 && upper_shadow < body * 0.5 && body_ratio < 0.4 {
        let is_bullish = close > open;
        return Some(PatternRecognition {
            pattern_type: if is_bullish { PatternType::Hammer.to_string() } else { PatternType::HangingMan.to_string() },
            is_bullish,
            reliability: 0.65,
            description: if is_bullish { "锤子线，可能反转上涨".to_string() } else { "吊颈线，可能见顶".to_string() },
        });
    }
    
    // 流星线/倒锤子：上影线长，下影线短，实体小
    if upper_shadow > body * 2.0 && lower_shadow < body * 0.5 && body_ratio < 0.4 {
        let is_bullish = close > open;
        return Some(PatternRecognition {
            pattern_type: if is_bullish { PatternType::InvertedHammer.to_string() } else { PatternType::ShootingStar.to_string() },
            is_bullish,
            reliability: 0.60,
            description: if is_bullish { "倒锤子，可能反转上涨".to_string() } else { "流星线，可能见顶".to_string() },
        });
    }
    
    // 纺锤线：实体较小，上下影线相近
    if body_ratio < 0.3 && (upper_shadow - lower_shadow).abs() < body * 0.5 {
        return Some(PatternRecognition {
            pattern_type: PatternType::SpinningTop.to_string(),
            is_bullish: false,
            reliability: 0.5,
            description: "纺锤线，市场方向不明".to_string(),
        });
    }
    
    None
}

/// 检测双根K线形态
fn detect_double_candle(
    opens: &[f64],
    closes: &[f64],
    _highs: &[f64],
    _lows: &[f64],
) -> Option<PatternRecognition> {
    if opens.len() < 2 || closes.len() < 2 {
        return None;
    }
    
    let prev_body = closes[0] - opens[0];
    let curr_body = closes[1] - opens[1];
    
    // 看涨吞没：前阴后阳，后实体完全吞没前实体
    if prev_body < 0.0 && curr_body > 0.0 && curr_body.abs() > prev_body.abs() * 1.2 {
        if opens[1] < closes[0] && closes[1] > opens[0] {
            return Some(PatternRecognition {
                pattern_type: PatternType::BullishEngulfing.to_string(),
                is_bullish: true,
                reliability: 0.70,
                description: "看涨吞没形态，可能反转上涨".to_string(),
            });
        }
    }
    
    // 看跌吞没：前阳后阴，后实体完全吞没前实体
    if prev_body > 0.0 && curr_body < 0.0 && curr_body.abs() > prev_body.abs() * 1.2 {
        if opens[1] > closes[0] && closes[1] < opens[0] {
            return Some(PatternRecognition {
                pattern_type: PatternType::BearishEngulfing.to_string(),
                is_bullish: false,
                reliability: 0.70,
                description: "看跌吞没形态，可能反转下跌".to_string(),
            });
        }
    }
    
    None
}

/// 检测三根K线形态
fn detect_triple_candle(
    opens: &[f64],
    closes: &[f64],
    _highs: &[f64],
    _lows: &[f64],
) -> Option<PatternRecognition> {
    if opens.len() < 3 || closes.len() < 3 {
        return None;
    }
    
    let body1 = closes[0] - opens[0];
    let body2 = closes[1] - opens[1];
    let body3 = closes[2] - opens[2];
    
    // 三只白兵：连续三根阳线，每根收盘价高于前一根
    if body1 > 0.0 && body2 > 0.0 && body3 > 0.0 {
        if closes[1] > closes[0] && closes[2] > closes[1] {
            return Some(PatternRecognition {
                pattern_type: PatternType::ThreeWhiteSoldiers.to_string(),
                is_bullish: true,
                reliability: 0.75,
                description: "三只白兵形态，强烈看涨信号".to_string(),
            });
        }
    }
    
    // 三只乌鸦：连续三根阴线，每根收盘价低于前一根
    if body1 < 0.0 && body2 < 0.0 && body3 < 0.0 {
        if closes[1] < closes[0] && closes[2] < closes[1] {
            return Some(PatternRecognition {
                pattern_type: PatternType::ThreeBlackCrows.to_string(),
                is_bullish: false,
                reliability: 0.75,
                description: "三只乌鸦形态，强烈看跌信号".to_string(),
            });
        }
    }
    
    // 早晨之星
    let mid_body_ratio = body2.abs() / body1.abs().max(body3.abs());
    if body1 < 0.0 && body3 > 0.0 && mid_body_ratio < 0.3 {
        if closes[2] > (opens[0] + closes[0]) / 2.0 {
            return Some(PatternRecognition {
                pattern_type: PatternType::MorningStar.to_string(),
                is_bullish: true,
                reliability: 0.70,
                description: "早晨之星形态，可能反转上涨".to_string(),
            });
        }
    }
    
    // 黄昏之星
    if body1 > 0.0 && body3 < 0.0 && mid_body_ratio < 0.3 {
        if closes[2] < (opens[0] + closes[0]) / 2.0 {
            return Some(PatternRecognition {
                pattern_type: PatternType::EveningStar.to_string(),
                is_bullish: false,
                reliability: 0.70,
                description: "黄昏之星形态，可能反转下跌".to_string(),
            });
        }
    }
    
    None
}

