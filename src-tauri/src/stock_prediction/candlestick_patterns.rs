/// K线形态识别模块
/// 识别经典的看涨/看跌K线形态

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandlePattern {
    Hammer,           // 锤子线（看涨）
    InvertedHammer,   // 倒锤子线（看涨）
    ShootingStar,     // 射击之星（看跌）
    HangingMan,       // 上吊线（看跌）
    Doji,             // 十字星（反转）
    BullishEngulfing, // 看涨吞没
    BearishEngulfing, // 看跌吞没
    PiercingLine,     // 刺透形态（看涨）
    DarkCloudCover,   // 乌云盖顶（看跌）
    MorningStar,      // 启明星（底部反转）
    EveningStar,      // 黄昏星（顶部反转）
    ThreeWhiteSoldiers, // 三只白鸦（看涨）
    ThreeBlackCrows,    // 三只黑鸦（看跌）
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Bullish,  // 看涨
    Bearish,  // 看跌
    Neutral,  // 中性
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognition {
    pub pattern: CandlePattern,
    pub position: usize,          // 形态出现的位置
    pub strength: f64,            // 形态强度 0-1
    pub reliability: f64,         // 可靠性 0-1
    pub direction: Direction,     // 看涨/看跌
    pub description: String,      // 形态描述
    pub location_type: String,    // 位置类型：顶部/底部/中继
    pub confirmed: bool,          // 是否已确认
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

impl Candle {
    // 实体大小
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
    
    // 上影线长度
    pub fn upper_shadow(&self) -> f64 {
        self.high - self.open.max(self.close)
    }
    
    // 下影线长度
    pub fn lower_shadow(&self) -> f64 {
        self.open.min(self.close) - self.low
    }
    
    // 是否阳线
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
    
    // 全天振幅
    pub fn range(&self) -> f64 {
        self.high - self.low
    }
    
    // 实体占全天振幅的比例
    pub fn body_ratio(&self) -> f64 {
        let range = self.range();
        if range > 0.0 {
            self.body_size() / range
        } else {
            0.0
        }
    }
}

/// 识别锤子线（Hammer）
/// 特征：下影线长，实体小，上影线短或无
pub fn is_hammer(candle: &Candle) -> Option<f64> {
    let body = candle.body_size();
    let lower_shadow = candle.lower_shadow();
    let upper_shadow = candle.upper_shadow();
    let range = candle.range();
    
    if range == 0.0 {
        return None;
    }
    
    // 条件：下影线至少是实体的2倍，上影线很小
    if lower_shadow >= body * 2.0 && upper_shadow < body * 0.5 {
        // 强度：下影线越长越强
        let strength = (lower_shadow / range).min(1.0);
        Some(strength)
    } else {
        None
    }
}

/// 识别射击之星（Shooting Star）
/// 特征：上影线长，实体小，下影线短或无
pub fn is_shooting_star(candle: &Candle) -> Option<f64> {
    let body = candle.body_size();
    let lower_shadow = candle.lower_shadow();
    let upper_shadow = candle.upper_shadow();
    let range = candle.range();
    
    if range == 0.0 {
        return None;
    }
    
    if upper_shadow >= body * 2.0 && lower_shadow < body * 0.5 {
        let strength = (upper_shadow / range).min(1.0);
        Some(strength)
    } else {
        None
    }
}

/// 识别十字星（Doji）
/// 特征：实体非常小，上下影线较长
pub fn is_doji(candle: &Candle) -> Option<f64> {
    let body = candle.body_size();
    let range = candle.range();
    
    if range == 0.0 {
        return None;
    }
    
    // 实体占振幅小于5%
    if body / range < 0.05 {
        let strength = 1.0 - (body / range) * 20.0; // 实体越小强度越高
        Some(strength.max(0.5))
    } else {
        None
    }
}

/// 识别看涨吞没（Bullish Engulfing）
/// 特征：前一根阴线，后一根阳线完全吞没前一根
pub fn is_bullish_engulfing(prev: &Candle, curr: &Candle) -> Option<f64> {
    if !prev.is_bullish() && curr.is_bullish() {
        // 当前阳线的实体完全包含前一根阴线的实体
        if curr.open <= prev.close && curr.close >= prev.open {
            let engulf_ratio = curr.body_size() / prev.body_size().max(0.001);
            let strength = engulf_ratio.min(2.0) / 2.0; // 吞没越多越强
            Some(strength)
        } else {
            None
        }
    } else {
        None
    }
}

/// 识别看跌吞没（Bearish Engulfing）
pub fn is_bearish_engulfing(prev: &Candle, curr: &Candle) -> Option<f64> {
    if prev.is_bullish() && !curr.is_bullish() {
        if curr.open >= prev.close && curr.close <= prev.open {
            let engulf_ratio = curr.body_size() / prev.body_size().max(0.001);
            let strength = engulf_ratio.min(2.0) / 2.0;
            Some(strength)
        } else {
            None
        }
    } else {
        None
    }
}

/// 识别刺透形态（Piercing Line）
/// 特征：前阴后阳，阳线收盘价超过前一根阴线实体的一半
pub fn is_piercing_line(prev: &Candle, curr: &Candle) -> Option<f64> {
    if !prev.is_bullish() && curr.is_bullish() {
        let prev_mid = (prev.open + prev.close) / 2.0;
        
        // 阳线开盘低于阴线收盘，收盘高于阴线中点
        if curr.open < prev.close && curr.close > prev_mid && curr.close < prev.open {
            let penetration = (curr.close - prev.close) / prev.body_size();
            let strength = penetration.min(1.0);
            Some(strength)
        } else {
            None
        }
    } else {
        None
    }
}

/// 识别乌云盖顶（Dark Cloud Cover）
pub fn is_dark_cloud_cover(prev: &Candle, curr: &Candle) -> Option<f64> {
    if prev.is_bullish() && !curr.is_bullish() {
        let prev_mid = (prev.open + prev.close) / 2.0;
        
        if curr.open > prev.close && curr.close < prev_mid && curr.close > prev.open {
            let penetration = (prev.close - curr.close) / prev.body_size();
            let strength = penetration.min(1.0);
            Some(strength)
        } else {
            None
        }
    } else {
        None
    }
}

/// 识别启明星（Morning Star）
/// 特征：三根K线，第一根阴线，第二根小实体（十字星），第三根阳线
pub fn is_morning_star(first: &Candle, second: &Candle, third: &Candle) -> Option<f64> {
    if !first.is_bullish() && third.is_bullish() {
        // 第二根K线实体很小
        let second_body_ratio = second.body_ratio();
        if second_body_ratio < 0.3 {
            // 第三根阳线收盘价进入第一根阴线实体
            if third.close > (first.open + first.close) / 2.0 {
                let strength = (1.0 - second_body_ratio) * 0.5 + 
                              ((third.close - first.close) / first.body_size()).min(0.5);
                return Some(strength.min(1.0));
            }
        }
    }
    None
}

/// 识别黄昏星（Evening Star）
pub fn is_evening_star(first: &Candle, second: &Candle, third: &Candle) -> Option<f64> {
    if first.is_bullish() && !third.is_bullish() {
        let second_body_ratio = second.body_ratio();
        if second_body_ratio < 0.3 {
            if third.close < (first.open + first.close) / 2.0 {
                let strength = (1.0 - second_body_ratio) * 0.5 + 
                              ((first.close - third.close) / first.body_size()).min(0.5);
                return Some(strength.min(1.0));
            }
        }
    }
    None
}

/// 识别三只白鸦（Three White Soldiers）
/// 特征：连续三根阳线，一根比一根高
pub fn is_three_white_soldiers(c1: &Candle, c2: &Candle, c3: &Candle) -> Option<f64> {
    if c1.is_bullish() && c2.is_bullish() && c3.is_bullish() {
        // 收盘价逐步上升
        if c2.close > c1.close && c3.close > c2.close {
            // 开盘价在前一根实体内
            if c2.open > c1.open && c2.open < c1.close &&
               c3.open > c2.open && c3.open < c2.close {
                let strength = 0.8; // 固定较高强度
                return Some(strength);
            }
        }
    }
    None
}

/// 识别三只黑鸦（Three Black Crows）
pub fn is_three_black_crows(c1: &Candle, c2: &Candle, c3: &Candle) -> Option<f64> {
    if !c1.is_bullish() && !c2.is_bullish() && !c3.is_bullish() {
        if c2.close < c1.close && c3.close < c2.close {
            if c2.open < c1.open && c2.open > c1.close &&
               c3.open < c2.open && c3.open > c2.close {
                let strength = 0.8;
                return Some(strength);
            }
        }
    }
    None
}

/// 判断形态位置类型（顶部/底部/中继）
fn determine_location_type(candles: &[Candle], position: usize, direction: Direction) -> String {
    if position < 20 || position >= candles.len() {
        return "中继".to_string();
    }
    
    // 取前20根K线作为判断依据
    let lookback = &candles[position.saturating_sub(20)..position];
    let current_price = candles[position].close;
    
    // 计算最高价和最低价
    let max_price = lookback.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let min_price = lookback.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    
    let range = max_price - min_price;
    let position_pct = (current_price - min_price) / range;
    
    match direction {
        Direction::Bullish => {
            // 看涨形态：在底部区域(< 30%)可靠性更高
            if position_pct < 0.3 {
                "底部反转".to_string()
            } else if position_pct < 0.7 {
                "中继上涨".to_string()
            } else {
                "顶部警惕".to_string()
            }
        }
        Direction::Bearish => {
            // 看跌形态：在顶部区域(> 70%)可靠性更高
            if position_pct > 0.7 {
                "顶部反转".to_string()
            } else if position_pct > 0.3 {
                "中继下跌".to_string()
            } else {
                "底部警惕".to_string()
            }
        }
        Direction::Neutral => "中继震荡".to_string(),
    }
}

/// 创建形态识别记录的辅助函数
fn create_pattern_recognition(
    candles: &[Candle],
    pattern: CandlePattern,
    position: usize,
    strength: f64,
    reliability: f64,
    direction: Direction,
    description: String,
) -> PatternRecognition {
    let location_type = determine_location_type(candles, position, direction);
    let confirmed = position < candles.len() - 1; // 不是最后一根K线才算确认
    
    PatternRecognition {
        pattern,
        position,
        strength,
        reliability,
        direction,
        description,
        location_type,
        confirmed,
    }
}

/// 综合识别所有K线形态（增强版）
pub fn identify_all_patterns(
    candles: &[Candle],
) -> Vec<PatternRecognition> {
    let mut patterns = Vec::new();
    let n = candles.len();
    
    if n < 3 {
        return patterns;
    }
    
    // 单根K线形态（最后一根）
    let last = &candles[n - 1];
    
    if let Some(strength) = is_hammer(last) {
        patterns.push(create_pattern_recognition(
            candles,
            CandlePattern::Hammer,
            n - 1,
            strength,
            0.75,
            Direction::Bullish,
            "锤子线：底部反转信号，看涨".to_string(),
        ));
    }
    
    if let Some(strength) = is_shooting_star(last) {
        patterns.push(create_pattern_recognition(
            candles,
            CandlePattern::ShootingStar,
            n - 1,
            strength,
            0.75,
            Direction::Bearish,
            "射击之星：顶部反转信号，看跌".to_string(),
        ));
    }
    
    if let Some(strength) = is_doji(last) {
        patterns.push(create_pattern_recognition(
            candles,
            CandlePattern::Doji,
            n - 1,
            strength,
            0.60,
            Direction::Neutral,
            "十字星：趋势可能反转，观望".to_string(),
        ));
    }
    
    // 两根K线形态
    if n >= 2 {
        let prev = &candles[n - 2];
        let curr = &candles[n - 1];
        
        if let Some(strength) = is_bullish_engulfing(prev, curr) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::BullishEngulfing,
                n - 1,
                strength,
                0.80,
                Direction::Bullish,
                "看涨吞没：强烈看涨信号".to_string(),
            ));
        }
        
        if let Some(strength) = is_bearish_engulfing(prev, curr) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::BearishEngulfing,
                n - 1,
                strength,
                0.80,
                Direction::Bearish,
                "看跌吞没：强烈看跌信号".to_string(),
            ));
        }
        
        if let Some(strength) = is_piercing_line(prev, curr) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::PiercingLine,
                n - 1,
                strength,
                0.70,
                Direction::Bullish,
                "刺透形态：底部反转，看涨".to_string(),
            ));
        }
        
        if let Some(strength) = is_dark_cloud_cover(prev, curr) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::DarkCloudCover,
                n - 1,
                strength,
                0.70,
                Direction::Bearish,
                "乌云盖顶：顶部反转，看跌".to_string(),
            ));
        }
    }
    
    // 三根K线形态
    if n >= 3 {
        let c1 = &candles[n - 3];
        let c2 = &candles[n - 2];
        let c3 = &candles[n - 1];
        
        if let Some(strength) = is_morning_star(c1, c2, c3) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::MorningStar,
                n - 1,
                strength,
                0.85,
                Direction::Bullish,
                "启明星：强烈底部反转信号".to_string(),
            ));
        }
        
        if let Some(strength) = is_evening_star(c1, c2, c3) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::EveningStar,
                n - 1,
                strength,
                0.85,
                Direction::Bearish,
                "黄昏星：强烈顶部反转信号".to_string(),
            ));
        }
        
        if let Some(strength) = is_three_white_soldiers(c1, c2, c3) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::ThreeWhiteSoldiers,
                n - 1,
                strength,
                0.80,
                Direction::Bullish,
                "三只白鸦：强势上涨信号".to_string(),
            ));
        }
        
        if let Some(strength) = is_three_black_crows(c1, c2, c3) {
            patterns.push(create_pattern_recognition(
                candles,
                CandlePattern::ThreeBlackCrows,
                n - 1,
                strength,
                0.80,
                Direction::Bearish,
                "三只黑鸦：强势下跌信号".to_string(),
            ));
        }
    }
    
    patterns
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hammer() {
        let hammer = Candle {
            open: 10.0,
            high: 10.5,
            low: 8.0,
            close: 10.2,
            volume: 1000,
        };
        
        assert!(is_hammer(&hammer).is_some());
    }
    
    #[test]
    fn test_doji() {
        let doji = Candle {
            open: 10.0,
            high: 10.5,
            low: 9.5,
            close: 10.05,
            volume: 1000,
        };
        
        assert!(is_doji(&doji).is_some());
    }
} 