//! 金融级技术指标计算模块
//! 
//! 提供高精度的核心技术指标计算

use serde::{Deserialize, Serialize};

// ============================================================================
// 数据结构定义
// ============================================================================

/// MACD指标数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdData {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
    pub trend: TrendDirection,
    pub cross_signal: CrossSignal,
}

/// RSI指标数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsiData {
    pub value: f64,
    pub signal: RsiSignal,
    pub divergence: Option<DivergenceType>,
}

/// 布林带数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
    pub width: f64,
    pub position: f64,  // 当前价格在带中的位置 (0-1)
    pub squeeze: bool,   // 布林带收缩
}

/// ATR波动率数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtrData {
    pub value: f64,
    pub normalized: f64,  // ATR/价格
    pub volatility_level: VolatilityLevel,
    pub trend_strength: f64,  // ADX值
}

/// 综合技术指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    pub macd: MacdData,
    pub rsi: RsiData,
    pub bollinger: BollingerBands,
    pub atr: AtrData,
    pub ema_20: f64,
    pub ema_50: f64,
    pub ema_200: f64,
    pub volume_ratio: f64,
    pub obv_trend: TrendDirection,
}

// ============================================================================
// 枚举类型
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrendDirection {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CrossSignal {
    GoldenCross,
    DeathCross,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RsiSignal {
    ExtemeOverbought,  // >80
    Overbought,        // 70-80
    Neutral,           // 30-70
    Oversold,          // 20-30
    ExtremeOversold,   // <20
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VolatilityLevel {
    VeryLow,
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DivergenceType {
    BullishDivergence,  // 价格新低，RSI没有新低
    BearishDivergence,  // 价格新高，RSI没有新高
}

// ============================================================================
// 核心计算函数
// ============================================================================

/// 计算所有技术指标
pub fn calculate_all_indicators(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
) -> TechnicalIndicators {
    // MACD
    let macd = calculate_macd_advanced(prices);
    
    // RSI
    let rsi = calculate_rsi_advanced(prices);
    
    // 布林带
    let bollinger = calculate_bollinger_bands(prices, 20, 2.0);
    
    // ATR和ADX
    let atr = calculate_atr_advanced(highs, lows, prices);
    
    // EMA
    let ema_20 = calculate_ema(prices, 20);
    let ema_50 = calculate_ema(prices, 50);
    let ema_200 = calculate_ema(prices, 200);
    
    // 成交量指标
    let volume_ratio = calculate_volume_ratio(volumes);
    let obv_trend = calculate_obv_trend(prices, volumes);
    
    TechnicalIndicators {
        macd,
        rsi,
        bollinger,
        atr,
        ema_20,
        ema_50,
        ema_200,
        volume_ratio,
        obv_trend,
    }
}

/// 高级MACD计算（包含趋势和交叉信号）
pub fn calculate_macd_advanced(prices: &[f64]) -> MacdData {
    if prices.len() < 26 {
        return MacdData {
            macd: 0.0,
            signal: 0.0,
            histogram: 0.0,
            trend: TrendDirection::Neutral,
            cross_signal: CrossSignal::None,
        };
    }
    
    // 计算EMA
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    
    // MACD线
    let macd = ema12 - ema26;
    
    // 信号线（MACD的9日EMA）
    let macd_values = calculate_macd_series(prices);
    let signal = if macd_values.len() >= 9 {
        calculate_ema(&macd_values, 9)
    } else {
        0.0
    };
    
    // 柱状图
    let histogram = macd - signal;
    
    // 判断趋势
    let trend = if macd > 0.0 && histogram > 0.0 {
        if macd > signal * 1.1 {
            TrendDirection::StrongBullish
        } else {
            TrendDirection::Bullish
        }
    } else if macd < 0.0 && histogram < 0.0 {
        if macd < signal * 1.1 {
            TrendDirection::StrongBearish
        } else {
            TrendDirection::Bearish
        }
    } else {
        TrendDirection::Neutral
    };
    
    // 检测金叉死叉
    let cross_signal = detect_macd_cross(&macd_values);
    
    MacdData {
        macd,
        signal,
        histogram,
        trend,
        cross_signal,
    }
}

/// 高级RSI计算（包含背离检测）
pub fn calculate_rsi_advanced(prices: &[f64]) -> RsiData {
    if prices.len() < 14 {
        return RsiData {
            value: 50.0,
            signal: RsiSignal::Neutral,
            divergence: None,
        };
    }
    
    let rsi_value = calculate_rsi_wilder(prices, 14);
    
    // 判断信号
    let signal = if rsi_value > 80.0 {
        RsiSignal::ExtemeOverbought
    } else if rsi_value > 70.0 {
        RsiSignal::Overbought
    } else if rsi_value < 20.0 {
        RsiSignal::ExtremeOversold
    } else if rsi_value < 30.0 {
        RsiSignal::Oversold
    } else {
        RsiSignal::Neutral
    };
    
    // 检测背离
    let divergence = detect_rsi_divergence(prices, rsi_value);
    
    RsiData {
        value: rsi_value,
        signal,
        divergence,
    }
}

/// 计算布林带
pub fn calculate_bollinger_bands(
    prices: &[f64],
    period: usize,
    std_multiplier: f64,
) -> BollingerBands {
    if prices.len() < period {
        let last_price = prices.last().copied().unwrap_or(100.0);
        return BollingerBands {
            upper: last_price * 1.02,
            middle: last_price,
            lower: last_price * 0.98,
            width: 0.04,
            position: 0.5,
            squeeze: false,
        };
    }
    
    // 计算移动平均（中轨）
    let middle = prices[prices.len()-period..]
        .iter()
        .sum::<f64>() / period as f64;
    
    // 计算标准差
    let variance = prices[prices.len()-period..]
        .iter()
        .map(|p| (p - middle).powi(2))
        .sum::<f64>() / period as f64;
    let std_dev = variance.sqrt();
    
    // 上下轨
    let upper = middle + std_multiplier * std_dev;
    let lower = middle - std_multiplier * std_dev;
    
    // 带宽
    let width = (upper - lower) / middle;
    
    // 当前价格位置
    let current_price = prices.last().copied().unwrap_or(middle);
    let position = if upper > lower {
        (current_price - lower) / (upper - lower)
    } else {
        0.5
    };
    
    // 检测布林带收缩（波动率降低）
    let historical_width = if prices.len() > period * 2 {
        let prev_middle = prices[prices.len()-period*2..prices.len()-period]
            .iter()
            .sum::<f64>() / period as f64;
        let prev_variance = prices[prices.len()-period*2..prices.len()-period]
            .iter()
            .map(|p| (p - prev_middle).powi(2))
            .sum::<f64>() / period as f64;
        let prev_std = prev_variance.sqrt();
        2.0 * std_multiplier * prev_std / prev_middle
    } else {
        width
    };
    
    let squeeze = width < historical_width * 0.7;
    
    BollingerBands {
        upper,
        middle,
        lower,
        width,
        position: position.clamp(0.0, 1.0),
        squeeze,
    }
}

/// 高级ATR计算（包含ADX）
pub fn calculate_atr_advanced(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) -> AtrData {
    let period = 14;
    
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return AtrData {
            value: 0.0,
            normalized: 0.02,
            volatility_level: VolatilityLevel::Medium,
            trend_strength: 0.0,
        };
    }
    
    // 计算真实波动幅度
    let mut true_ranges = Vec::new();
    for i in 1..highs.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i-1]).abs();
        let low_close = (lows[i] - closes[i-1]).abs();
        let tr = high_low.max(high_close).max(low_close);
        true_ranges.push(tr);
    }
    
    // ATR（使用Wilder平滑）
    let atr = if true_ranges.len() >= period {
        let initial_atr = true_ranges[..period].iter().sum::<f64>() / period as f64;
        let mut atr = initial_atr;
        
        for i in period..true_ranges.len() {
            atr = (atr * (period - 1) as f64 + true_ranges[i]) / period as f64;
        }
        atr
    } else {
        0.0
    };
    
    // 标准化ATR
    let last_price = closes.last().copied().unwrap_or(100.0);
    let normalized = atr / last_price;
    
    // 判断波动率级别
    let volatility_level = if normalized < 0.005 {
        VolatilityLevel::VeryLow
    } else if normalized < 0.01 {
        VolatilityLevel::Low
    } else if normalized < 0.02 {
        VolatilityLevel::Medium
    } else if normalized < 0.04 {
        VolatilityLevel::High
    } else {
        VolatilityLevel::Extreme
    };
    
    // 计算ADX（趋势强度）
    let adx = calculate_adx(highs, lows, closes, period);
    
    AtrData {
        value: atr,
        normalized,
        volatility_level,
        trend_strength: adx,
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 计算指数移动平均
pub fn calculate_ema(data: &[f64], period: usize) -> f64 {
    if data.is_empty() || period == 0 || data.len() < period {
        return data.last().copied().unwrap_or(0.0);
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[..period].iter().sum::<f64>() / period as f64;
    
    for i in period..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }
    
    ema
}

/// 计算MACD序列
fn calculate_macd_series(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 26 {
        return vec![];
    }
    
    let mut macd_values = Vec::new();
    
    for i in 26..=prices.len() {
        let subset = &prices[..i];
        let ema12 = calculate_ema(subset, 12);
        let ema26 = calculate_ema(subset, 26);
        macd_values.push(ema12 - ema26);
    }
    
    macd_values
}

/// 检测MACD金叉死叉
fn detect_macd_cross(macd_values: &[f64]) -> CrossSignal {
    if macd_values.len() < 2 {
        return CrossSignal::None;
    }
    
    let n = macd_values.len();
    let current = macd_values[n-1];
    let previous = macd_values[n-2];
    
    // 简化判断：MACD穿越零轴
    if previous <= 0.0 && current > 0.0 {
        CrossSignal::GoldenCross
    } else if previous >= 0.0 && current < 0.0 {
        CrossSignal::DeathCross
    } else {
        CrossSignal::None
    }
}

/// Wilder平滑RSI计算
fn calculate_rsi_wilder(prices: &[f64], period: usize) -> f64 {
    if prices.len() <= period {
        return 50.0;
    }
    
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    // Wilder平滑
    let mut avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss = losses[..period].iter().sum::<f64>() / period as f64;
    
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
    }
    
    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

/// 检测RSI背离
fn detect_rsi_divergence(prices: &[f64], current_rsi: f64) -> Option<DivergenceType> {
    if prices.len() < 20 {
        return None;
    }
    
    // 找出最近的高点和低点
    let n = prices.len();
    let recent_prices = &prices[n-20..];
    
    let max_price = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_price = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let current_price = prices[n-1];
    
    // 简化的背离检测
    if current_price >= max_price * 0.99 && current_rsi < 70.0 {
        // 价格接近新高但RSI没有超买
        Some(DivergenceType::BearishDivergence)
    } else if current_price <= min_price * 1.01 && current_rsi > 30.0 {
        // 价格接近新低但RSI没有超卖
        Some(DivergenceType::BullishDivergence)
    } else {
        None
    }
}

/// 计算ADX（平均趋向指数）
fn calculate_adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period + 1 {
        return 0.0;
    }
    
    let mut plus_dm = Vec::new();
    let mut minus_dm = Vec::new();
    let mut tr = Vec::new();
    
    for i in 1..highs.len() {
        // 计算方向移动
        let up_move = highs[i] - highs[i-1];
        let down_move = lows[i-1] - lows[i];
        
        if up_move > down_move && up_move > 0.0 {
            plus_dm.push(up_move);
            minus_dm.push(0.0);
        } else if down_move > up_move && down_move > 0.0 {
            plus_dm.push(0.0);
            minus_dm.push(down_move);
        } else {
            plus_dm.push(0.0);
            minus_dm.push(0.0);
        }
        
        // 真实波动幅度
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i-1]).abs();
        let low_close = (lows[i] - closes[i-1]).abs();
        tr.push(high_low.max(high_close).max(low_close));
    }
    
    // 平滑处理
    let smooth_plus = smooth_series(&plus_dm, period);
    let smooth_minus = smooth_series(&minus_dm, period);
    let smooth_tr = smooth_series(&tr, period);
    
    if smooth_tr == 0.0 {
        return 0.0;
    }
    
    // 计算DI
    let plus_di = 100.0 * smooth_plus / smooth_tr;
    let minus_di = 100.0 * smooth_minus / smooth_tr;
    
    // 计算DX
    let di_sum = plus_di + minus_di;
    if di_sum == 0.0 {
        return 0.0;
    }
    
    let dx = 100.0 * (plus_di - minus_di).abs() / di_sum;
    
    // ADX是DX的平滑值，这里简化返回DX
    dx.min(100.0)
}

/// 平滑序列（Wilder平滑）
fn smooth_series(data: &[f64], period: usize) -> f64 {
    if data.len() < period {
        return 0.0;
    }
    
    let mut smooth = data[..period].iter().sum::<f64>() / period as f64;
    
    for i in period..data.len() {
        smooth = (smooth * (period - 1) as f64 + data[i]) / period as f64;
    }
    
    smooth
}

/// 计算成交量比率
fn calculate_volume_ratio(volumes: &[i64]) -> f64 {
    if volumes.len() < 5 {
        return 1.0;
    }
    
    let n = volumes.len();
    let current_volume = volumes[n-1] as f64;
    let avg_volume = volumes[n-5..n-1]
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>() / 4.0;
    
    if avg_volume > 0.0 {
        current_volume / avg_volume
    } else {
        1.0
    }
}

/// 计算OBV趋势
fn calculate_obv_trend(prices: &[f64], volumes: &[i64]) -> TrendDirection {
    if prices.len() < 10 || volumes.len() < 10 {
        return TrendDirection::Neutral;
    }
    
    let mut obv = 0i64;
    let mut obv_values = Vec::new();
    
    for i in 1..prices.len().min(volumes.len()) {
        if prices[i] > prices[i-1] {
            obv += volumes[i];
        } else if prices[i] < prices[i-1] {
            obv -= volumes[i];
        }
        obv_values.push(obv);
    }
    
    if obv_values.len() < 5 {
        return TrendDirection::Neutral;
    }
    
    // 判断OBV趋势
    let recent_obv = obv_values[obv_values.len()-5..].iter().sum::<i64>() as f64 / 5.0;
    let prev_obv = obv_values[obv_values.len()-10..obv_values.len()-5]
        .iter()
        .sum::<i64>() as f64 / 5.0;
    
    let change_rate = if prev_obv != 0.0 {
        (recent_obv - prev_obv) / prev_obv.abs()
    } else {
        0.0
    };
    
    if change_rate > 0.1 {
        TrendDirection::Bullish
    } else if change_rate < -0.1 {
        TrendDirection::Bearish
    } else {
        TrendDirection::Neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ema_calculation() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5];
        let ema = calculate_ema(&prices, 3);
        assert!(ema > 100.0 && ema < 105.0);
    }
    
    #[test]
    fn test_rsi_calculation() {
        let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0];
        let rsi = calculate_rsi_advanced(&prices);
        assert!(rsi.value >= 0.0 && rsi.value <= 100.0);
    }
}
