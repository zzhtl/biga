//! 转折点检测与市场状态判定

use super::{MarketRegime, VolatilityLevel};
use crate::prediction::indicators::{macd, rsi};

/// 检测潜在转折点
pub(super) fn detect_turning_points(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    ma_alignment: &f64,
    momentum: f64,
) -> (bool, bool) {
    let len = prices.len();
    if len < 30 {
        return (false, false);
    }

    let current = *prices.last().unwrap();

    // 计算RSI
    let rsi = rsi::calculate_rsi(&prices[len.saturating_sub(15)..]);

    // 计算MACD
    let (dif, dea, hist) = macd::calculate_macd_full(prices);
    let prev_hist = if len > 27 {
        let (_, _, h) = macd::calculate_macd_full(&prices[..len - 1]);
        h
    } else {
        hist
    };

    // 近期高低点
    let recent_high = highs[len.saturating_sub(10)..]
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let recent_low = lows[len.saturating_sub(10)..]
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));

    // 潜在顶部信号
    let is_potential_top = (rsi > 70.0 && momentum > 0.3 && *ma_alignment > 0.6) ||  // RSI超买 + 强势环境
        (current > recent_high * 0.98 && hist < prev_hist && hist > 0.0) ||  // 接近高点 + MACD柱缩小
        (rsi > 75.0 && dif < dea); // 极度超买 + MACD死叉

    // 潜在底部信号
    let is_potential_bottom = (rsi < 30.0 && momentum < -0.3 && *ma_alignment < -0.6) ||  // RSI超卖 + 弱势环境
        (current < recent_low * 1.02 && hist > prev_hist && hist < 0.0) ||  // 接近低点 + MACD柱缩小
        (rsi < 25.0 && dif > dea); // 极度超卖 + MACD金叉

    (is_potential_top, is_potential_bottom)
}

/// 综合判断市场状态
pub(super) fn determine_regime(
    ma_alignment: f64,
    adx: f64,
    momentum: f64,
    is_potential_top: bool,
    is_potential_bottom: bool,
    volatility_level: VolatilityLevel,
) -> (MarketRegime, f64) {
    // 优先检测转折点
    if is_potential_top && adx > 25.0 {
        return (MarketRegime::PotentialTop, 0.70);
    }
    if is_potential_bottom && adx > 25.0 {
        return (MarketRegime::PotentialBottom, 0.70);
    }

    // 趋势强度判断
    let is_strong_trend = adx > 35.0;
    let is_moderate_trend = adx > 20.0;

    // 方向判断
    let is_bullish = ma_alignment > 0.3 && momentum > 0.1;
    let is_strongly_bullish = ma_alignment > 0.6 && momentum > 0.2;
    let is_bearish = ma_alignment < -0.3 && momentum < -0.1;
    let is_strongly_bearish = ma_alignment < -0.6 && momentum < -0.2;

    // 状态分类
    let (regime, base_confidence) = if is_strongly_bullish && is_strong_trend {
        (MarketRegime::StrongUptrend, 0.85)
    } else if is_bullish && is_moderate_trend {
        (MarketRegime::ModerateUptrend, 0.75)
    } else if is_strongly_bearish && is_strong_trend {
        (MarketRegime::StrongDowntrend, 0.85)
    } else if is_bearish && is_moderate_trend {
        (MarketRegime::ModerateDowntrend, 0.75)
    } else {
        (MarketRegime::Ranging, 0.65)
    };

    // 根据波动率调整置信度
    let volatility_factor: f64 = match volatility_level {
        VolatilityLevel::VeryHigh => 0.85,
        VolatilityLevel::High => 0.90,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => 0.95, // 极低波动率可能预示突破
    };

    let confidence = (base_confidence * volatility_factor).clamp(0.3_f64, 0.95_f64);

    (regime, confidence)
}

/// 计算趋势强度值
pub(super) fn calculate_trend_strength_value(
    ma_alignment: f64,
    adx: f64,
    momentum: f64,
    regime: &MarketRegime,
) -> f64 {
    let direction = if regime.is_bullish() {
        1.0
    } else if regime.is_bearish() {
        -1.0
    } else {
        0.0
    };

    let strength = (ma_alignment.abs() * 0.40
        + (adx / 50.0).min(1.0) * 0.35
        + momentum.abs() * 0.25)
        .min(1.0);

    direction * strength
}

/// 生成状态描述
pub(super) fn generate_regime_description(
    regime: &MarketRegime,
    adx: f64,
    volatility: VolatilityLevel,
    ma_alignment: f64,
) -> String {
    let trend_desc = if adx > 35.0 {
        "强趋势"
    } else if adx > 20.0 {
        "中等趋势"
    } else {
        "弱趋势/震荡"
    };

    let ma_desc = if ma_alignment > 0.6 {
        "均线多头排列"
    } else if ma_alignment > 0.2 {
        "均线偏多"
    } else if ma_alignment < -0.6 {
        "均线空头排列"
    } else if ma_alignment < -0.2 {
        "均线偏空"
    } else {
        "均线粘合"
    };

    format!(
        "{} | {} | ADX={:.1} | 波动率:{} | {}",
        regime.to_string(),
        trend_desc,
        adx,
        volatility.to_string(),
        ma_desc
    )
}
