//! 波动率计算、百分位与收敛检测

use super::VolatilityLevel;

/// 计算波动率
pub(super) fn calculate_volatility(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 0.02;
    }

    let start = prices.len() - period;
    let mut returns = Vec::new();

    for i in (start + 1)..prices.len() {
        let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
        returns.push(ret);
    }

    if returns.is_empty() {
        return 0.02;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}

/// 计算波动率百分位
pub(super) fn calculate_volatility_percentile(prices: &[f64], current_volatility: f64) -> f64 {
    let len = prices.len();
    if len < 120 {
        return 50.0;
    }

    // 计算历史波动率分布
    let mut historical_vols = Vec::new();
    for i in 60..len {
        let vol = calculate_volatility(&prices[..i], 20);
        historical_vols.push(vol);
    }

    if historical_vols.is_empty() {
        return 50.0;
    }

    // 计算百分位
    let below_count = historical_vols
        .iter()
        .filter(|&&v| v < current_volatility)
        .count();

    (below_count as f64 / historical_vols.len() as f64) * 100.0
}

/// 分类波动率水平
pub(super) fn classify_volatility_level(percentile: f64) -> VolatilityLevel {
    if percentile < 15.0 {
        VolatilityLevel::VeryLow
    } else if percentile < 35.0 {
        VolatilityLevel::Low
    } else if percentile < 65.0 {
        VolatilityLevel::Normal
    } else if percentile < 85.0 {
        VolatilityLevel::High
    } else {
        VolatilityLevel::VeryHigh
    }
}

/// 计算波动率收敛程度（用于预测突破）
pub(super) fn calculate_volatility_contraction(prices: &[f64], period: usize) -> f64 {
    let len = prices.len();
    if len < period * 2 {
        return 0.5;
    }

    let current_vol = calculate_volatility(prices, period);
    let prev_vol = calculate_volatility(&prices[..len - period], period);

    if prev_vol == 0.0 {
        return 0.5;
    }

    let contraction_ratio = 1.0 - (current_vol / prev_vol);
    contraction_ratio.clamp(0.0, 1.0)
}
