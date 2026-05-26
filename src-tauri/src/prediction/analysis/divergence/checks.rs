//! 常规/隐藏背离判定

use super::extremes::{calculate_divergence_confidence, determine_divergence_strength};
use super::{DivergenceSignal, DivergenceStrength, DivergenceType};

/// 检测常规看涨背离
pub(super) fn check_regular_bullish_divergence(
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
        let confidence =
            calculate_divergence_confidence(price_change.abs(), ind_change.abs(), latest_ind.1);

        return Some(DivergenceSignal {
            divergence_type: DivergenceType::RegularBullish,
            indicator: indicator_name.to_string(),
            strength,
            confidence,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!(
                "{}底背离: 价格下跌{:.1}%但指标上升{:.1}",
                indicator_name,
                price_change.abs(),
                ind_change
            ),
        });
    }

    None
}

/// 检测常规看跌背离
pub(super) fn check_regular_bearish_divergence(
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
        let confidence =
            calculate_divergence_confidence(price_change, ind_change.abs(), latest_ind.1);

        return Some(DivergenceSignal {
            divergence_type: DivergenceType::RegularBearish,
            indicator: indicator_name.to_string(),
            strength,
            confidence,
            price_change,
            indicator_change: ind_change,
            duration_bars: (latest_price.0 - prev_price.0).max(1),
            description: format!(
                "{}顶背离: 价格上涨{:.1}%但指标下降{:.1}",
                indicator_name,
                price_change,
                ind_change.abs()
            ),
        });
    }

    None
}

/// 检测隐藏看涨背离（趋势延续信号）
pub(super) fn check_hidden_bullish_divergence(
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
pub(super) fn check_hidden_bearish_divergence(
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
