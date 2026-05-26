//! 极值点查找与背离强度/置信度计算

use super::DivergenceStrength;

/// 寻找局部极值点
pub(super) fn find_local_extremes(
    data: &[f64],
    window: usize,
) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    let mut lows = Vec::new();
    let mut highs = Vec::new();

    if data.len() < window * 2 + 1 {
        return (lows, highs);
    }

    for i in window..(data.len() - window) {
        let current = data[i];

        // 检查是否为局部最低点
        let is_low = data[i.saturating_sub(window)..i].iter().all(|&x| x >= current)
            && data[(i + 1)..=(i + window).min(data.len() - 1)]
                .iter()
                .all(|&x| x >= current);

        // 检查是否为局部最高点
        let is_high = data[i.saturating_sub(window)..i].iter().all(|&x| x <= current)
            && data[(i + 1)..=(i + window).min(data.len() - 1)]
                .iter()
                .all(|&x| x <= current);

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
pub(super) fn determine_divergence_strength(price_change: f64, rsi_change: f64) -> DivergenceStrength {
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
pub(super) fn determine_macd_divergence_strength(
    price_change: f64,
    macd_change: f64,
) -> DivergenceStrength {
    if price_change > 5.0 && macd_change > 0.01 {
        DivergenceStrength::Strong
    } else if price_change > 2.5 || macd_change > 0.005 {
        DivergenceStrength::Moderate
    } else {
        DivergenceStrength::Weak
    }
}

/// 计算背离置信度
pub(super) fn calculate_divergence_confidence(
    price_change: f64,
    indicator_change: f64,
    indicator_value: f64,
) -> f64 {
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
