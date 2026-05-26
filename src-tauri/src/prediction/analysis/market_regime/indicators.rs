//! 市场状态分类所用的指标：均线、ADX、动量

use crate::prediction::indicators::rsi;

/// 计算简单移动平均
pub(super) fn calculate_ma(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return prices.last().copied().unwrap_or(0.0);
    }
    let start = prices.len() - period;
    prices[start..].iter().sum::<f64>() / period as f64
}

/// 计算均线排列得分
pub(super) fn calculate_ma_alignment_score(
    current_price: f64,
    ma5: f64,
    ma10: f64,
    ma20: f64,
    ma60: f64,
) -> f64 {
    let mut score: f64 = 0.0;

    // 价格与MA5的关系
    if current_price > ma5 {
        score += 0.15;
    } else {
        score -= 0.15;
    }

    // MA5与MA10的关系
    if ma5 > ma10 {
        score += 0.20;
    } else {
        score -= 0.20;
    }

    // MA10与MA20的关系
    if ma10 > ma20 {
        score += 0.25;
    } else {
        score -= 0.25;
    }

    // MA20与MA60的关系
    if ma20 > ma60 {
        score += 0.30;
    } else {
        score -= 0.30;
    }

    // 完美多头/空头排列额外加分
    if current_price > ma5 && ma5 > ma10 && ma10 > ma20 && ma20 > ma60 {
        score += 0.10;
    } else if current_price < ma5 && ma5 < ma10 && ma10 < ma20 && ma20 < ma60 {
        score -= 0.10;
    }

    score.clamp(-1.0, 1.0)
}

/// 计算ADX（简化版本）
pub(super) fn calculate_adx(highs: &[f64], lows: &[f64], prices: &[f64], period: usize) -> f64 {
    if highs.len() < period + 1 || lows.len() < period + 1 || prices.len() < period + 1 {
        return 25.0; // 默认中性值
    }

    let len = prices.len();
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;

    for i in (len - period)..len {
        if i == 0 {
            continue;
        }

        let high_diff = highs[i] - highs[i - 1];
        let low_diff = lows[i - 1] - lows[i];

        // +DM
        if high_diff > low_diff && high_diff > 0.0 {
            plus_dm_sum += high_diff;
        }
        // -DM
        if low_diff > high_diff && low_diff > 0.0 {
            minus_dm_sum += low_diff;
        }

        // True Range
        let tr = (highs[i] - lows[i])
            .max((highs[i] - prices[i - 1]).abs())
            .max((lows[i] - prices[i - 1]).abs());
        tr_sum += tr;
    }

    if tr_sum == 0.0 {
        return 25.0;
    }

    let plus_di = (plus_dm_sum / tr_sum) * 100.0;
    let minus_di = (minus_dm_sum / tr_sum) * 100.0;

    let di_sum = plus_di + minus_di;
    if di_sum == 0.0 {
        return 25.0;
    }

    let dx = ((plus_di - minus_di).abs() / di_sum) * 100.0;

    // 返回DX作为ADX的近似值
    dx.clamp(0.0, 100.0)
}

/// 计算动量得分
pub(super) fn calculate_momentum_score(prices: &[f64]) -> f64 {
    let len = prices.len();
    if len < 20 {
        return 0.0;
    }

    let current = *prices.last().unwrap();

    // 短期动量（5日）
    let momentum_5d = (current - prices[len - 5]) / prices[len - 5];

    // 中期动量（10日）
    let momentum_10d = (current - prices[len - 10]) / prices[len - 10];

    // 长期动量（20日）
    let momentum_20d = (current - prices[len - 20]) / prices[len - 20];

    // RSI动量
    let rsi = rsi::calculate_rsi(&prices[len.saturating_sub(15)..]);
    let rsi_score = (rsi - 50.0) / 50.0;

    // 综合动量得分
    let score = momentum_5d * 0.30 + momentum_10d * 0.25 + momentum_20d * 0.20 + rsi_score * 0.25;

    score.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ma_alignment() {
        let score = calculate_ma_alignment_score(100.0, 99.0, 98.0, 97.0, 95.0);
        assert!(score > 0.8, "Perfect uptrend should have high score");

        let score = calculate_ma_alignment_score(100.0, 101.0, 102.0, 103.0, 105.0);
        assert!(score < -0.8, "Perfect downtrend should have low score");
    }
}
