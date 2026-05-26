//! 各指标背离检测

use super::checks::{
    check_hidden_bearish_divergence, check_hidden_bullish_divergence,
    check_regular_bearish_divergence, check_regular_bullish_divergence,
};
use super::extremes::{
    calculate_divergence_confidence, determine_divergence_strength,
    determine_macd_divergence_strength, find_local_extremes,
};
use super::{DivergenceSignal, DivergenceStrength, DivergenceType};
use crate::prediction::indicators::{macd, roc, rsi, williams};

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
                let price_change =
                    (latest_price_low.1 - prev_price_low.1) / prev_price_low.1 * 100.0;
                let rsi_change = latest_rsi_low.1 - prev_rsi_low.1;

                let strength = determine_divergence_strength(price_change.abs(), rsi_change.abs());
                let confidence = calculate_divergence_confidence(
                    price_change.abs(),
                    rsi_change.abs(),
                    latest_rsi_low.1,
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
                        price_change.abs(),
                        rsi_change
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
                let price_change =
                    (latest_price_high.1 - prev_price_high.1) / prev_price_high.1 * 100.0;
                let rsi_change = latest_rsi_high.1 - prev_rsi_high.1;

                let strength = determine_divergence_strength(price_change, rsi_change.abs());
                let confidence = calculate_divergence_confidence(
                    price_change,
                    rsi_change.abs(),
                    latest_rsi_high.1,
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
                        price_change,
                        rsi_change.abs()
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
