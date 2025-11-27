//! 量价分析模块

use serde::{Deserialize, Serialize};

/// 量价关系信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePriceSignal {
    pub direction: String,
    pub confidence: f64,
    pub change_range: (f64, f64),
    pub signal: String,
    pub price_trend: String,
    pub volume_trend: String,
    pub key_factors: Vec<String>,
}

/// 分析量价关系
pub fn analyze_volume_price(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
) -> VolumePriceSignal {
    let len = prices.len();
    
    if len < 10 {
        return VolumePriceSignal {
            direction: "横盘".to_string(),
            confidence: 0.3,
            change_range: (-1.0, 1.0),
            signal: "数据不足".to_string(),
            price_trend: "未知".to_string(),
            volume_trend: "未知".to_string(),
            key_factors: vec!["数据不足".to_string()],
        };
    }
    
    let current_price = *prices.last().unwrap();
    
    // 价格趋势分析
    let recent_5_avg = prices[len - 5..].iter().sum::<f64>() / 5.0;
    let recent_10_avg = prices[len - 10..].iter().sum::<f64>() / 10.0;
    let price_momentum_5d = (current_price - prices[len - 5]) / prices[len - 5] * 100.0;
    
    let price_trend = if current_price > recent_5_avg && recent_5_avg > recent_10_avg {
        if price_momentum_5d > 3.0 { "强势上涨" } else { "温和上涨" }
    } else if current_price < recent_5_avg && recent_5_avg < recent_10_avg {
        if price_momentum_5d < -3.0 { "强势下跌" } else { "温和下跌" }
    } else {
        "横盘震荡"
    };
    
    // 成交量趋势分析
    let recent_5_vol_avg = volumes[len - 5..].iter().sum::<i64>() as f64 / 5.0;
    let latest_volume = *volumes.last().unwrap() as f64;
    
    let volume_trend = if latest_volume > recent_5_vol_avg * 1.5 {
        "显著放量"
    } else if latest_volume > recent_5_vol_avg * 1.2 {
        "温和放量"
    } else if latest_volume < recent_5_vol_avg * 0.7 {
        "明显缩量"
    } else if latest_volume < recent_5_vol_avg * 0.8 {
        "温和缩量"
    } else {
        "量能平稳"
    };
    
    // 量价关系判断
    let mut bullish_score = 0;
    let mut bearish_score = 0;
    let mut key_factors = Vec::new();
    
    match (price_trend, volume_trend) {
        ("强势上涨", "显著放量") => {
            bullish_score += 5;
            key_factors.push("放量强势上涨".to_string());
        }
        ("强势上涨", "温和放量") => {
            bullish_score += 4;
            key_factors.push("放量上涨".to_string());
        }
        ("温和上涨", "显著放量") => {
            bullish_score += 4;
            key_factors.push("放量推升".to_string());
        }
        ("温和上涨", "温和放量") => {
            bullish_score += 3;
            key_factors.push("温和放量上涨".to_string());
        }
        ("强势上涨", "明显缩量") | ("温和上涨", "明显缩量") => {
            bearish_score += 1;
            key_factors.push("上涨无量警示".to_string());
        }
        ("强势下跌", "显著放量") => {
            bearish_score += 5;
            key_factors.push("放量大跌".to_string());
        }
        ("强势下跌", "温和放量") => {
            bearish_score += 4;
            key_factors.push("放量下跌".to_string());
        }
        ("温和下跌", "显著放量") => {
            bearish_score += 4;
            key_factors.push("放量打压".to_string());
        }
        ("强势下跌", "明显缩量") | ("温和下跌", "明显缩量") => {
            bullish_score += 2;
            key_factors.push("下跌缩量止跌".to_string());
        }
        ("横盘震荡", _) => {
            key_factors.push("横盘整理".to_string());
        }
        _ => {
            key_factors.push("量价关系复杂".to_string());
        }
    }
    
    // 技术位置确认
    let highest_10d = highs[len - 10..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest_10d = lows[len - 10..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let position_ratio = (current_price - lowest_10d) / (highest_10d - lowest_10d);
    
    if position_ratio > 0.8 {
        bearish_score += 1;
        key_factors.push("接近10日高位".to_string());
    } else if position_ratio < 0.2 {
        bullish_score += 1;
        key_factors.push("接近10日低位".to_string());
    }
    
    // 综合判断
    let (direction, confidence, change_range) = if bullish_score >= bearish_score + 3 {
        let conf = (0.7 + (bullish_score - bearish_score) as f64 * 0.05).min(0.95);
        ("上涨".to_string(), conf, (0.8, 6.0))
    } else if bearish_score >= bullish_score + 3 {
        let conf = (0.7 + (bearish_score - bullish_score) as f64 * 0.05).min(0.95);
        ("下跌".to_string(), conf, (-6.0, -0.8))
    } else if bullish_score > bearish_score {
        let conf = 0.55 + (bullish_score - bearish_score) as f64 * 0.03;
        ("上涨".to_string(), conf, (0.3, 3.5))
    } else if bearish_score > bullish_score {
        let conf = 0.55 + (bearish_score - bullish_score) as f64 * 0.03;
        ("下跌".to_string(), conf, (-3.5, -0.3))
    } else {
        ("横盘".to_string(), 0.5, (-2.0, 2.0))
    };
    
    let signal = match (&direction[..], bullish_score.max(bearish_score)) {
        ("上涨", score) if score >= 4 => "强烈买入".to_string(),
        ("上涨", score) if score >= 2 => "买入".to_string(),
        ("下跌", score) if score >= 4 => "强烈卖出".to_string(),
        ("下跌", score) if score >= 2 => "卖出".to_string(),
        _ => "持有".to_string(),
    };
    
    VolumePriceSignal {
        direction,
        confidence,
        change_range,
        signal,
        price_trend: price_trend.to_string(),
        volume_trend: volume_trend.to_string(),
        key_factors,
    }
}

/// 量价背离分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Divergence {
    pub has_bullish: bool,
    pub has_bearish: bool,
    pub strength: f64,
    pub message: String,
}

/// 检测量价背离
pub fn detect_divergence(
    prices: &[f64],
    volumes: &[i64],
    period: usize,
) -> Divergence {
    if prices.len() < period || volumes.len() < period {
        return Divergence {
            has_bullish: false,
            has_bearish: false,
            strength: 0.0,
            message: "数据不足".to_string(),
        };
    }
    
    let len = prices.len();
    let recent_prices = &prices[len - period..];
    let recent_volumes = &volumes[len - period..];
    
    // 计算价格和成交量的变化趋势
    let price_change = (recent_prices.last().unwrap() - recent_prices.first().unwrap()) 
        / recent_prices.first().unwrap() * 100.0;
    let vol_change = (*recent_volumes.last().unwrap() as f64 - *recent_volumes.first().unwrap() as f64) 
        / *recent_volumes.first().unwrap() as f64 * 100.0;
    
    // 底背离：价格下跌但成交量增加
    let has_bullish = price_change < -3.0 && vol_change > 20.0;
    
    // 顶背离：价格上涨但成交量减少
    let has_bearish = price_change > 3.0 && vol_change < -20.0;
    
    let strength = (price_change.abs() + vol_change.abs()) / 2.0;
    
    let message = if has_bullish {
        "检测到底背离信号，价格下跌但成交量增加，可能即将反弹".to_string()
    } else if has_bearish {
        "检测到顶背离信号，价格上涨但成交量减少，可能即将回调".to_string()
    } else {
        "未检测到明显背离信号".to_string()
    };
    
    Divergence {
        has_bullish,
        has_bearish,
        strength,
        message,
    }
}

