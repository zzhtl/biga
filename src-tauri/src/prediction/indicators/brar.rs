//! BRAR 人气意愿指标
//! 
//! BR (意愿指标): 反映市场买卖意愿强弱
//! AR (人气指标): 反映市场交易人气
//! 
//! 使用方法：
//! - BR > AR 且两者都上涨：市场看涨
//! - BR < AR：市场观望
//! - BR 和 AR 同时急速上升：市场过热
//! - BR 和 AR 同时低位：市场低迷，可能反弹

/// 计算 AR 人气指标
/// 
/// 公式: AR = ∑(High - Open) / ∑(Open - Low) * 100
pub fn calculate_ar(opens: &[f64], highs: &[f64], lows: &[f64], period: usize) -> f64 {
    if opens.len() < period || highs.len() < period || lows.len() < period {
        return 100.0; // 默认中性值
    }
    
    let len = opens.len();
    let start = len - period;
    
    let mut up_sum = 0.0;
    let mut down_sum = 0.0;
    
    for i in start..len {
        up_sum += highs[i] - opens[i];
        down_sum += opens[i] - lows[i];
    }
    
    if down_sum == 0.0 {
        return 200.0;
    }
    
    (up_sum / down_sum) * 100.0
}

/// 计算 BR 意愿指标
/// 
/// 公式: BR = ∑(High - PrevClose) / ∑(PrevClose - Low) * 100
/// 其中 PrevClose 为前一日收盘价
pub fn calculate_br(closes: &[f64], highs: &[f64], lows: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 || highs.len() < period || lows.len() < period {
        return 100.0;
    }
    
    let len = closes.len();
    let start = len - period;
    
    let mut up_sum = 0.0;
    let mut down_sum = 0.0;
    
    for i in start..len {
        let prev_close = closes[i - 1];
        up_sum += (highs[i] - prev_close).max(0.0);
        down_sum += (prev_close - lows[i]).max(0.0);
    }
    
    if down_sum == 0.0 {
        return 200.0;
    }
    
    (up_sum / down_sum) * 100.0
}

/// 计算 BRAR 指标
pub fn calculate_brar(
    opens: &[f64],
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    period: usize,
) -> (f64, f64) {
    let br = calculate_br(closes, highs, lows, period);
    let ar = calculate_ar(opens, highs, lows, period);
    (br, ar)
}

/// BRAR 信号分析
#[derive(Debug, Clone)]
pub struct BrarSignal {
    pub br: f64,
    pub ar: f64,
    pub market_status: MarketStatus,
    pub trend_signal: BrarTrendSignal,
    pub strength: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketStatus {
    Overheated,   // 过热 (BR > 400 或 AR > 180)
    Bullish,      // 看涨 (BR > AR, 都在上升)
    Neutral,      // 中性
    Bearish,      // 看跌 (BR < AR, 都在下降)
    Oversold,     // 超卖 (BR < 40 或 AR < 60)
}

impl MarketStatus {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::Overheated => -0.3,   // 过热容易回调
            Self::Bullish => 0.7,
            Self::Neutral => 0.0,
            Self::Bearish => -0.5,
            Self::Oversold => 0.5,      // 超卖可能反弹
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrarTrendSignal {
    BuySignal,    // 买入信号
    SellSignal,   // 卖出信号
    HoldSignal,   // 持有信号
}

/// 分析 BRAR 信号
pub fn analyze_brar_signal(
    opens: &[f64],
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    period: usize,
) -> BrarSignal {
    let (br, ar) = calculate_brar(opens, closes, highs, lows, period);
    
    // 市场状态判断
    let market_status = if br > 400.0 || ar > 180.0 {
        MarketStatus::Overheated
    } else if br < 40.0 || ar < 60.0 {
        MarketStatus::Oversold
    } else if br > ar && br > 100.0 {
        MarketStatus::Bullish
    } else if br < ar && br < 100.0 {
        MarketStatus::Bearish
    } else {
        MarketStatus::Neutral
    };
    
    // 趋势信号
    let trend_signal = match market_status {
        MarketStatus::Oversold => BrarTrendSignal::BuySignal,
        MarketStatus::Overheated => BrarTrendSignal::SellSignal,
        MarketStatus::Bullish if br > 150.0 => BrarTrendSignal::HoldSignal,
        MarketStatus::Bullish => BrarTrendSignal::BuySignal,
        MarketStatus::Bearish if br < 70.0 => BrarTrendSignal::BuySignal, // 反转
        MarketStatus::Bearish => BrarTrendSignal::SellSignal,
        _ => BrarTrendSignal::HoldSignal,
    };
    
    // 强度计算
    let br_deviation = (br - 100.0).abs() / 100.0;
    let ar_deviation = (ar - 100.0).abs() / 100.0;
    let strength = ((br_deviation + ar_deviation) / 2.0).min(1.0);
    
    BrarSignal {
        br,
        ar,
        market_status,
        trend_signal,
        strength,
    }
}

/// BR/AR 背离检测
pub fn detect_brar_divergence(
    br_series: &[f64],
    ar_series: &[f64],
    prices: &[f64],
) -> Option<BrarDivergence> {
    if br_series.len() < 10 || ar_series.len() < 10 || prices.len() < 10 {
        return None;
    }
    
    let len = prices.len();
    let recent_price_high = prices[len - 5..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let recent_price_low = prices[len - 5..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let prev_price_high = prices[len - 10..len - 5].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let prev_price_low = prices[len - 10..len - 5].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    let br_len = br_series.len();
    let recent_br = br_series[br_len - 5..].iter().sum::<f64>() / 5.0;
    let prev_br = br_series[br_len - 10..br_len - 5].iter().sum::<f64>() / 5.0;
    
    // 顶背离：价格新高，BR没新高
    if recent_price_high > prev_price_high && recent_br < prev_br {
        return Some(BrarDivergence::TopDivergence);
    }
    
    // 底背离：价格新低，BR没新低
    if recent_price_low < prev_price_low && recent_br > prev_br {
        return Some(BrarDivergence::BottomDivergence);
    }
    
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrarDivergence {
    TopDivergence,    // 顶背离，可能下跌
    BottomDivergence, // 底背离，可能上涨
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_brar() {
        let opens = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let closes = vec![101.0, 102.0, 103.0, 104.0, 105.0];
        let highs = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let lows = vec![99.0, 100.0, 101.0, 102.0, 103.0];
        
        let ar = calculate_ar(&opens, &highs, &lows, 4);
        assert!(ar > 0.0);
        
        let br = calculate_br(&closes, &highs, &lows, 4);
        assert!(br > 0.0);
    }
}

