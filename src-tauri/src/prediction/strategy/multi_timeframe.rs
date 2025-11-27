//! 多周期分析策略

use crate::prediction::indicators::{macd, kdj};
use serde::{Deserialize, Serialize};

/// 多周期信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeSignal {
    pub date: String,
    pub daily_trend: String,
    pub weekly_trend: String,
    pub monthly_trend: String,
    pub resonance_level: i32,
    pub resonance_direction: String,
    pub signal_quality: f64,
    pub buy_signal: bool,
    pub sell_signal: bool,
}

/// 生成多周期信号
pub fn generate_multi_timeframe_signals(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    dates: &[String],
) -> Vec<MultiTimeframeSignal> {
    let mut signals = Vec::new();
    let len = prices.len();
    
    if len < 60 {
        return signals;
    }
    
    // 分析最近30天的信号
    for i in (len - 30)..len {
        if i < 60 {
            continue;
        }
        
        let signal = analyze_single_day(
            &prices[..=i],
            &highs[..=i],
            &lows[..=i],
            &dates[i],
        );
        
        signals.push(signal);
    }
    
    signals
}

/// 获取最新的多周期信号
pub fn get_latest_signal(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    date: &str,
) -> Option<MultiTimeframeSignal> {
    if prices.len() < 60 {
        return None;
    }
    
    Some(analyze_single_day(prices, highs, lows, date))
}

fn analyze_single_day(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    date: &str,
) -> MultiTimeframeSignal {
    let len = prices.len();
    
    // 日线分析
    let daily_data = &prices[len.saturating_sub(30)..];
    let daily_highs = &highs[len.saturating_sub(30)..];
    let daily_lows = &lows[len.saturating_sub(30)..];
    let daily_trend = analyze_timeframe_trend(daily_data, daily_highs, daily_lows);
    
    // 周线分析（模拟）
    let weekly_data = aggregate_to_weekly(prices, 12);
    let weekly_highs = aggregate_highs_weekly(highs, 12);
    let weekly_lows = aggregate_lows_weekly(lows, 12);
    let weekly_trend = analyze_timeframe_trend(&weekly_data, &weekly_highs, &weekly_lows);
    
    // 月线分析（模拟）
    let monthly_data = aggregate_to_monthly(prices, 6);
    let monthly_highs = aggregate_highs_monthly(highs, 6);
    let monthly_lows = aggregate_lows_monthly(lows, 6);
    let monthly_trend = analyze_timeframe_trend(&monthly_data, &monthly_highs, &monthly_lows);
    
    // 计算共振
    let (resonance_level, resonance_direction) = 
        calculate_resonance(&daily_trend, &weekly_trend, &monthly_trend);
    
    // 信号质量
    let signal_quality = calculate_signal_quality(resonance_level, &daily_trend);
    
    // 买卖信号
    let buy_signal = resonance_level >= 2 && resonance_direction == "看涨";
    let sell_signal = resonance_level >= 2 && resonance_direction == "看跌";
    
    MultiTimeframeSignal {
        date: date.to_string(),
        daily_trend,
        weekly_trend,
        monthly_trend,
        resonance_level,
        resonance_direction,
        signal_quality,
        buy_signal,
        sell_signal,
    }
}

fn analyze_timeframe_trend(prices: &[f64], highs: &[f64], lows: &[f64]) -> String {
    if prices.len() < 9 {
        return "中性".to_string();
    }
    
    // MACD 分析
    let (dif, dea, hist) = if prices.len() >= 26 {
        macd::calculate_macd_full(prices)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    // KDJ 分析
    let (k, d, _j) = kdj::calculate_kdj(highs, lows, prices, 9);
    
    let macd_bullish = dif > dea && hist > 0.0;
    let macd_bearish = dif < dea && hist < 0.0;
    let kdj_bullish = k > d;
    let kdj_bearish = k < d;
    
    if macd_bullish && kdj_bullish {
        "强势看涨".to_string()
    } else if macd_bullish || kdj_bullish {
        "看涨".to_string()
    } else if macd_bearish && kdj_bearish {
        "强势看跌".to_string()
    } else if macd_bearish || kdj_bearish {
        "看跌".to_string()
    } else {
        "中性".to_string()
    }
}

fn aggregate_to_weekly(prices: &[f64], weeks: usize) -> Vec<f64> {
    let step = 5;
    let mut result = Vec::new();
    let len = prices.len();
    
    for i in (step..=len.min(weeks * step)).step_by(step) {
        let idx = len.saturating_sub(i);
        if idx < len {
            result.push(prices[idx]);
        }
    }
    
    result.reverse();
    result
}

fn aggregate_highs_weekly(highs: &[f64], weeks: usize) -> Vec<f64> {
    let step = 5;
    let mut result = Vec::new();
    let len = highs.len();
    
    for i in (step..=len.min(weeks * step)).step_by(step) {
        let start = len.saturating_sub(i);
        let end = len.saturating_sub(i - step);
        if start < end && end <= len {
            let max = highs[start..end].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            result.push(max);
        }
    }
    
    result.reverse();
    result
}

fn aggregate_lows_weekly(lows: &[f64], weeks: usize) -> Vec<f64> {
    let step = 5;
    let mut result = Vec::new();
    let len = lows.len();
    
    for i in (step..=len.min(weeks * step)).step_by(step) {
        let start = len.saturating_sub(i);
        let end = len.saturating_sub(i - step);
        if start < end && end <= len {
            let min = lows[start..end].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            result.push(min);
        }
    }
    
    result.reverse();
    result
}

fn aggregate_to_monthly(prices: &[f64], months: usize) -> Vec<f64> {
    let step = 20;
    let mut result = Vec::new();
    let len = prices.len();
    
    for i in (step..=len.min(months * step)).step_by(step) {
        let idx = len.saturating_sub(i);
        if idx < len {
            result.push(prices[idx]);
        }
    }
    
    result.reverse();
    result
}

fn aggregate_highs_monthly(highs: &[f64], months: usize) -> Vec<f64> {
    let step = 20;
    let mut result = Vec::new();
    let len = highs.len();
    
    for i in (step..=len.min(months * step)).step_by(step) {
        let start = len.saturating_sub(i);
        let end = len.saturating_sub(i - step);
        if start < end && end <= len {
            let max = highs[start..end].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            result.push(max);
        }
    }
    
    result.reverse();
    result
}

fn aggregate_lows_monthly(lows: &[f64], months: usize) -> Vec<f64> {
    let step = 20;
    let mut result = Vec::new();
    let len = lows.len();
    
    for i in (step..=len.min(months * step)).step_by(step) {
        let start = len.saturating_sub(i);
        let end = len.saturating_sub(i - step);
        if start < end && end <= len {
            let min = lows[start..end].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            result.push(min);
        }
    }
    
    result.reverse();
    result
}

fn calculate_resonance(daily: &str, weekly: &str, monthly: &str) -> (i32, String) {
    let is_daily_bullish = daily.contains("涨");
    let is_weekly_bullish = weekly.contains("涨");
    let is_monthly_bullish = monthly.contains("涨");
    
    let is_daily_bearish = daily.contains("跌");
    let is_weekly_bearish = weekly.contains("跌");
    let is_monthly_bearish = monthly.contains("跌");
    
    let bullish_count = [is_daily_bullish, is_weekly_bullish, is_monthly_bullish]
        .iter().filter(|&&b| b).count();
    let bearish_count = [is_daily_bearish, is_weekly_bearish, is_monthly_bearish]
        .iter().filter(|&&b| b).count();
    
    if bullish_count >= 2 {
        (bullish_count as i32, "看涨".to_string())
    } else if bearish_count >= 2 {
        (bearish_count as i32, "看跌".to_string())
    } else {
        (0, "中性".to_string())
    }
}

fn calculate_signal_quality(resonance_level: i32, daily_trend: &str) -> f64 {
    let base: f64 = match resonance_level {
        3 => 90.0,
        2 => 70.0,
        1 => 50.0,
        _ => 30.0,
    };
    
    let boost: f64 = if daily_trend.contains("强势") { 5.0 } else { 0.0 };
    
    (base + boost).min(100.0)
}

