//! 趋势分析模块

use crate::prediction::indicators::{macd, kdj};
use serde::{Deserialize, Serialize};

/// 趋势状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrendState {
    StrongBullish,
    Bullish,
    Neutral,
    Bearish,
    StrongBearish,
}

impl TrendState {
    pub fn to_string(&self) -> String {
        match self {
            Self::StrongBullish => "强烈上涨".to_string(),
            Self::Bullish => "上涨".to_string(),
            Self::Neutral => "震荡".to_string(),
            Self::Bearish => "下跌".to_string(),
            Self::StrongBearish => "强烈下跌".to_string(),
        }
    }
    
    pub fn is_bullish(&self) -> bool {
        matches!(self, Self::StrongBullish | Self::Bullish)
    }
    
    pub fn is_bearish(&self) -> bool {
        matches!(self, Self::StrongBearish | Self::Bearish)
    }
}

/// 趋势分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub daily_trend: TrendState,
    pub weekly_trend: TrendState,
    pub overall_trend: TrendState,
    pub trend_strength: f64,
    pub trend_confidence: f64,
    pub bias_multiplier: f64,
    pub description: String,
}

/// 分析股票趋势
pub fn analyze_trend(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
) -> TrendAnalysis {
    let len = prices.len();
    
    if len < 120 {
        return TrendAnalysis {
            daily_trend: TrendState::Neutral,
            weekly_trend: TrendState::Neutral,
            overall_trend: TrendState::Neutral,
            trend_strength: 0.0,
            trend_confidence: 0.3,
            bias_multiplier: 1.0,
            description: "数据不足，无法准确判断趋势".to_string(),
        };
    }
    
    // 日线级别分析
    let daily_trend = analyze_daily_trend(prices, highs, lows);
    
    // 周线级别分析（模拟）
    let weekly_trend = analyze_weekly_trend(prices, highs, lows);
    
    // 综合趋势判断
    let overall_trend = combine_trends(&daily_trend, &weekly_trend);
    
    // 计算趋势强度
    let trend_strength = calculate_trend_strength(&overall_trend, &daily_trend, &weekly_trend);
    
    // 计算置信度
    let trend_confidence = calculate_confidence(&daily_trend, &weekly_trend);
    
    // 计算偏向乘数
    let bias_multiplier = calculate_bias_multiplier(&daily_trend, &weekly_trend);
    
    // 生成描述
    let description = generate_description(&daily_trend, &weekly_trend, trend_confidence);
    
    TrendAnalysis {
        daily_trend,
        weekly_trend,
        overall_trend,
        trend_strength,
        trend_confidence,
        bias_multiplier,
        description,
    }
}

/// 分析日线趋势
fn analyze_daily_trend(prices: &[f64], highs: &[f64], lows: &[f64]) -> TrendState {
    let len = prices.len();
    let period = 30.min(len);
    
    let daily_data = &prices[len - period..];
    let daily_highs = &highs[len - period..];
    let daily_lows = &lows[len - period..];
    
    // 计算 MACD
    let (dif, dea, hist) = if daily_data.len() >= 26 {
        macd::calculate_macd_full(daily_data)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    // 计算 KDJ
    let (k, d, _j) = if daily_data.len() >= 9 {
        kdj::calculate_kdj(daily_highs, daily_lows, daily_data, 9)
    } else {
        (50.0, 50.0, 50.0)
    };
    
    // 判断趋势
    let macd_bullish = dif > dea && hist > 0.0;
    let macd_bearish = dif < dea && hist < 0.0;
    let kdj_bullish = k > d;
    let kdj_bearish = k < d;
    
    if macd_bullish && kdj_bullish && hist > 0.0 {
        TrendState::StrongBullish
    } else if macd_bullish || kdj_bullish {
        TrendState::Bullish
    } else if macd_bearish && kdj_bearish && hist < 0.0 {
        TrendState::StrongBearish
    } else if macd_bearish || kdj_bearish {
        TrendState::Bearish
    } else {
        TrendState::Neutral
    }
}

/// 分析周线趋势（模拟）
fn analyze_weekly_trend(prices: &[f64], highs: &[f64], lows: &[f64]) -> TrendState {
    let len = prices.len();
    let period = 120.min(len);
    let weekly_step = 5;
    
    let mut weekly_prices = Vec::new();
    let mut weekly_highs = Vec::new();
    let mut weekly_lows = Vec::new();
    
    for i in (weekly_step..=period).step_by(weekly_step) {
        let start_idx = len - period + i - weekly_step;
        let end_idx = len - period + i - 1;
        
        if end_idx < len {
            weekly_prices.push(prices[end_idx]);
            weekly_highs.push(highs[start_idx..=end_idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            weekly_lows.push(lows[start_idx..=end_idx].iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        }
    }
    
    if weekly_prices.len() < 5 {
        return TrendState::Neutral;
    }
    
    // 计算周线 MACD
    let (dif, dea, hist) = macd::calculate_macd_full(&weekly_prices);
    
    // 简化的周线趋势判断
    if dif > dea && hist > 0.0 {
        TrendState::Bullish
    } else if dif < dea && hist < 0.0 {
        TrendState::Bearish
    } else {
        TrendState::Neutral
    }
}

/// 综合趋势
fn combine_trends(daily: &TrendState, weekly: &TrendState) -> TrendState {
    match (daily, weekly) {
        (TrendState::StrongBullish, TrendState::Bullish | TrendState::StrongBullish) => TrendState::StrongBullish,
        (TrendState::StrongBearish, TrendState::Bearish | TrendState::StrongBearish) => TrendState::StrongBearish,
        (TrendState::StrongBullish, _) => TrendState::Bullish,
        (TrendState::StrongBearish, _) => TrendState::Bearish,
        (TrendState::Bullish, _) => TrendState::Bullish,
        (TrendState::Bearish, _) => TrendState::Bearish,
        (TrendState::Neutral, TrendState::Bullish | TrendState::StrongBullish) => TrendState::Bullish,
        (TrendState::Neutral, TrendState::Bearish | TrendState::StrongBearish) => TrendState::Bearish,
        _ => TrendState::Neutral,
    }
}

/// 计算趋势强度
fn calculate_trend_strength(overall: &TrendState, daily: &TrendState, weekly: &TrendState) -> f64 {
    let base: f64 = match overall {
        TrendState::StrongBullish => 0.8,
        TrendState::Bullish => 0.5,
        TrendState::Neutral => 0.0,
        TrendState::Bearish => -0.5,
        TrendState::StrongBearish => -0.8,
    };
    
    let boost: f64 = match (daily, weekly) {
        (TrendState::StrongBullish, TrendState::StrongBullish) |
        (TrendState::StrongBearish, TrendState::StrongBearish) => 0.1,
        _ => 0.0,
    };
    
    (base + boost).clamp(-1.0, 1.0)
}

/// 计算置信度
fn calculate_confidence(daily: &TrendState, weekly: &TrendState) -> f64 {
    match (daily, weekly) {
        (TrendState::StrongBullish, TrendState::StrongBullish) |
        (TrendState::StrongBearish, TrendState::StrongBearish) => 0.95,
        (TrendState::StrongBullish, TrendState::Bullish) |
        (TrendState::StrongBearish, TrendState::Bearish) => 0.88,
        (TrendState::StrongBullish, _) |
        (TrendState::StrongBearish, _) => 0.75,
        (TrendState::Bullish, TrendState::Bullish) |
        (TrendState::Bearish, TrendState::Bearish) => 0.70,
        (TrendState::Bullish, _) |
        (TrendState::Bearish, _) => 0.60,
        (TrendState::Neutral, TrendState::Neutral) => 0.30,
        _ => 0.45,
    }
}

/// 计算偏向乘数
fn calculate_bias_multiplier(daily: &TrendState, weekly: &TrendState) -> f64 {
    match daily {
        TrendState::StrongBullish => match weekly {
            TrendState::StrongBullish | TrendState::Bullish => 1.9,
            TrendState::Neutral => 1.7,
            _ => 1.5,
        },
        TrendState::Bullish => match weekly {
            TrendState::StrongBullish | TrendState::Bullish => 1.5,
            TrendState::Neutral => 1.4,
            _ => 1.2,
        },
        TrendState::Neutral => match weekly {
            TrendState::StrongBullish => 1.2,
            TrendState::Bullish => 1.1,
            TrendState::Bearish => 0.9,
            TrendState::StrongBearish => 0.8,
            _ => 1.0,
        },
        TrendState::Bearish => match weekly {
            TrendState::StrongBearish | TrendState::Bearish => 0.5,
            TrendState::Neutral => 0.6,
            _ => 0.7,
        },
        TrendState::StrongBearish => match weekly {
            TrendState::StrongBearish | TrendState::Bearish => 0.2,
            TrendState::Neutral => 0.3,
            _ => 0.4,
        },
    }
}

/// 生成描述
fn generate_description(daily: &TrendState, weekly: &TrendState, confidence: f64) -> String {
    let trend_desc = match (daily, weekly) {
        (TrendState::StrongBullish, TrendState::StrongBullish) => "强烈上涨趋势 - 日线周线双确认",
        (TrendState::StrongBullish, _) => "短期强势上涨 - 日线强势",
        (TrendState::Bullish, _) => "上涨趋势 - 日线偏多",
        (TrendState::StrongBearish, TrendState::StrongBearish) => "强烈下跌趋势 - 日线周线双确认",
        (TrendState::StrongBearish, _) => "短期强势下跌 - 日线强势",
        (TrendState::Bearish, _) => "下跌趋势 - 日线偏空",
        (TrendState::Neutral, _) => "震荡趋势 - 方向不明",
    };
    
    format!("{} (置信度:{:.0}%)", trend_desc, confidence * 100.0)
}

/// 计算历史波动率
pub fn calculate_historical_volatility(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 0.02;
    }
    
    let mut daily_returns = Vec::new();
    let start = prices.len().saturating_sub(period);
    
    for i in (start + 1)..prices.len() {
        let return_rate = (prices[i] - prices[i - 1]) / prices[i - 1];
        daily_returns.push(return_rate);
    }
    
    if daily_returns.is_empty() {
        return 0.02;
    }
    
    let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
    let variance = daily_returns.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / daily_returns.len() as f64;
    
    variance.sqrt().min(0.10)
}

/// 计算近期趋势
pub fn calculate_recent_trend(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period {
        return 0.0;
    }
    
    let recent = &prices[prices.len() - period..];
    let n = period as f64;
    
    let sum_x: f64 = (0..period).map(|i| i as f64).sum();
    let sum_y: f64 = recent.iter().sum();
    let sum_xy: f64 = recent.iter().enumerate().map(|(i, &v)| i as f64 * v).sum();
    let sum_x2: f64 = (0..period).map(|i| (i * i) as f64).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let avg_price = sum_y / n;
    
    (slope / avg_price).clamp(-0.05, 0.05)
}

