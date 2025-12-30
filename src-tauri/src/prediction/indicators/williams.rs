//! Williams %R 指标
//! 
//! 威廉指标是一个敏感的超买超卖指标，对于短期交易非常有效
//! - %R > -20：超买区域
//! - %R < -80：超卖区域
//! - 比RSI更敏感，适合捕捉短期转折点

/// 计算 Williams %R
/// 
/// 公式: %R = (最高价 - 收盘价) / (最高价 - 最低价) * -100
pub fn calculate_williams_r(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return -50.0; // 默认中性值
    }
    
    let len = highs.len();
    let start = len - period;
    
    let highest = highs[start..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = lows[start..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let close = closes[len - 1];
    
    let range = highest - lowest;
    if range == 0.0 {
        return -50.0;
    }
    
    ((highest - close) / range) * -100.0
}

/// 计算 Williams %R 序列
pub fn calculate_williams_r_series(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> Vec<f64> {
    let len = highs.len().min(lows.len()).min(closes.len());
    let mut result = Vec::with_capacity(len);
    
    for i in 0..len {
        if i + 1 < period {
            result.push(-50.0);
        } else {
            let h = &highs[..=i];
            let l = &lows[..=i];
            let c = &closes[..=i];
            result.push(calculate_williams_r(h, l, c, period));
        }
    }
    
    result
}

/// Williams %R 信号分析
#[derive(Debug, Clone)]
pub struct WilliamsSignal {
    pub value: f64,
    pub zone: WilliamsZone,
    pub is_divergence: bool,
    pub signal_strength: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WilliamsZone {
    Overbought,     // > -20
    NearOverbought, // -20 to -30
    Neutral,        // -30 to -70
    NearOversold,   // -70 to -80
    Oversold,       // < -80
}

impl WilliamsZone {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::Overbought => -0.8,
            Self::NearOverbought => -0.3,
            Self::Neutral => 0.0,
            Self::NearOversold => 0.3,
            Self::Oversold => 0.8,
        }
    }
}

/// 分析 Williams %R 信号
pub fn analyze_williams_signal(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: usize,
) -> WilliamsSignal {
    let wr = calculate_williams_r(highs, lows, closes, period);
    
    let zone = if wr > -20.0 {
        WilliamsZone::Overbought
    } else if wr > -30.0 {
        WilliamsZone::NearOverbought
    } else if wr > -70.0 {
        WilliamsZone::Neutral
    } else if wr > -80.0 {
        WilliamsZone::NearOversold
    } else {
        WilliamsZone::Oversold
    };
    
    // 检测简单背离（需要更多历史数据才能精确检测）
    let is_divergence = false; // 简化处理
    
    // 信号强度基于极端程度
    let signal_strength = if wr > -10.0 || wr < -90.0 {
        0.9
    } else if wr > -20.0 || wr < -80.0 {
        0.7
    } else if wr > -30.0 || wr < -70.0 {
        0.4
    } else {
        0.2
    };
    
    WilliamsSignal {
        value: wr,
        zone,
        is_divergence,
        signal_strength,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_williams_r() {
        let highs = vec![10.0, 11.0, 12.0, 11.5, 12.5];
        let lows = vec![9.0, 9.5, 10.0, 10.0, 10.5];
        let closes = vec![9.5, 10.5, 11.0, 11.0, 12.0];
        
        let wr = calculate_williams_r(&highs, &lows, &closes, 5);
        assert!(wr >= -100.0 && wr <= 0.0);
    }
}

