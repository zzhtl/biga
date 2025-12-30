//! VWAP (Volume Weighted Average Price) 成交量加权平均价格
//! 
//! VWAP是机构交易最重要的参考指标之一
//! - 价格 > VWAP：多头占优
//! - 价格 < VWAP：空头占优
//! - VWAP可作为动态支撑/阻力位

/// 计算VWAP
/// 
/// 公式: VWAP = ∑(典型价格 × 成交量) / ∑成交量
/// 典型价格 = (最高价 + 最低价 + 收盘价) / 3
pub fn calculate_vwap(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
) -> f64 {
    if highs.is_empty() || lows.is_empty() || closes.is_empty() || volumes.is_empty() {
        return 0.0;
    }
    
    let len = highs.len().min(lows.len()).min(closes.len()).min(volumes.len());
    let mut cumulative_tpv = 0.0;  // 累计(典型价格 × 成交量)
    let mut cumulative_volume = 0i64;
    
    for i in 0..len {
        let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
        cumulative_tpv += typical_price * volumes[i] as f64;
        cumulative_volume += volumes[i];
    }
    
    if cumulative_volume == 0 {
        return closes.last().copied().unwrap_or(0.0);
    }
    
    cumulative_tpv / cumulative_volume as f64
}

/// 计算滚动VWAP
pub fn calculate_rolling_vwap(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
    period: usize,
) -> f64 {
    let len = highs.len().min(lows.len()).min(closes.len()).min(volumes.len());
    if len < period {
        return calculate_vwap(highs, lows, closes, volumes);
    }
    
    let start = len - period;
    calculate_vwap(
        &highs[start..],
        &lows[start..],
        &closes[start..],
        &volumes[start..],
    )
}

/// 计算VWAP序列
pub fn calculate_vwap_series(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
) -> Vec<f64> {
    let len = highs.len().min(lows.len()).min(closes.len()).min(volumes.len());
    let mut result = Vec::with_capacity(len);
    
    let mut cumulative_tpv = 0.0;
    let mut cumulative_volume = 0i64;
    
    for i in 0..len {
        let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
        cumulative_tpv += typical_price * volumes[i] as f64;
        cumulative_volume += volumes[i];
        
        if cumulative_volume > 0 {
            result.push(cumulative_tpv / cumulative_volume as f64);
        } else {
            result.push(closes[i]);
        }
    }
    
    result
}

/// VWAP 标准差带
pub struct VwapBands {
    pub vwap: f64,
    pub upper_band_1: f64,  // +1标准差
    pub upper_band_2: f64,  // +2标准差
    pub lower_band_1: f64,  // -1标准差
    pub lower_band_2: f64,  // -2标准差
}

/// 计算VWAP及其标准差带
pub fn calculate_vwap_bands(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
    period: usize,
) -> VwapBands {
    let len = highs.len().min(lows.len()).min(closes.len()).min(volumes.len());
    let start = if len > period { len - period } else { 0 };
    
    let vwap = calculate_rolling_vwap(highs, lows, closes, volumes, period);
    
    // 计算标准差
    let mut sum_squared_diff = 0.0;
    let mut total_volume = 0i64;
    
    for i in start..len {
        let typical_price = (highs[i] + lows[i] + closes[i]) / 3.0;
        let diff = typical_price - vwap;
        sum_squared_diff += diff * diff * volumes[i] as f64;
        total_volume += volumes[i];
    }
    
    let std_dev = if total_volume > 0 {
        (sum_squared_diff / total_volume as f64).sqrt()
    } else {
        0.0
    };
    
    VwapBands {
        vwap,
        upper_band_1: vwap + std_dev,
        upper_band_2: vwap + 2.0 * std_dev,
        lower_band_1: vwap - std_dev,
        lower_band_2: vwap - 2.0 * std_dev,
    }
}

/// VWAP 信号分析
#[derive(Debug, Clone)]
pub struct VwapSignal {
    pub vwap: f64,
    pub current_price: f64,
    pub deviation: f64,           // 当前价格与VWAP的偏离率(%)
    pub position: VwapPosition,   // 当前价格相对位置
    pub signal_strength: f64,     // 信号强度 (0-1)
    pub support_resistance: f64,  // VWAP作为支撑/阻力的强度
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VwapPosition {
    FarAbove,     // 远高于VWAP (>2%)
    Above,        // 高于VWAP
    AtVwap,       // 接近VWAP (±0.5%)
    Below,        // 低于VWAP
    FarBelow,     // 远低于VWAP (<-2%)
}

impl VwapPosition {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::FarAbove => 0.8,
            Self::Above => 0.5,
            Self::AtVwap => 0.0,
            Self::Below => -0.5,
            Self::FarBelow => -0.8,
        }
    }
}

/// 分析VWAP信号
pub fn analyze_vwap_signal(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
    period: usize,
) -> VwapSignal {
    let vwap = calculate_rolling_vwap(highs, lows, closes, volumes, period);
    let current_price = *closes.last().unwrap_or(&0.0);
    
    let deviation = if vwap > 0.0 {
        (current_price - vwap) / vwap * 100.0
    } else {
        0.0
    };
    
    let position = if deviation > 2.0 {
        VwapPosition::FarAbove
    } else if deviation > 0.5 {
        VwapPosition::Above
    } else if deviation < -2.0 {
        VwapPosition::FarBelow
    } else if deviation < -0.5 {
        VwapPosition::Below
    } else {
        VwapPosition::AtVwap
    };
    
    // 信号强度基于偏离程度
    let signal_strength = (deviation.abs() / 3.0).min(1.0);
    
    // VWAP作为支撑/阻力的强度
    // 当价格接近VWAP时，支撑/阻力强度更高
    let support_resistance = (1.0 - deviation.abs() / 5.0).max(0.0);
    
    VwapSignal {
        vwap,
        current_price,
        deviation,
        position,
        signal_strength,
        support_resistance,
    }
}

/// 计算VWAP均值回归预期
/// 返回预期回归幅度(%)
pub fn calculate_vwap_mean_reversion(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    volumes: &[i64],
    period: usize,
) -> f64 {
    let signal = analyze_vwap_signal(highs, lows, closes, volumes, period);
    
    // 根据偏离程度计算回归力量
    // 偏离越大，回归力量越强
    let reversion_strength = if signal.deviation.abs() > 3.0 {
        0.7  // 强回归
    } else if signal.deviation.abs() > 2.0 {
        0.5  // 中等回归
    } else if signal.deviation.abs() > 1.0 {
        0.3  // 弱回归
    } else {
        0.1  // 微弱回归
    };
    
    // 返回预期回归幅度（负号表示向VWAP回归）
    -signal.deviation * reversion_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vwap() {
        let highs = vec![102.0, 104.0, 106.0, 105.0, 107.0];
        let lows = vec![99.0, 101.0, 103.0, 102.0, 104.0];
        let closes = vec![100.0, 103.0, 105.0, 104.0, 106.0];
        let volumes = vec![1000, 1500, 1200, 1800, 2000];
        
        let vwap = calculate_vwap(&highs, &lows, &closes, &volumes);
        assert!(vwap > 0.0);
        assert!(vwap >= lows.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        assert!(vwap <= highs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    }
}

