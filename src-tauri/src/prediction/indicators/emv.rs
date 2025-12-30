//! EMV (Ease of Movement) 简易波动指标
//! 
//! EMV综合考虑价格变化和成交量，衡量价格移动的难易程度
//! - EMV > 0：上涨容易
//! - EMV < 0：下跌容易
//! - EMV 接近 0：价格移动困难

/// 计算 EMV 指标
pub fn calculate_emv(
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
    period: usize,
) -> f64 {
    if highs.len() < period + 1 || lows.len() < period + 1 || volumes.len() < period + 1 {
        return 0.0;
    }
    
    let len = highs.len();
    let mut emv_sum = 0.0;
    
    for i in (len - period)..len {
        if i == 0 {
            continue;
        }
        
        let high = highs[i];
        let low = lows[i];
        let prev_high = highs[i - 1];
        let prev_low = lows[i - 1];
        let volume = volumes[i] as f64;
        
        // 中点移动
        let mid_move = (high + low) / 2.0 - (prev_high + prev_low) / 2.0;
        
        // 箱体比率
        let box_ratio = if high != low && volume != 0.0 {
            (volume / 100_000_000.0) / (high - low)
        } else {
            0.0
        };
        
        // 单日EMV
        let emv = if box_ratio != 0.0 {
            mid_move / box_ratio
        } else {
            0.0
        };
        
        emv_sum += emv;
    }
    
    emv_sum / period as f64
}

/// 计算 EMV 序列
pub fn calculate_emv_series(
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
    period: usize,
) -> Vec<f64> {
    let len = highs.len().min(lows.len()).min(volumes.len());
    let mut result = Vec::with_capacity(len);
    
    for i in 0..len {
        if i + 1 < period {
            result.push(0.0);
        } else {
            let h = &highs[..=i];
            let l = &lows[..=i];
            let v = &volumes[..=i];
            result.push(calculate_emv(h, l, v, period));
        }
    }
    
    result
}

/// EMV 信号分析
#[derive(Debug, Clone)]
pub struct EmvSignal {
    pub value: f64,
    pub direction: EmvDirection,
    pub strength: f64,
    pub trend_quality: TrendQuality,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmvDirection {
    EasyUp,    // 上涨轻松
    DifficultUp, // 上涨困难
    Flat,      // 横盘
    DifficultDown, // 下跌困难
    EasyDown,  // 下跌轻松
}

impl EmvDirection {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::EasyUp => 0.8,
            Self::DifficultUp => 0.3,
            Self::Flat => 0.0,
            Self::DifficultDown => -0.3,
            Self::EasyDown => -0.8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendQuality {
    HighQuality,   // 趋势质量高（价格容易移动）
    Normal,        // 正常
    LowQuality,    // 趋势质量低（价格移动困难）
}

/// 分析 EMV 信号
pub fn analyze_emv_signal(
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
    period: usize,
) -> EmvSignal {
    let value = calculate_emv(highs, lows, volumes, period);
    
    // 方向判断
    let direction = if value > 0.5 {
        EmvDirection::EasyUp
    } else if value > 0.1 {
        EmvDirection::DifficultUp
    } else if value < -0.5 {
        EmvDirection::EasyDown
    } else if value < -0.1 {
        EmvDirection::DifficultDown
    } else {
        EmvDirection::Flat
    };
    
    // 强度计算
    let strength = (value.abs() * 2.0).min(1.0);
    
    // 趋势质量
    let trend_quality = if value.abs() > 0.3 {
        TrendQuality::HighQuality
    } else if value.abs() > 0.1 {
        TrendQuality::Normal
    } else {
        TrendQuality::LowQuality
    };
    
    EmvSignal {
        value,
        direction,
        strength,
        trend_quality,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_emv() {
        let highs = vec![10.0, 10.5, 11.0, 11.5, 12.0];
        let lows = vec![9.0, 9.5, 10.0, 10.5, 11.0];
        let volumes = vec![1000000, 1200000, 1100000, 1300000, 1400000];
        
        let emv = calculate_emv(&highs, &lows, &volumes, 4);
        // EMV 应该是有效值
        assert!(emv.is_finite());
    }
}

