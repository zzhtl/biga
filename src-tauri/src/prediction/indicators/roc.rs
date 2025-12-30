//! ROC (Rate of Change) 变动率指标
//! 
//! ROC衡量当前价格与N日前价格的变化百分比
//! 是动量指标的一种，用于：
//! - 识别超买超卖
//! - 确认趋势强度
//! - 发现背离信号

/// 计算 ROC 指标
/// 
/// 公式: ROC = (当前收盘价 - N日前收盘价) / N日前收盘价 * 100
pub fn calculate_roc(prices: &[f64], period: usize) -> f64 {
    if prices.len() <= period {
        return 0.0;
    }
    
    let len = prices.len();
    let current = prices[len - 1];
    let past = prices[len - 1 - period];
    
    if past == 0.0 {
        return 0.0;
    }
    
    ((current - past) / past) * 100.0
}

/// 计算 ROC 序列
pub fn calculate_roc_series(prices: &[f64], period: usize) -> Vec<f64> {
    let len = prices.len();
    let mut result = Vec::with_capacity(len);
    
    for i in 0..len {
        if i < period {
            result.push(0.0);
        } else {
            let current = prices[i];
            let past = prices[i - period];
            if past == 0.0 {
                result.push(0.0);
            } else {
                result.push(((current - past) / past) * 100.0);
            }
        }
    }
    
    result
}

/// 计算平滑ROC (SROC)
/// 使用移动平均对ROC进行平滑处理，减少噪音
pub fn calculate_smoothed_roc(prices: &[f64], roc_period: usize, smooth_period: usize) -> f64 {
    let roc_series = calculate_roc_series(prices, roc_period);
    
    if roc_series.len() < smooth_period {
        return 0.0;
    }
    
    let start = roc_series.len() - smooth_period;
    roc_series[start..].iter().sum::<f64>() / smooth_period as f64
}

/// ROC 信号分析结果
#[derive(Debug, Clone)]
pub struct RocSignal {
    /// ROC值
    pub value: f64,
    /// 平滑ROC值
    pub smoothed_value: f64,
    /// 动量方向
    pub direction: RocDirection,
    /// 动量强度 (0-1)
    pub strength: f64,
    /// 是否处于极端区域
    pub is_extreme: bool,
    /// 与零轴的交叉
    pub zero_cross: ZeroCross,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RocDirection {
    StrongUp,   // ROC > 5
    Up,         // 0 < ROC <= 5
    Flat,       // ROC ≈ 0
    Down,       // -5 <= ROC < 0
    StrongDown, // ROC < -5
}

impl RocDirection {
    pub fn to_score(&self) -> f64 {
        match self {
            Self::StrongUp => 1.0,
            Self::Up => 0.5,
            Self::Flat => 0.0,
            Self::Down => -0.5,
            Self::StrongDown => -1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZeroCross {
    CrossUp,   // 向上穿越零轴
    CrossDown, // 向下穿越零轴
    None,      // 无交叉
}

/// 分析 ROC 信号
pub fn analyze_roc_signal(prices: &[f64], period: usize) -> RocSignal {
    let value = calculate_roc(prices, period);
    let smoothed_value = calculate_smoothed_roc(prices, period, 3);
    
    // 方向判断
    let direction = if value > 8.0 {
        RocDirection::StrongUp
    } else if value > 2.0 {
        RocDirection::Up
    } else if value < -8.0 {
        RocDirection::StrongDown
    } else if value < -2.0 {
        RocDirection::Down
    } else {
        RocDirection::Flat
    };
    
    // 强度计算
    let strength = (value.abs() / 15.0).min(1.0);
    
    // 极端区域判断
    let is_extreme = value.abs() > 12.0;
    
    // 零轴交叉检测
    let zero_cross = if prices.len() > period + 1 {
        let prev_roc = calculate_roc(&prices[..prices.len() - 1], period);
        if prev_roc <= 0.0 && value > 0.0 {
            ZeroCross::CrossUp
        } else if prev_roc >= 0.0 && value < 0.0 {
            ZeroCross::CrossDown
        } else {
            ZeroCross::None
        }
    } else {
        ZeroCross::None
    };
    
    RocSignal {
        value,
        smoothed_value,
        direction,
        strength,
        is_extreme,
        zero_cross,
    }
}

/// 多周期ROC综合分析
pub struct MultiPeriodRoc {
    pub short_roc: f64,  // 短期 (5日)
    pub medium_roc: f64, // 中期 (10日)
    pub long_roc: f64,   // 长期 (20日)
    pub consensus: f64,  // 综合得分 (-1 到 1)
}

pub fn analyze_multi_period_roc(prices: &[f64]) -> MultiPeriodRoc {
    let short_roc = calculate_roc(prices, 5);
    let medium_roc = calculate_roc(prices, 10);
    let long_roc = calculate_roc(prices, 20);
    
    // 综合得分（加权平均）
    let consensus = (short_roc * 0.4 + medium_roc * 0.35 + long_roc * 0.25) / 10.0;
    let consensus = consensus.clamp(-1.0, 1.0);
    
    MultiPeriodRoc {
        short_roc,
        medium_roc,
        long_roc,
        consensus,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_roc() {
        let prices: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        let roc = calculate_roc(&prices, 10);
        assert!(roc > 0.0, "Upward trend should have positive ROC");
    }
    
    #[test]
    fn test_roc_series() {
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let series = calculate_roc_series(&prices, 2);
        assert_eq!(series.len(), prices.len());
    }
}

