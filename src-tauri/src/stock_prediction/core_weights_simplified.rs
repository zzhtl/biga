//! 简化的核心权重配置
//! 
//! 只保留最关键的权重参数，其他通过自适应算法动态调整

use std::collections::HashMap;

// ============================================================================
// 核心预测权重（自适应）
// ============================================================================

/// 权重管理器
pub struct WeightManager {
    weights: HashMap<String, f64>,
    market_state: MarketState,
}

/// 市场状态
#[derive(Debug, Clone, Copy)]
pub enum MarketState {
    Trending,
    Volatile,
    Stable,
}

impl WeightManager {
    /// 创建默认权重管理器
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        
        // 基础权重（总和为1.0）
        weights.insert("trend".to_string(), 0.30);
        weights.insert("momentum".to_string(), 0.25);
        weights.insert("volatility".to_string(), 0.15);
        weights.insert("volume".to_string(), 0.15);
        weights.insert("sentiment".to_string(), 0.15);
        
        Self {
            weights,
            market_state: MarketState::Stable,
        }
    }
    
    /// 根据市场状态动态调整权重
    pub fn adapt_to_market(&mut self, volatility: f64, trend_strength: f64) {
        // 识别市场状态
        self.market_state = if volatility > 0.03 {
            MarketState::Volatile
        } else if trend_strength > 0.7 {
            MarketState::Trending
        } else {
            MarketState::Stable
        };
        
        // 调整权重
        match self.market_state {
            MarketState::Trending => {
                self.weights.insert("trend".to_string(), 0.40);
                self.weights.insert("momentum".to_string(), 0.30);
                self.weights.insert("volatility".to_string(), 0.10);
                self.weights.insert("volume".to_string(), 0.10);
                self.weights.insert("sentiment".to_string(), 0.10);
            }
            MarketState::Volatile => {
                self.weights.insert("trend".to_string(), 0.20);
                self.weights.insert("momentum".to_string(), 0.20);
                self.weights.insert("volatility".to_string(), 0.30);
                self.weights.insert("volume".to_string(), 0.20);
                self.weights.insert("sentiment".to_string(), 0.10);
            }
            MarketState::Stable => {
                // 使用默认权重
                self.reset_to_default();
            }
        }
        
        // 归一化
        self.normalize_weights();
    }
    
    /// 获取权重
    pub fn get_weight(&self, key: &str) -> f64 {
        self.weights.get(key).copied().unwrap_or(0.0)
    }
    
    /// 重置为默认权重
    fn reset_to_default(&mut self) {
        self.weights.insert("trend".to_string(), 0.30);
        self.weights.insert("momentum".to_string(), 0.25);
        self.weights.insert("volatility".to_string(), 0.15);
        self.weights.insert("volume".to_string(), 0.15);
        self.weights.insert("sentiment".to_string(), 0.15);
    }
    
    /// 归一化权重
    fn normalize_weights(&mut self) {
        let sum: f64 = self.weights.values().sum();
        if sum > 0.0 {
            for weight in self.weights.values_mut() {
                *weight /= sum;
            }
        }
    }
}

// ============================================================================
// 固定参数（最小配置集）
// ============================================================================

/// 技术指标周期
pub mod periods {
    pub const MACD_FAST: usize = 12;
    pub const MACD_SLOW: usize = 26;
    pub const MACD_SIGNAL: usize = 9;
    pub const RSI: usize = 14;
    pub const BOLLINGER: usize = 20;
    pub const ATR: usize = 14;
}

/// 阈值参数
pub mod thresholds {
    pub const RSI_OVERBOUGHT: f64 = 70.0;
    pub const RSI_OVERSOLD: f64 = 30.0;
    pub const VOLATILITY_HIGH: f64 = 0.03;
    pub const VOLATILITY_LOW: f64 = 0.01;
    pub const TREND_STRONG: f64 = 0.7;
    pub const VOLUME_SURGE: f64 = 1.5;
}

/// 风险参数
pub mod risk {
    pub const MAX_POSITION_SIZE: f64 = 0.25;
    pub const MAX_DAILY_LOSS: f64 = 0.02;
    pub const STOP_LOSS_MULTIPLIER: f64 = 2.0;
    pub const RISK_FREE_RATE: f64 = 0.03;
}

// ============================================================================
// 动态参数计算
// ============================================================================

/// 根据市场条件计算动态参数
pub fn calculate_dynamic_params(
    volatility: f64,
    trend_strength: f64,
    volume_ratio: f64,
) -> DynamicParams {
    DynamicParams {
        prediction_confidence: calculate_confidence(volatility, trend_strength),
        position_size: calculate_position_size(volatility),
        stop_loss_distance: calculate_stop_loss(volatility),
        time_decay_factor: calculate_time_decay(volatility),
    }
}

#[derive(Debug, Clone)]
pub struct DynamicParams {
    pub prediction_confidence: f64,
    pub position_size: f64,
    pub stop_loss_distance: f64,
    pub time_decay_factor: f64,
}

fn calculate_confidence(volatility: f64, trend_strength: f64) -> f64 {
    let vol_factor = if volatility < 0.01 {
        1.2
    } else if volatility > 0.03 {
        0.8
    } else {
        1.0
    };
    
    let trend_factor = trend_strength.max(0.5);
    
    (vol_factor * trend_factor).min(1.0).max(0.3)
}

fn calculate_position_size(volatility: f64) -> f64 {
    if volatility > 0.04 {
        0.1  // 10% in extreme volatility
    } else if volatility > 0.02 {
        0.15  // 15% in high volatility
    } else {
        0.25  // 25% in normal conditions
    }
}

fn calculate_stop_loss(volatility: f64) -> f64 {
    // 基于波动率的止损距离
    (volatility * 2.0).max(0.02).min(0.05)
}

fn calculate_time_decay(volatility: f64) -> f64 {
    // 高波动时预测衰减更快
    if volatility > 0.03 {
        0.90
    } else if volatility < 0.01 {
        0.98
    } else {
        0.95
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weight_manager() {
        let mut wm = WeightManager::new();
        assert_eq!(wm.get_weight("trend"), 0.30);
        
        wm.adapt_to_market(0.04, 0.5);  // High volatility
        assert!(wm.get_weight("volatility") > 0.25);
        
        wm.adapt_to_market(0.01, 0.8);  // Strong trend
        assert!(wm.get_weight("trend") > 0.35);
    }
    
    #[test]
    fn test_dynamic_params() {
        let params = calculate_dynamic_params(0.02, 0.8, 1.2);
        assert!(params.prediction_confidence > 0.0);
        assert!(params.position_size <= 0.25);
    }
}
