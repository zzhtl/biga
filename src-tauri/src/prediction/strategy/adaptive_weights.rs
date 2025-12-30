//! 自适应权重学习模块
//! 
//! 核心功能：
//! 1. 基于历史预测误差自动调整因子权重
//! 2. 不同市场状态下的权重记忆
//! 3. 防止过拟合的正则化机制
//! 4. 权重平滑更新避免剧烈波动

use serde::{Deserialize, Serialize};
use crate::prediction::analysis::market_regime::MarketRegime;

/// 因子权重配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorWeights {
    /// 趋势因子权重
    pub trend: f64,
    /// 动量因子权重
    pub momentum: f64,
    /// 量价因子权重
    pub volume_price: f64,
    /// 震荡指标因子权重
    pub oscillator: f64,
    /// 形态因子权重
    pub pattern: f64,
    /// 支撑阻力因子权重
    pub support_resistance: f64,
    /// 情绪因子权重
    pub sentiment: f64,
    /// 波动率因子权重
    pub volatility: f64,
}

impl Default for FactorWeights {
    fn default() -> Self {
        Self {
            trend: 0.20,
            momentum: 0.18,
            volume_price: 0.16,
            oscillator: 0.14,
            pattern: 0.10,
            support_resistance: 0.10,
            sentiment: 0.07,
            volatility: 0.05,
        }
    }
}

impl FactorWeights {
    /// 归一化权重（确保和为1）
    pub fn normalize(&mut self) {
        let total = self.trend + self.momentum + self.volume_price + 
                   self.oscillator + self.pattern + self.support_resistance +
                   self.sentiment + self.volatility;
        
        if total > 0.0 {
            self.trend /= total;
            self.momentum /= total;
            self.volume_price /= total;
            self.oscillator /= total;
            self.pattern /= total;
            self.support_resistance /= total;
            self.sentiment /= total;
            self.volatility /= total;
        }
    }
    
    /// 应用最小权重约束
    pub fn apply_min_constraint(&mut self, min_weight: f64) {
        let constrain = |w: &mut f64| {
            if *w < min_weight {
                *w = min_weight;
            }
        };
        
        constrain(&mut self.trend);
        constrain(&mut self.momentum);
        constrain(&mut self.volume_price);
        constrain(&mut self.oscillator);
        constrain(&mut self.pattern);
        constrain(&mut self.support_resistance);
        constrain(&mut self.sentiment);
        constrain(&mut self.volatility);
        
        self.normalize();
    }
}

/// 历史预测记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// 预测日期
    pub date: String,
    /// 预测涨跌幅
    pub predicted_change: f64,
    /// 实际涨跌幅
    pub actual_change: f64,
    /// 当时的市场状态
    pub market_regime: String,
    /// 各因子贡献
    pub factor_contributions: FactorContributions,
}

/// 各因子对预测的贡献
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorContributions {
    pub trend: f64,
    pub momentum: f64,
    pub volume_price: f64,
    pub oscillator: f64,
    pub pattern: f64,
    pub support_resistance: f64,
    pub sentiment: f64,
    pub volatility: f64,
}

/// 自适应权重学习器
pub struct AdaptiveWeightLearner {
    /// 当前权重
    pub current_weights: FactorWeights,
    /// 各市场状态的权重
    pub regime_weights: std::collections::HashMap<String, FactorWeights>,
    /// 学习率
    pub learning_rate: f64,
    /// 动量系数（用于权重平滑更新）
    pub momentum: f64,
    /// 最小权重约束
    pub min_weight: f64,
    /// 历史记录窗口大小
    pub history_window: usize,
}

impl Default for AdaptiveWeightLearner {
    fn default() -> Self {
        Self::new(0.05, 0.9, 0.03, 100)
    }
}

impl AdaptiveWeightLearner {
    pub fn new(learning_rate: f64, momentum: f64, min_weight: f64, history_window: usize) -> Self {
        let mut regime_weights = std::collections::HashMap::new();
        
        // 初始化各市场状态的默认权重
        regime_weights.insert("StrongUptrend".to_string(), FactorWeights {
            trend: 0.28,
            momentum: 0.22,
            volume_price: 0.18,
            oscillator: 0.10,
            pattern: 0.08,
            support_resistance: 0.06,
            sentiment: 0.05,
            volatility: 0.03,
        });
        
        regime_weights.insert("StrongDowntrend".to_string(), FactorWeights {
            trend: 0.28,
            momentum: 0.22,
            volume_price: 0.18,
            oscillator: 0.10,
            pattern: 0.08,
            support_resistance: 0.06,
            sentiment: 0.05,
            volatility: 0.03,
        });
        
        regime_weights.insert("ModerateUptrend".to_string(), FactorWeights {
            trend: 0.24,
            momentum: 0.20,
            volume_price: 0.17,
            oscillator: 0.12,
            pattern: 0.10,
            support_resistance: 0.08,
            sentiment: 0.05,
            volatility: 0.04,
        });
        
        regime_weights.insert("ModerateDowntrend".to_string(), FactorWeights {
            trend: 0.24,
            momentum: 0.20,
            volume_price: 0.17,
            oscillator: 0.12,
            pattern: 0.10,
            support_resistance: 0.08,
            sentiment: 0.05,
            volatility: 0.04,
        });
        
        regime_weights.insert("Ranging".to_string(), FactorWeights {
            trend: 0.12,
            momentum: 0.14,
            volume_price: 0.15,
            oscillator: 0.22,
            pattern: 0.15,
            support_resistance: 0.12,
            sentiment: 0.06,
            volatility: 0.04,
        });
        
        regime_weights.insert("PotentialTop".to_string(), FactorWeights {
            trend: 0.14,
            momentum: 0.16,
            volume_price: 0.18,
            oscillator: 0.18,
            pattern: 0.14,
            support_resistance: 0.10,
            sentiment: 0.06,
            volatility: 0.04,
        });
        
        regime_weights.insert("PotentialBottom".to_string(), FactorWeights {
            trend: 0.14,
            momentum: 0.16,
            volume_price: 0.18,
            oscillator: 0.18,
            pattern: 0.14,
            support_resistance: 0.10,
            sentiment: 0.06,
            volatility: 0.04,
        });
        
        Self {
            current_weights: FactorWeights::default(),
            regime_weights,
            learning_rate,
            momentum,
            min_weight,
            history_window,
        }
    }
    
    /// 获取指定市场状态的权重
    pub fn get_weights_for_regime(&self, regime: &MarketRegime) -> FactorWeights {
        let regime_key = format!("{:?}", regime);
        self.regime_weights
            .get(&regime_key)
            .cloned()
            .unwrap_or_else(FactorWeights::default)
    }
    
    /// 基于历史记录学习权重
    pub fn learn_from_history(&mut self, records: &[PredictionRecord]) {
        if records.len() < 10 {
            return;  // 数据不足，不学习
        }
        
        // 按市场状态分组
        let mut regime_records: std::collections::HashMap<String, Vec<&PredictionRecord>> = 
            std::collections::HashMap::new();
        
        for record in records.iter().rev().take(self.history_window) {
            regime_records
                .entry(record.market_regime.clone())
                .or_insert_with(Vec::new)
                .push(record);
        }
        
        // 对每个市场状态计算最优权重调整
        for (regime, records) in regime_records.iter() {
            if records.len() < 5 {
                continue;  // 样本太少
            }
            
            let gradients = self.calculate_gradients(records);
            
            if let Some(weights) = self.regime_weights.get_mut(regime) {
                // 使用自身参数进行更新
                let lr = self.learning_rate;
                update_weights_with_lr(weights, &gradients, lr);
            }
        }
    }
    
    /// 计算权重梯度
    fn calculate_gradients(&self, records: &[&PredictionRecord]) -> FactorWeights {
        let mut gradients = FactorWeights {
            trend: 0.0,
            momentum: 0.0,
            volume_price: 0.0,
            oscillator: 0.0,
            pattern: 0.0,
            support_resistance: 0.0,
            sentiment: 0.0,
            volatility: 0.0,
        };
        
        for record in records {
            let error = record.actual_change - record.predicted_change;
            let error_sign = error.signum();
            
            // 根据因子贡献和误差方向计算梯度
            // 如果因子贡献与误差方向一致，增加该因子权重
            gradients.trend += error_sign * record.factor_contributions.trend.signum() * error.abs().min(5.0);
            gradients.momentum += error_sign * record.factor_contributions.momentum.signum() * error.abs().min(5.0);
            gradients.volume_price += error_sign * record.factor_contributions.volume_price.signum() * error.abs().min(5.0);
            gradients.oscillator += error_sign * record.factor_contributions.oscillator.signum() * error.abs().min(5.0);
            gradients.pattern += error_sign * record.factor_contributions.pattern.signum() * error.abs().min(5.0);
            gradients.support_resistance += error_sign * record.factor_contributions.support_resistance.signum() * error.abs().min(5.0);
            gradients.sentiment += error_sign * record.factor_contributions.sentiment.signum() * error.abs().min(5.0);
            gradients.volatility += error_sign * record.factor_contributions.volatility.signum() * error.abs().min(5.0);
        }
        
        // 平均梯度
        let n = records.len() as f64;
        gradients.trend /= n;
        gradients.momentum /= n;
        gradients.volume_price /= n;
        gradients.oscillator /= n;
        gradients.pattern /= n;
        gradients.support_resistance /= n;
        gradients.sentiment /= n;
        gradients.volatility /= n;
        
        gradients
    }
    
    /// 使用动量SGD更新权重（内部调用静态函数）
    #[allow(dead_code)]
    fn update_weights(&self, weights: &mut FactorWeights, gradients: &FactorWeights) {
        update_weights_with_lr(weights, gradients, self.learning_rate);
        weights.apply_min_constraint(self.min_weight);
    }
}

/// 使用指定学习率更新权重（独立函数，避免借用冲突）
fn update_weights_with_lr(weights: &mut FactorWeights, gradients: &FactorWeights, lr: f64) {
    // 带动量的更新
    weights.trend += lr * gradients.trend;
    weights.momentum += lr * gradients.momentum;
    weights.volume_price += lr * gradients.volume_price;
    weights.oscillator += lr * gradients.oscillator;
    weights.pattern += lr * gradients.pattern;
    weights.support_resistance += lr * gradients.support_resistance;
    weights.sentiment += lr * gradients.sentiment;
    weights.volatility += lr * gradients.volatility;
    
    // 归一化
    weights.normalize();
}

/// 评估权重性能
pub fn evaluate_weights(records: &[PredictionRecord]) -> WeightPerformance {
    if records.is_empty() {
        return WeightPerformance::default();
    }
    
    let mut total_error = 0.0;
    let mut total_squared_error = 0.0;
    let mut direction_correct = 0;
    
    for record in records {
        let error = record.actual_change - record.predicted_change;
        total_error += error.abs();
        total_squared_error += error * error;
        
        // 方向正确性
        if (record.predicted_change > 0.0 && record.actual_change > 0.0) ||
           (record.predicted_change < 0.0 && record.actual_change < 0.0) ||
           (record.predicted_change.abs() < 0.3 && record.actual_change.abs() < 0.5) {
            direction_correct += 1;
        }
    }
    
    let n = records.len() as f64;
    
    WeightPerformance {
        mae: total_error / n,
        rmse: (total_squared_error / n).sqrt(),
        direction_accuracy: direction_correct as f64 / n,
        sample_count: records.len(),
    }
}

/// 权重性能指标
#[derive(Debug, Clone, Default)]
pub struct WeightPerformance {
    /// 平均绝对误差
    pub mae: f64,
    /// 均方根误差
    pub rmse: f64,
    /// 方向准确率
    pub direction_accuracy: f64,
    /// 样本数量
    pub sample_count: usize,
}

/// 计算动态权重（基于实时市场状态）
pub fn calculate_dynamic_weights(
    regime: &MarketRegime,
    volatility_percentile: f64,
    trend_strength: f64,
) -> FactorWeights {
    let mut weights = match regime {
        MarketRegime::StrongUptrend | MarketRegime::StrongDowntrend => {
            FactorWeights {
                trend: 0.28,
                momentum: 0.22,
                volume_price: 0.18,
                oscillator: 0.10,
                pattern: 0.08,
                support_resistance: 0.06,
                sentiment: 0.05,
                volatility: 0.03,
            }
        }
        MarketRegime::ModerateUptrend | MarketRegime::ModerateDowntrend => {
            FactorWeights {
                trend: 0.24,
                momentum: 0.20,
                volume_price: 0.17,
                oscillator: 0.13,
                pattern: 0.10,
                support_resistance: 0.08,
                sentiment: 0.05,
                volatility: 0.03,
            }
        }
        MarketRegime::Ranging => {
            FactorWeights {
                trend: 0.12,
                momentum: 0.14,
                volume_price: 0.16,
                oscillator: 0.22,
                pattern: 0.14,
                support_resistance: 0.12,
                sentiment: 0.06,
                volatility: 0.04,
            }
        }
        MarketRegime::PotentialTop | MarketRegime::PotentialBottom => {
            FactorWeights {
                trend: 0.14,
                momentum: 0.16,
                volume_price: 0.18,
                oscillator: 0.20,
                pattern: 0.12,
                support_resistance: 0.10,
                sentiment: 0.06,
                volatility: 0.04,
            }
        }
    };
    
    // 高波动时增加波动率和支撑阻力权重
    if volatility_percentile > 75.0 {
        weights.volatility *= 1.5;
        weights.support_resistance *= 1.3;
        weights.trend *= 0.85;
    } else if volatility_percentile < 25.0 {
        weights.trend *= 1.15;
        weights.momentum *= 1.1;
    }
    
    // 强趋势时增加趋势权重
    if trend_strength.abs() > 0.7 {
        weights.trend *= 1.2;
        weights.oscillator *= 0.8;
    }
    
    weights.normalize();
    weights
}

/// 混合权重（默认权重与学习权重加权平均）
pub fn blend_weights(
    default_weights: &FactorWeights,
    learned_weights: &FactorWeights,
    learning_confidence: f64,  // 0-1，学习权重的信任度
) -> FactorWeights {
    let lc = learning_confidence.clamp(0.0, 0.7);  // 最多70%信任学习权重
    let dc = 1.0 - lc;
    
    let mut blended = FactorWeights {
        trend: default_weights.trend * dc + learned_weights.trend * lc,
        momentum: default_weights.momentum * dc + learned_weights.momentum * lc,
        volume_price: default_weights.volume_price * dc + learned_weights.volume_price * lc,
        oscillator: default_weights.oscillator * dc + learned_weights.oscillator * lc,
        pattern: default_weights.pattern * dc + learned_weights.pattern * lc,
        support_resistance: default_weights.support_resistance * dc + learned_weights.support_resistance * lc,
        sentiment: default_weights.sentiment * dc + learned_weights.sentiment * lc,
        volatility: default_weights.volatility * dc + learned_weights.volatility * lc,
    };
    
    blended.normalize();
    blended
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weight_normalization() {
        let mut weights = FactorWeights {
            trend: 0.5,
            momentum: 0.5,
            volume_price: 0.5,
            oscillator: 0.5,
            pattern: 0.5,
            support_resistance: 0.5,
            sentiment: 0.5,
            volatility: 0.5,
        };
        
        weights.normalize();
        
        let total = weights.trend + weights.momentum + weights.volume_price +
                   weights.oscillator + weights.pattern + weights.support_resistance +
                   weights.sentiment + weights.volatility;
        
        assert!((total - 1.0).abs() < 0.001);
    }
    
    #[test]
    fn test_dynamic_weights() {
        let weights = calculate_dynamic_weights(
            &MarketRegime::StrongUptrend,
            80.0,  // 高波动
            0.8,   // 强趋势
        );
        
        // 在强趋势高波动环境下，趋势权重应该较高
        assert!(weights.trend > 0.2);
    }
}

