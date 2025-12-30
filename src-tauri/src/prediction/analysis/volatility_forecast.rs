//! 波动率预测模块
//! 
//! 基于GARCH思想的简化实现：
//! 1. 波动率聚集效应（Volatility Clustering）
//! 2. 均值回归特性
//! 3. 非对称效应（杠杆效应）
//! 
//! 用途：
//! - 预测未来波动率
//! - 调整预测置信区间
//! - 风险评估

use serde::{Deserialize, Serialize};

/// GARCH(1,1) 简化模型参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchParams {
    /// 长期平均波动率 (omega / (1 - alpha - beta))
    pub long_term_var: f64,
    /// ARCH系数 (对过去残差的反应)
    pub alpha: f64,
    /// GARCH系数 (对过去波动率的持续性)
    pub beta: f64,
    /// 非对称系数 (杠杆效应，负收益对波动率影响更大)
    pub gamma: f64,
}

impl Default for GarchParams {
    fn default() -> Self {
        Self {
            long_term_var: 0.0004,  // 约2%日波动率
            alpha: 0.10,
            beta: 0.85,
            gamma: 0.05,  // 轻微的杠杆效应
        }
    }
}

impl GarchParams {
    /// 检查参数有效性
    pub fn is_valid(&self) -> bool {
        self.alpha >= 0.0 && 
        self.beta >= 0.0 && 
        self.alpha + self.beta < 1.0 &&  // 确保平稳性
        self.long_term_var > 0.0
    }
    
    /// 计算无条件方差（长期平均方差）
    pub fn unconditional_variance(&self) -> f64 {
        let omega = self.long_term_var * (1.0 - self.alpha - self.beta);
        omega / (1.0 - self.alpha - self.beta)
    }
    
    /// 计算半衰期（波动率回到长期均值的一半所需天数）
    pub fn half_life(&self) -> f64 {
        let persistence = self.alpha + self.beta;
        if persistence >= 1.0 || persistence <= 0.0 {
            return 100.0;  // 默认大值
        }
        -persistence.ln().recip() * 0.693  // ln(2) ≈ 0.693
    }
}

/// 波动率预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityForecast {
    /// 当前条件波动率（日化）
    pub current_volatility: f64,
    /// 预测波动率序列
    pub forecast: Vec<f64>,
    /// 长期平均波动率
    pub long_term_volatility: f64,
    /// 波动率状态
    pub volatility_regime: VolatilityRegime,
    /// 预期回归时间（天）
    pub expected_reversion_days: f64,
    /// 波动率趋势
    pub volatility_trend: VolatilityTrend,
}

/// 波动率状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    VeryLow,   // 极低波动
    Low,       // 低波动
    Normal,    // 正常波动
    High,      // 高波动
    VeryHigh,  // 极高波动
    Extreme,   // 极端波动（可能的黑天鹅）
}

impl VolatilityRegime {
    pub fn from_percentile(percentile: f64) -> Self {
        if percentile < 10.0 {
            Self::VeryLow
        } else if percentile < 30.0 {
            Self::Low
        } else if percentile < 70.0 {
            Self::Normal
        } else if percentile < 90.0 {
            Self::High
        } else if percentile < 98.0 {
            Self::VeryHigh
        } else {
            Self::Extreme
        }
    }
    
    pub fn to_risk_multiplier(&self) -> f64 {
        match self {
            Self::VeryLow => 0.7,
            Self::Low => 0.85,
            Self::Normal => 1.0,
            Self::High => 1.3,
            Self::VeryHigh => 1.6,
            Self::Extreme => 2.0,
        }
    }
}

/// 波动率趋势
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityTrend {
    Expanding,   // 波动率扩张
    Contracting, // 波动率收缩
    Stable,      // 波动率稳定
}

/// 简化GARCH波动率预测器
pub struct GarchForecaster {
    params: GarchParams,
    /// 当前条件方差
    current_variance: f64,
    /// 历史收益率
    returns: Vec<f64>,
    /// 历史波动率序列
    volatility_history: Vec<f64>,
}

impl GarchForecaster {
    pub fn new(params: GarchParams) -> Self {
        Self {
            params,
            current_variance: 0.0004,  // 默认初始方差
            returns: Vec::new(),
            volatility_history: Vec::new(),
        }
    }
    
    /// 从价格数据初始化
    pub fn from_prices(prices: &[f64]) -> Self {
        let params = estimate_garch_params(prices);
        let mut forecaster = Self::new(params);
        
        // 计算收益率
        for i in 1..prices.len() {
            let ret = (prices[i] - prices[i - 1]) / prices[i - 1];
            forecaster.update(ret);
        }
        
        forecaster
    }
    
    /// 更新模型（新数据到来时）
    pub fn update(&mut self, return_value: f64) {
        self.returns.push(return_value);
        
        // 保持历史窗口大小
        if self.returns.len() > 500 {
            self.returns.remove(0);
        }
        
        // 更新条件方差 (GJR-GARCH风格，考虑非对称效应)
        let residual_sq = return_value * return_value;
        let asymmetric_term = if return_value < 0.0 {
            self.params.gamma * residual_sq
        } else {
            0.0
        };
        
        let omega = self.params.long_term_var * (1.0 - self.params.alpha - self.params.beta);
        
        self.current_variance = omega 
            + self.params.alpha * residual_sq 
            + asymmetric_term
            + self.params.beta * self.current_variance;
        
        // 确保方差为正
        self.current_variance = self.current_variance.max(0.000001);
        
        self.volatility_history.push(self.current_variance.sqrt());
        if self.volatility_history.len() > 500 {
            self.volatility_history.remove(0);
        }
    }
    
    /// 预测未来N天的波动率
    pub fn forecast(&self, days: usize) -> VolatilityForecast {
        let mut forecasts = Vec::with_capacity(days);
        let mut h = self.current_variance;
        
        let omega = self.params.long_term_var * (1.0 - self.params.alpha - self.params.beta);
        let persistence = self.params.alpha + self.params.beta;
        
        for _ in 0..days {
            // 预测波动率（标准差）
            forecasts.push(h.sqrt());
            
            // 递推预测方差
            // E[h_{t+k}] = omega + (alpha + beta)^k * (h_t - long_term_var) + long_term_var
            h = omega + persistence * h;
        }
        
        // 计算波动率状态
        let current_vol = self.current_variance.sqrt();
        let long_term_vol = self.params.long_term_var.sqrt();
        let percentile = self.calculate_volatility_percentile(current_vol);
        let volatility_regime = VolatilityRegime::from_percentile(percentile);
        
        // 计算预期回归时间
        let half_life = self.params.half_life();
        let deviation = (current_vol - long_term_vol).abs();
        let expected_reversion = if deviation > 0.001 {
            half_life * (deviation / long_term_vol).ln().abs()
        } else {
            0.0
        };
        
        // 判断波动率趋势
        let volatility_trend = self.determine_volatility_trend();
        
        VolatilityForecast {
            current_volatility: current_vol,
            forecast: forecasts,
            long_term_volatility: long_term_vol,
            volatility_regime,
            expected_reversion_days: expected_reversion.min(30.0),
            volatility_trend,
        }
    }
    
    /// 计算波动率百分位
    fn calculate_volatility_percentile(&self, current: f64) -> f64 {
        if self.volatility_history.is_empty() {
            return 50.0;
        }
        
        let count_below = self.volatility_history.iter()
            .filter(|&&v| v < current)
            .count();
        
        (count_below as f64 / self.volatility_history.len() as f64) * 100.0
    }
    
    /// 判断波动率趋势
    fn determine_volatility_trend(&self) -> VolatilityTrend {
        let len = self.volatility_history.len();
        if len < 10 {
            return VolatilityTrend::Stable;
        }
        
        // 比较近期和较早期的平均波动率
        let recent_avg: f64 = self.volatility_history[len - 5..].iter().sum::<f64>() / 5.0;
        let older_avg: f64 = self.volatility_history[len - 10..len - 5].iter().sum::<f64>() / 5.0;
        
        let change_ratio = (recent_avg - older_avg) / older_avg;
        
        if change_ratio > 0.15 {
            VolatilityTrend::Expanding
        } else if change_ratio < -0.15 {
            VolatilityTrend::Contracting
        } else {
            VolatilityTrend::Stable
        }
    }
}

/// 从历史数据估计GARCH参数
pub fn estimate_garch_params(prices: &[f64]) -> GarchParams {
    if prices.len() < 30 {
        return GarchParams::default();
    }
    
    // 计算收益率
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    // 计算样本方差
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let sample_variance = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    // 简化的参数估计（使用经验值和样本统计）
    // 实际应用中应使用最大似然估计
    
    // 长期方差
    let long_term_var = sample_variance;
    
    // 估计持续性（通过自相关）
    let mut acf1 = 0.0;
    let sq_returns: Vec<f64> = returns.iter().map(|r| r * r).collect();
    let sq_mean = sq_returns.iter().sum::<f64>() / sq_returns.len() as f64;
    
    if sq_returns.len() > 1 {
        let numerator: f64 = sq_returns.windows(2)
            .map(|w| (w[0] - sq_mean) * (w[1] - sq_mean))
            .sum();
        let denominator: f64 = sq_returns.iter()
            .map(|s| (s - sq_mean).powi(2))
            .sum();
        
        if denominator > 0.0 {
            acf1 = (numerator / denominator).clamp(0.0, 0.99);
        }
    }
    
    // 根据自相关估计参数
    let persistence = acf1.max(0.8).min(0.98);  // 通常波动率持续性较高
    
    // 分配到alpha和beta
    let alpha = 0.08_f64.min(persistence * 0.15);
    let beta = persistence - alpha;
    
    // 估计杠杆效应（负收益时波动率增加更多）
    let negative_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r * r)
        .collect();
    let positive_returns: Vec<f64> = returns.iter()
        .filter(|&&r| r >= 0.0)
        .map(|r| r * r)
        .collect();
    
    let gamma = if !negative_returns.is_empty() && !positive_returns.is_empty() {
        let neg_avg = negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
        let pos_avg = positive_returns.iter().sum::<f64>() / positive_returns.len() as f64;
        
        ((neg_avg / pos_avg - 1.0) * 0.1).clamp(0.0, 0.15)
    } else {
        0.05
    };
    
    GarchParams {
        long_term_var,
        alpha,
        beta,
        gamma,
    }
}

/// 快速波动率预测（不需要完整的GARCH模型）
pub fn quick_volatility_forecast(prices: &[f64], forecast_days: usize) -> Vec<f64> {
    if prices.len() < 20 {
        return vec![0.02; forecast_days];  // 默认2%日波动率
    }
    
    // 计算近期波动率
    let recent_vol = calculate_realized_volatility(&prices[prices.len() - 20..]);
    
    // 计算长期波动率
    let long_term_vol = calculate_realized_volatility(prices);
    
    // 简单的均值回归模型
    let decay_rate = 0.92;  // 每天向长期均值回归8%
    
    let mut forecasts = Vec::with_capacity(forecast_days);
    let mut current = recent_vol;
    
    for _ in 0..forecast_days {
        forecasts.push(current);
        current = decay_rate * current + (1.0 - decay_rate) * long_term_vol;
    }
    
    forecasts
}

/// 计算已实现波动率
pub fn calculate_realized_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.02;
    }
    
    let mut returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    variance.sqrt()
}

/// 波动率比率（当前vs长期）
pub fn calculate_volatility_ratio(prices: &[f64], short_period: usize, long_period: usize) -> f64 {
    let len = prices.len();
    
    if len < long_period {
        return 1.0;
    }
    
    let short_vol = calculate_realized_volatility(&prices[len.saturating_sub(short_period)..]);
    let long_vol = calculate_realized_volatility(&prices[len.saturating_sub(long_period)..]);
    
    if long_vol > 0.0001 {
        short_vol / long_vol
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_garch_params() {
        let params = GarchParams::default();
        assert!(params.is_valid());
        assert!(params.half_life() > 0.0);
    }
    
    #[test]
    fn test_volatility_forecast() {
        // 生成模拟价格
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01).sin() * 5.0).collect();
        
        let forecaster = GarchForecaster::from_prices(&prices);
        let forecast = forecaster.forecast(5);
        
        assert_eq!(forecast.forecast.len(), 5);
        assert!(forecast.current_volatility > 0.0);
    }
}

