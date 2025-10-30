//! 金融级股票预测引擎
//! 
//! 集成专业金融模型，提供高准确率的股票价格预测
//! 核心特性：
//! - GARCH波动率模型
//! - 市场微结构分析
//! - 自适应权重系统
//! - 风险调整收益优化

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// 核心数据结构
// ============================================================================

/// 金融预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialPrediction {
    pub predicted_price: f64,
    pub predicted_return: f64,
    pub confidence: f64,
    pub volatility: f64,
    pub risk_adjusted_return: f64,
    pub market_regime: MarketRegime,
    pub trading_signal: TradingSignal,
    pub risk_metrics: RiskMetrics,
}

/// 市场状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Trending,       // 趋势市
    MeanReverting, // 均值回归
    Volatile,      // 高波动
    Stable,        // 稳定
}

/// 交易信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub action: TradeAction,
    pub strength: f64,
    pub optimal_position: f64,  // 凯利公式计算的最优仓位
    pub stop_loss: f64,
    pub take_profit: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

/// 风险指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub value_at_risk: f64,      // VaR
    pub expected_shortfall: f64,  // CVaR
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub beta: f64,
}

// ============================================================================
// GARCH波动率模型
// ============================================================================

/// GARCH(1,1)模型参数
pub struct GarchModel {
    omega: f64,  // 常数项
    alpha: f64,  // ARCH项系数
    beta: f64,   // GARCH项系数
}

impl GarchModel {
    /// 估计GARCH参数
    pub fn estimate(returns: &[f64]) -> Self {
        // 使用最大似然估计
        // 简化实现，实际应使用优化算法
        let variance = calculate_variance(returns);
        
        Self {
            omega: variance * 0.1,  // 长期方差的10%
            alpha: 0.1,              // ARCH效应
            beta: 0.85,              // 持续性
        }
    }
    
    /// 预测未来波动率
    pub fn forecast_volatility(&self, returns: &[f64], horizon: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(horizon);
        let last_return = returns.last().unwrap_or(&0.0);
        let last_variance = last_return.powi(2);
        
        let mut sigma_squared = self.omega + self.alpha * last_variance;
        
        for _ in 0..horizon {
            forecasts.push(sigma_squared.sqrt());
            // 多步预测收敛到长期方差
            sigma_squared = self.omega / (1.0 - self.alpha - self.beta) * 0.9 
                          + sigma_squared * 0.1;
        }
        
        forecasts
    }
}

// ============================================================================
// 市场微结构分析
// ============================================================================

pub struct MarketMicrostructure {
    pub bid_ask_spread: f64,
    pub order_imbalance: f64,
    pub trade_intensity: f64,
    pub price_impact: f64,
}

impl MarketMicrostructure {
    pub fn analyze(
        prices: &[f64],
        volumes: &[i64],
        highs: &[f64],
        lows: &[f64],
    ) -> Self {
        let n = prices.len();
        
        // 估算买卖价差（使用高低价近似）
        let spread = if n > 0 {
            let avg_range: f64 = (0..n)
                .map(|i| (highs[i] - lows[i]) / prices[i])
                .sum::<f64>() / n as f64;
            avg_range * 0.5  // 假设价差是日内波动的一半
        } else {
            0.001
        };
        
        // 订单失衡（使用价格变化和成交量）
        let order_imbalance = calculate_order_imbalance(prices, volumes);
        
        // 交易强度（标准化成交量）
        let avg_volume = volumes.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let recent_volume = volumes.last().unwrap_or(&0) * 1.0;
        let trade_intensity = recent_volume as f64 / avg_volume.max(1.0);
        
        // 价格冲击（大单对价格的影响）
        let price_impact = calculate_price_impact(prices, volumes);
        
        Self {
            bid_ask_spread: spread,
            order_imbalance,
            trade_intensity,
            price_impact,
        }
    }
}

// ============================================================================
// 核心预测引擎
// ============================================================================

pub struct FinancialPredictionEngine {
    garch_model: Option<GarchModel>,
    market_regime: MarketRegime,
    adaptive_weights: HashMap<String, f64>,
}

impl FinancialPredictionEngine {
    pub fn new() -> Self {
        Self {
            garch_model: None,
            market_regime: MarketRegime::Stable,
            adaptive_weights: Self::initialize_weights(),
        }
    }
    
    /// 初始化自适应权重
    fn initialize_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        
        // 核心权重（总和为1.0）
        weights.insert("trend".to_string(), 0.25);
        weights.insert("momentum".to_string(), 0.20);
        weights.insert("mean_reversion".to_string(), 0.15);
        weights.insert("volatility".to_string(), 0.15);
        weights.insert("microstructure".to_string(), 0.15);
        weights.insert("sentiment".to_string(), 0.10);
        
        weights
    }
    
    /// 主预测函数
    pub fn predict(
        &mut self,
        prices: &[f64],
        volumes: &[i64],
        highs: &[f64],
        lows: &[f64],
        horizon: usize,
    ) -> Vec<FinancialPrediction> {
        // 1. 数据预处理
        let returns = calculate_returns(prices);
        
        // 2. 识别市场状态
        self.market_regime = identify_market_regime(&returns, prices);
        
        // 3. 训练GARCH模型
        self.garch_model = Some(GarchModel::estimate(&returns));
        
        // 4. 预测波动率
        let volatility_forecast = self.garch_model
            .as_ref()
            .unwrap()
            .forecast_volatility(&returns, horizon);
        
        // 5. 市场微结构分析
        let microstructure = MarketMicrostructure::analyze(prices, volumes, highs, lows);
        
        // 6. 自适应权重调整
        self.adjust_weights(&self.market_regime, &microstructure);
        
        // 7. 生成预测
        let mut predictions = Vec::new();
        let last_price = prices.last().unwrap_or(&100.0);
        
        for i in 0..horizon {
            let prediction = self.generate_prediction(
                *last_price,
                &returns,
                &volatility_forecast[i],
                &microstructure,
                i + 1,
            );
            predictions.push(prediction);
        }
        
        predictions
    }
    
    /// 生成单个预测
    fn generate_prediction(
        &self,
        last_price: f64,
        returns: &[f64],
        volatility: &f64,
        microstructure: &MarketMicrostructure,
        days_ahead: usize,
    ) -> FinancialPrediction {
        // 趋势成分
        let trend_return = calculate_trend_component(returns) 
            * self.adaptive_weights["trend"];
        
        // 动量成分
        let momentum_return = calculate_momentum_component(returns)
            * self.adaptive_weights["momentum"];
        
        // 均值回归成分
        let mean_reversion_return = calculate_mean_reversion_component(returns, last_price)
            * self.adaptive_weights["mean_reversion"];
        
        // 微结构成分
        let microstructure_return = microstructure.order_imbalance * 0.01
            * self.adaptive_weights["microstructure"];
        
        // 综合预测收益率
        let mut predicted_return = trend_return + momentum_return 
            + mean_reversion_return + microstructure_return;
        
        // 根据市场状态调整
        predicted_return = self.adjust_for_market_regime(predicted_return, *volatility);
        
        // 时间衰减
        let decay_factor = 0.95_f64.powi(days_ahead as i32);
        predicted_return *= decay_factor;
        
        // 计算预测价格
        let predicted_price = last_price * (1.0 + predicted_return);
        
        // 计算风险调整收益
        let risk_adjusted_return = predicted_return / volatility.max(0.01);
        
        // 生成交易信号
        let trading_signal = generate_trading_signal(
            predicted_return,
            *volatility,
            risk_adjusted_return,
        );
        
        // 计算风险指标
        let risk_metrics = calculate_risk_metrics(
            returns,
            predicted_return,
            *volatility,
        );
        
        // 置信度计算
        let confidence = calculate_confidence(
            &self.market_regime,
            volatility,
            microstructure,
        );
        
        FinancialPrediction {
            predicted_price,
            predicted_return,
            confidence,
            volatility: *volatility,
            risk_adjusted_return,
            market_regime: self.market_regime.clone(),
            trading_signal,
            risk_metrics,
        }
    }
    
    /// 根据市场状态调整预测
    fn adjust_for_market_regime(&self, return_: f64, volatility: f64) -> f64 {
        match self.market_regime {
            MarketRegime::Trending => {
                // 趋势市场，增强趋势信号
                return_ * 1.2
            }
            MarketRegime::MeanReverting => {
                // 均值回归市场，减弱极端预测
                return_ * 0.7
            }
            MarketRegime::Volatile => {
                // 高波动市场，限制预测幅度
                return_.clamp(-volatility * 2.0, volatility * 2.0)
            }
            MarketRegime::Stable => {
                // 稳定市场，保持原预测
                return_
            }
        }
    }
    
    /// 动态调整权重
    fn adjust_weights(&mut self, regime: &MarketRegime, micro: &MarketMicrostructure) {
        match regime {
            MarketRegime::Trending => {
                self.adaptive_weights.insert("trend".to_string(), 0.35);
                self.adaptive_weights.insert("momentum".to_string(), 0.25);
                self.adaptive_weights.insert("mean_reversion".to_string(), 0.10);
            }
            MarketRegime::MeanReverting => {
                self.adaptive_weights.insert("trend".to_string(), 0.15);
                self.adaptive_weights.insert("momentum".to_string(), 0.15);
                self.adaptive_weights.insert("mean_reversion".to_string(), 0.30);
            }
            MarketRegime::Volatile => {
                self.adaptive_weights.insert("volatility".to_string(), 0.25);
                self.adaptive_weights.insert("microstructure".to_string(), 0.20);
            }
            MarketRegime::Stable => {
                // 使用默认权重
            }
        }
        
        // 根据微结构调整
        if micro.order_imbalance.abs() > 0.5 {
            let current = self.adaptive_weights["microstructure"];
            self.adaptive_weights.insert("microstructure".to_string(), current * 1.2);
        }
        
        // 归一化权重
        let sum: f64 = self.adaptive_weights.values().sum();
        for weight in self.adaptive_weights.values_mut() {
            *weight /= sum;
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 计算收益率
fn calculate_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return vec![];
    }
    
    prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// 计算方差
fn calculate_variance(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64
}

/// 识别市场状态
fn identify_market_regime(returns: &[f64], prices: &[f64]) -> MarketRegime {
    if returns.len() < 20 {
        return MarketRegime::Stable;
    }
    
    // 计算各种统计量
    let volatility = calculate_variance(returns).sqrt();
    let trend_strength = calculate_trend_strength(prices);
    let mean_reversion_score = calculate_mean_reversion_score(returns);
    
    // 判断市场状态
    if volatility > 0.03 {
        MarketRegime::Volatile
    } else if trend_strength > 0.7 {
        MarketRegime::Trending
    } else if mean_reversion_score > 0.6 {
        MarketRegime::MeanReverting
    } else {
        MarketRegime::Stable
    }
}

/// 计算趋势强度
fn calculate_trend_strength(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    
    // 使用线性回归R²
    let n = prices.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = prices.iter().sum::<f64>() / n;
    
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for (i, &y) in prices.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }
    
    if denominator == 0.0 {
        return 0.0;
    }
    
    let slope = numerator / denominator;
    let intercept = y_mean - slope * x_mean;
    
    for (i, &y) in prices.iter().enumerate() {
        let x = i as f64;
        let y_pred = slope * x + intercept;
        ss_res += (y - y_pred).powi(2);
        ss_tot += (y - y_mean).powi(2);
    }
    
    if ss_tot == 0.0 {
        return 0.0;
    }
    
    1.0 - (ss_res / ss_tot)
}

/// 计算均值回归得分
fn calculate_mean_reversion_score(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    // 计算自相关系数（负自相关表示均值回归）
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    if variance == 0.0 {
        return 0.0;
    }
    
    let mut autocorr = 0.0;
    for i in 1..returns.len() {
        autocorr += (returns[i] - mean) * (returns[i-1] - mean);
    }
    autocorr /= (returns.len() - 1) as f64 * variance;
    
    // 负自相关越强，均值回归越明显
    (-autocorr).max(0.0).min(1.0)
}

/// 计算订单失衡
fn calculate_order_imbalance(prices: &[f64], volumes: &[i64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    
    let mut buy_volume = 0.0;
    let mut sell_volume = 0.0;
    
    for i in 1..prices.len() {
        let volume = volumes[i] as f64;
        if prices[i] > prices[i-1] {
            buy_volume += volume;
        } else {
            sell_volume += volume;
        }
    }
    
    let total_volume = buy_volume + sell_volume;
    if total_volume == 0.0 {
        return 0.0;
    }
    
    (buy_volume - sell_volume) / total_volume
}

/// 计算价格冲击
fn calculate_price_impact(prices: &[f64], volumes: &[i64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    
    let mut impacts = Vec::new();
    
    for i in 1..prices.len() {
        let price_change = (prices[i] - prices[i-1]).abs() / prices[i-1];
        let volume = volumes[i] as f64;
        let avg_volume = volumes.iter().map(|&v| v as f64).sum::<f64>() / volumes.len() as f64;
        
        if volume > avg_volume * 1.5 {
            // 大单的价格冲击
            impacts.push(price_change);
        }
    }
    
    if impacts.is_empty() {
        return 0.0;
    }
    
    impacts.iter().sum::<f64>() / impacts.len() as f64
}

/// 计算趋势成分
fn calculate_trend_component(returns: &[f64]) -> f64 {
    if returns.len() < 5 {
        return 0.0;
    }
    
    // 使用指数加权移动平均
    let alpha = 2.0 / (5.0 + 1.0);
    let mut ema = returns[0];
    
    for &r in &returns[1..] {
        ema = alpha * r + (1.0 - alpha) * ema;
    }
    
    ema
}

/// 计算动量成分
fn calculate_momentum_component(returns: &[f64]) -> f64 {
    if returns.len() < 10 {
        return 0.0;
    }
    
    // 最近10日动量
    let recent_returns: f64 = returns[returns.len()-10..].iter().sum();
    recent_returns / 10.0
}

/// 计算均值回归成分
fn calculate_mean_reversion_component(returns: &[f64], current_price: f64) -> f64 {
    if returns.len() < 20 {
        return 0.0;
    }
    
    // 计算历史均值
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let last_return = returns.last().unwrap_or(&0.0);
    
    // 偏离均值的程度
    let deviation = last_return - mean_return;
    
    // 均值回归力量（反向）
    -deviation * 0.5
}

/// 生成交易信号
fn generate_trading_signal(
    predicted_return: f64,
    volatility: f64,
    risk_adjusted_return: f64,
) -> TradingSignal {
    // 确定交易动作
    let action = if risk_adjusted_return > 2.0 {
        TradeAction::StrongBuy
    } else if risk_adjusted_return > 1.0 {
        TradeAction::Buy
    } else if risk_adjusted_return < -2.0 {
        TradeAction::StrongSell
    } else if risk_adjusted_return < -1.0 {
        TradeAction::Sell
    } else {
        TradeAction::Hold
    };
    
    // 信号强度
    let strength = risk_adjusted_return.abs().min(3.0) / 3.0;
    
    // 凯利公式计算最优仓位
    let win_rate = 0.55;  // 假设55%胜率
    let win_loss_ratio = 1.5;  // 假设盈亏比1.5
    let kelly_fraction = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio;
    let optimal_position = (kelly_fraction * 0.25).max(0.0).min(0.25);  // 限制最大25%仓位
    
    // 止损止盈
    let stop_loss = -volatility * 2.0;
    let take_profit = vec![
        volatility * 1.5,
        volatility * 2.5,
        volatility * 4.0,
    ];
    
    TradingSignal {
        action,
        strength,
        optimal_position,
        stop_loss,
        take_profit,
    }
}

/// 计算风险指标
fn calculate_risk_metrics(
    returns: &[f64],
    predicted_return: f64,
    volatility: f64,
) -> RiskMetrics {
    // VaR (95%置信水平)
    let value_at_risk = -1.645 * volatility;
    
    // CVaR (期望短缺)
    let expected_shortfall = -2.063 * volatility;
    
    // 夏普比率
    let risk_free_rate = 0.03 / 252.0;  // 年化3%的日收益率
    let sharpe_ratio = (predicted_return - risk_free_rate) / volatility.max(0.001);
    
    // 最大回撤
    let max_drawdown = calculate_max_drawdown(returns);
    
    // Beta (相对市场)
    let beta = 1.0;  // 简化，实际需要市场数据
    
    RiskMetrics {
        value_at_risk,
        expected_shortfall,
        sharpe_ratio,
        max_drawdown,
        beta,
    }
}

/// 计算最大回撤
fn calculate_max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mut cumulative = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    
    for &r in returns {
        cumulative *= 1.0 + r;
        peak = peak.max(cumulative);
        let drawdown = (peak - cumulative) / peak;
        max_dd = max_dd.max(drawdown);
    }
    
    max_dd
}

/// 计算置信度
fn calculate_confidence(
    regime: &MarketRegime,
    volatility: &f64,
    micro: &MarketMicrostructure,
) -> f64 {
    let mut confidence = 0.5;
    
    // 市场状态影响
    confidence += match regime {
        MarketRegime::Trending => 0.2,
        MarketRegime::Stable => 0.15,
        MarketRegime::MeanReverting => 0.1,
        MarketRegime::Volatile => -0.1,
    };
    
    // 波动率影响
    if *volatility < 0.01 {
        confidence += 0.1;
    } else if *volatility > 0.03 {
        confidence -= 0.1;
    }
    
    // 微结构影响
    if micro.order_imbalance.abs() > 0.3 {
        confidence += 0.1;
    }
    
    confidence.max(0.1).min(0.9)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_garch_model() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01];
        let model = GarchModel::estimate(&returns);
        let forecast = model.forecast_volatility(&returns, 5);
        
        assert_eq!(forecast.len(), 5);
        assert!(forecast[0] > 0.0);
    }
    
    #[test]
    fn test_market_regime_identification() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let returns = calculate_returns(&prices);
        let regime = identify_market_regime(&returns, &prices);
        
        match regime {
            MarketRegime::Trending => assert!(true),
            _ => assert!(false, "Expected trending market"),
        }
    }
}
