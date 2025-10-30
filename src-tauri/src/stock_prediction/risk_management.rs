//! 风险管理模块
//! 
//! 提供专业的风险控制和仓位管理功能

use serde::{Deserialize, Serialize};

/// 风险管理器
pub struct RiskManager {
    max_position_size: f64,     // 最大仓位
    max_portfolio_risk: f64,    // 最大组合风险
    risk_free_rate: f64,        // 无风险利率
}

impl RiskManager {
    pub fn new() -> Self {
        Self {
            max_position_size: 0.25,    // 单个仓位不超过25%
            max_portfolio_risk: 0.10,   // 组合风险不超过10%
            risk_free_rate: 0.03,        // 年化3%
        }
    }
    
    /// 凯利公式计算最优仓位
    /// f = (p * b - q) / b
    /// p: 获胜概率, b: 盈亏比, q: 失败概率
    pub fn kelly_criterion(
        &self,
        win_probability: f64,
        win_loss_ratio: f64,
    ) -> f64 {
        let q = 1.0 - win_probability;
        let kelly_fraction = (win_probability * win_loss_ratio - q) / win_loss_ratio;
        
        // 使用部分凯利（25%）以降低风险
        let position = kelly_fraction * 0.25;
        
        // 限制最大仓位
        position.max(0.0).min(self.max_position_size)
    }
    
    /// 计算止损位
    pub fn calculate_stop_loss(
        &self,
        entry_price: f64,
        volatility: f64,
        atr: f64,
    ) -> f64 {
        // 使用2倍ATR或2倍标准差作为止损
        let stop_distance = (atr * 2.0).max(volatility * 2.0);
        entry_price * (1.0 - stop_distance)
    }
    
    /// 计算止盈位（多个目标）
    pub fn calculate_take_profits(
        &self,
        entry_price: f64,
        volatility: f64,
        risk_reward_ratios: &[f64],
    ) -> Vec<f64> {
        risk_reward_ratios
            .iter()
            .map(|&ratio| entry_price * (1.0 + volatility * 2.0 * ratio))
            .collect()
    }
    
    /// 风险价值(VaR)计算
    pub fn calculate_var(
        &self,
        returns: &[f64],
        confidence_level: f64,
        holding_period: usize,
    ) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        // 参数法VaR（假设正态分布）
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = calculate_std_dev(returns, mean);
        
        // Z值（95%置信度 = 1.645, 99% = 2.326）
        let z_score = match confidence_level {
            0.95 => 1.645,
            0.99 => 2.326,
            _ => 1.645,
        };
        
        // VaR = μ - z * σ * √t
        let var = mean * holding_period as f64 
                - z_score * std_dev * (holding_period as f64).sqrt();
        
        -var  // VaR通常表示为正值
    }
    
    /// 条件风险价值(CVaR)计算
    pub fn calculate_cvar(
        &self,
        returns: &[f64],
        var: f64,
    ) -> f64 {
        // 计算超过VaR的平均损失
        let losses: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < -var)
            .copied()
            .collect();
        
        if losses.is_empty() {
            return var;
        }
        
        -losses.iter().sum::<f64>() / losses.len() as f64
    }
    
    /// 夏普比率计算
    pub fn calculate_sharpe_ratio(
        &self,
        returns: &[f64],
    ) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = calculate_std_dev(returns, mean_return);
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        // 日化夏普比率
        let daily_risk_free = self.risk_free_rate / 252.0;
        (mean_return - daily_risk_free) / std_dev
    }
    
    /// 最大回撤计算
    pub fn calculate_max_drawdown(
        &self,
        prices: &[f64],
    ) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let mut peak = prices[0];
        let mut max_dd = 0.0;
        
        for &price in prices {
            peak = peak.max(price);
            let drawdown = (peak - price) / peak;
            max_dd = max_dd.max(drawdown);
        }
        
        max_dd
    }
    
    /// 风险调整后收益(RAROC)
    pub fn calculate_raroc(
        &self,
        expected_return: f64,
        risk_capital: f64,
    ) -> f64 {
        if risk_capital == 0.0 {
            return 0.0;
        }
        
        expected_return / risk_capital
    }
    
    /// 动态风险限额
    pub fn calculate_risk_limit(
        &self,
        portfolio_value: f64,
        market_volatility: f64,
    ) -> RiskLimits {
        // 根据市场波动率动态调整风险限额
        let vol_adjustment = if market_volatility > 0.03 {
            0.5  // 高波动时减半
        } else if market_volatility < 0.01 {
            1.5  // 低波动时增加50%
        } else {
            1.0
        };
        
        RiskLimits {
            max_position: self.max_position_size * vol_adjustment,
            max_daily_loss: portfolio_value * 0.02 * vol_adjustment,  // 2%日损失限制
            max_weekly_loss: portfolio_value * 0.05 * vol_adjustment,  // 5%周损失限制
            max_leverage: 2.0 / (1.0 + market_volatility * 10.0),     // 动态杠杆
        }
    }
}

/// 风险限额
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position: f64,
    pub max_daily_loss: f64,
    pub max_weekly_loss: f64,
    pub max_leverage: f64,
}

/// 仓位优化器
pub struct PositionOptimizer {
    risk_manager: RiskManager,
}

impl PositionOptimizer {
    pub fn new() -> Self {
        Self {
            risk_manager: RiskManager::new(),
        }
    }
    
    /// 马科维茨组合优化（简化版）
    pub fn optimize_portfolio(
        &self,
        expected_returns: &[f64],
        covariance_matrix: &[Vec<f64>],
        target_return: f64,
    ) -> Vec<f64> {
        let n = expected_returns.len();
        
        // 简化：等权重分配
        let equal_weight = 1.0 / n as f64;
        vec![equal_weight; n]
    }
    
    /// 风险平价配置
    pub fn risk_parity_allocation(
        &self,
        volatilities: &[f64],
    ) -> Vec<f64> {
        let inverse_vols: Vec<f64> = volatilities
            .iter()
            .map(|&v| 1.0 / v.max(0.001))
            .collect();
        
        let sum: f64 = inverse_vols.iter().sum();
        
        inverse_vols
            .iter()
            .map(|&iv| iv / sum)
            .collect()
    }
}

// 辅助函数
fn calculate_std_dev(returns: &[f64], mean: f64) -> f64 {
    let variance = returns
        .iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kelly_criterion() {
        let rm = RiskManager::new();
        let position = rm.kelly_criterion(0.6, 1.5);
        assert!(position > 0.0 && position <= 0.25);
    }
    
    #[test]
    fn test_var_calculation() {
        let rm = RiskManager::new();
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
        let var = rm.calculate_var(&returns, 0.95, 1);
        assert!(var > 0.0);
    }
}
