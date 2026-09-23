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

/// RiskMetrics 标准的 EWMA 衰减系数（日频）
pub const EWMA_LAMBDA: f64 = 0.94;

/// 日收益序列（简单收益）。
fn daily_returns(closes: &[f64]) -> Vec<f64> {
    closes
        .windows(2)
        .filter(|w| w[0] > 0.0)
        .map(|w| (w[1] - w[0]) / w[0])
        .filter(|r| r.is_finite())
        .collect()
}

/// EWMA 日波动率（指数加权、**零均值**）。
///
/// 与 [`calculate_realized_volatility`] 的两处关键差别：
/// 1. **不减样本均值**。日频下漂移相对波动是二阶小量，减样本均值等于把一部分真实漂移
///    当成噪声扣掉，还额外引入估计误差；短窗口尤其明显。
/// 2. **指数加权**而非等权。波动率有聚集效应，20 日等权窗把 20 天前的信息和昨天同等看待，
///    在波动状态切换时反应迟钝——这正是"名义 80% 的带在高波动段覆盖不足"的来源之一。
///
/// λ 用 [`EWMA_LAMBDA`] 时有效窗口约 1/(1−λ) ≈ 17 个交易日。
pub fn ewma_daily_vol(closes: &[f64], lambda: f64) -> Option<f64> {
    let returns = daily_returns(closes);
    if returns.is_empty() || !(0.0..1.0).contains(&lambda) {
        return None;
    }
    // σ² = (1−λ)/(1−λ^N) · Σ λ^i r²_{t−i}，归一化写法，省掉初值选择
    let mut weighted = 0.0;
    let mut weight_sum = 0.0;
    let mut w = 1.0;
    for r in returns.iter().rev() {
        weighted += w * r * r;
        weight_sum += w;
        w *= lambda;
        if w < 1e-12 {
            break;
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let var = weighted / weight_sum;
    (var.is_finite() && var > 0.0).then(|| var.sqrt())
}

/// 零均值已实现日波动率（等权），用作长期波动基准。
pub fn realized_daily_vol_zero_mean(closes: &[f64]) -> Option<f64> {
    let returns = daily_returns(closes);
    if returns.is_empty() {
        return None;
    }
    let var = returns.iter().map(|r| r * r).sum::<f64>() / returns.len() as f64;
    (var.is_finite() && var > 0.0).then(|| var.sqrt())
}

/// 累计波动率的期限结构：返回第 1..=`days` 个预测日的**累计**标准差。
///
/// 现行区间用的是 `σ·√d`，等价于假设日收益 IID、且当前波动会一直持续下去。实际上波动率
/// 均值回归：当前处在高波动段时 `√d` 会高估远期不确定性，低波动段则低估。在方差上按
/// GARCH 的期限结构递推可以消掉这一层系统性偏差：
///
/// ```text
/// σ²_{t+k} = V + φ^k (σ²_t − V)
/// 累计方差(H) = Σ_{k=1..H} σ²_{t+k} = H·V + (σ²_t − V)·φ(1−φ^H)/(1−φ)
/// ```
///
/// `φ`（持续性）为 1 时退化回 `σ·√d`；`sigma_now == sigma_long` 时也精确等于 `σ·√d`。
///
/// - `sigma_now`：当前条件日波动（建议用 [`ewma_daily_vol`]）
/// - `sigma_long`：长期日波动（建议用长窗的 [`realized_daily_vol_zero_mean`]）
/// - `persistence`：φ ∈ [0, 1)，可由 [`estimate_garch_params`] 的 `alpha + beta` 得到
pub fn cumulative_sigma_path(
    sigma_now: f64,
    sigma_long: f64,
    persistence: f64,
    days: usize,
) -> Vec<f64> {
    if days == 0 || !sigma_now.is_finite() || sigma_now <= 0.0 {
        return Vec::new();
    }
    let var_now = sigma_now * sigma_now;
    let var_long = if sigma_long.is_finite() && sigma_long > 0.0 {
        sigma_long * sigma_long
    } else {
        var_now
    };
    let phi = persistence.clamp(0.0, 1.0);

    // σ²_{t+k} = V + φ^(k−1)·(σ²_now − V)。指数是 k−1 而不是 k：`sigma_now` 是**下一日**的
    // 条件波动预测（EWMA 在 t 时刻给出的就是对 t+1 的预测），所以 d=1 必须精确等于它，
    // 均值回归从第二天才开始起作用。
    let mut out = Vec::with_capacity(days);
    let mut cumulative_var = 0.0;
    let mut phi_pow = 1.0;
    for _ in 1..=days {
        cumulative_var += var_long + phi_pow * (var_now - var_long);
        out.push(cumulative_var.max(0.0).sqrt());
        phi_pow *= phi;
    }
    out
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
    
    /// 由日收益序列造价格序列
    fn prices_from_returns(start: f64, rets: &[f64]) -> Vec<f64> {
        let mut out = vec![start];
        for r in rets {
            let last = *out.last().unwrap();
            out.push(last * (1.0 + r));
        }
        out
    }

    #[test]
    fn test_ewma_recovers_constant_volatility() {
        // 收益恒为 ±1% → 零均值日波动就是 1%
        let rets: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();
        let prices = prices_from_returns(100.0, &rets);
        let v = ewma_daily_vol(&prices, EWMA_LAMBDA).expect("应能算出");
        assert!((v - 0.01).abs() < 5e-4, "EWMA 应恢复出 1%，得到 {v}");
    }

    #[test]
    fn test_ewma_reacts_faster_than_equal_weight() {
        // 前 100 天日波动 0.5%，最近 20 天跳到 3%
        let mut rets: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.005 } else { -0.005 })
            .collect();
        rets.extend((0..20).map(|i| if i % 2 == 0 { 0.03 } else { -0.03 }));
        let prices = prices_from_returns(100.0, &rets);

        let ewma = ewma_daily_vol(&prices, EWMA_LAMBDA).expect("应能算出");
        let equal_all = realized_daily_vol_zero_mean(&prices).expect("应能算出");

        assert!(
            ewma > equal_all,
            "波动跳升后 EWMA 必须高于全样本等权: ewma={ewma} equal={equal_all}"
        );
        assert!(ewma > 0.02, "EWMA 应贴近新状态的 3%，得到 {ewma}");
    }

    #[test]
    fn test_zero_mean_vol_keeps_drift_in() {
        // 单边上涨：减样本均值会把漂移当噪声扣掉，零均值口径则保留
        let rets: Vec<f64> = (0..100).map(|_| 0.01).collect();
        let prices = prices_from_returns(100.0, &rets);
        let zero_mean = realized_daily_vol_zero_mean(&prices).expect("应能算出");
        let demeaned = calculate_realized_volatility(&prices);
        assert!((zero_mean - 0.01).abs() < 1e-6, "零均值口径应为 1%，得到 {zero_mean}");
        assert!(demeaned < 1e-6, "减掉样本均值后恒定漂移的波动为 0，得到 {demeaned}");
        assert!(zero_mean > demeaned);
    }

    #[test]
    fn test_ewma_rejects_bad_input() {
        assert!(ewma_daily_vol(&[], EWMA_LAMBDA).is_none());
        assert!(ewma_daily_vol(&[100.0], EWMA_LAMBDA).is_none());
        assert!(ewma_daily_vol(&[100.0, 101.0], 1.0).is_none(), "λ 必须 <1");
        // 价格恒定 → 方差为 0，无法用作区间宽度
        assert!(ewma_daily_vol(&[100.0; 30], EWMA_LAMBDA).is_none());
    }

    #[test]
    fn test_sigma_path_reduces_to_sqrt_d_when_no_mean_reversion() {
        let sigma = 0.02;
        // φ=1：完全持续，退化回 σ·√d
        let path = cumulative_sigma_path(sigma, 0.05, 1.0, 5);
        for (i, v) in path.iter().enumerate() {
            let want = sigma * ((i + 1) as f64).sqrt();
            assert!((v - want).abs() < 1e-12, "d={} 得到 {v} 期望 {want}", i + 1);
        }
        // 当前波动恰等于长期波动时也应精确等于 σ·√d
        let path = cumulative_sigma_path(sigma, sigma, 0.9, 5);
        for (i, v) in path.iter().enumerate() {
            let want = sigma * ((i + 1) as f64).sqrt();
            assert!((v - want).abs() < 1e-12, "d={} 得到 {v} 期望 {want}", i + 1);
        }
    }

    #[test]
    fn test_sigma_path_mean_reverts_in_both_directions() {
        let phi = 0.94;
        let horizon = 20;

        // 当前高于长期 → 累计波动应低于 √d 外推（√d 高估了远期不确定性）
        let high = cumulative_sigma_path(0.05, 0.02, phi, horizon);
        let naive_high = 0.05 * (horizon as f64).sqrt();
        assert!(
            *high.last().unwrap() < naive_high,
            "高波动段应低于 √d 外推: {} vs {naive_high}",
            high.last().unwrap()
        );

        // 当前低于长期 → 累计波动应高于 √d 外推
        let low = cumulative_sigma_path(0.01, 0.03, phi, horizon);
        let naive_low = 0.01 * (horizon as f64).sqrt();
        assert!(
            *low.last().unwrap() > naive_low,
            "低波动段应高于 √d 外推: {} vs {naive_low}",
            low.last().unwrap()
        );

        // 第一天永远等于当日条件波动，与均值回归无关
        assert!((high[0] - 0.05).abs() < 1e-12);
        assert!((low[0] - 0.01).abs() < 1e-12);

        // 累计波动必须随 horizon 单调变宽
        assert!(high.windows(2).all(|w| w[1] > w[0]));
        assert!(low.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn test_sigma_path_converges_to_long_run() {
        // H 足够大时 σ_H/√H 应收敛到长期波动
        // 收敛速度是 O(1/H)：初始偏离被摊到 H 天上，所以要取足够大的 H 才看得出极限
        let long_run = 0.025;
        let horizon = 5000;
        let path = cumulative_sigma_path(0.06, long_run, 0.94, horizon);
        let implied = path.last().unwrap() / (horizon as f64).sqrt();
        assert!(
            (implied - long_run).abs() < 1e-3,
            "远期隐含波动应收敛到长期值: {implied} vs {long_run}"
        );
    }

    #[test]
    fn test_sigma_path_edge_cases() {
        assert!(cumulative_sigma_path(0.02, 0.02, 0.9, 0).is_empty());
        assert!(cumulative_sigma_path(0.0, 0.02, 0.9, 3).is_empty());
        assert!(cumulative_sigma_path(f64::NAN, 0.02, 0.9, 3).is_empty());
        // 长期波动非法时退化成用当前波动（即 √d）
        let path = cumulative_sigma_path(0.02, 0.0, 0.5, 3);
        for (i, v) in path.iter().enumerate() {
            assert!((v - 0.02 * ((i + 1) as f64).sqrt()).abs() < 1e-12);
        }
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

