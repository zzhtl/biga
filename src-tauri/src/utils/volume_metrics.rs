//! 量比与换手率计算
//!
//! - 量比 = 当日成交量 / 过去 N 日平均成交量（与单位无关，比值）
//! - 换手率 = 成交额 / 流通市值 × 100（单位安全：成交额与市值均为元）
//!
//! 之所以用成交额而非成交量计算换手率，是因为库内 `volume` 单位为"手"，
//! 直接除以流通股本会引入手/股的单位换算误差；用成交额 / 流通市值则完全规避。
//!
//! 放在 utils（叶子层）以便 db 与 prediction 都能依赖，避免层级循环。

/// 默认量比统计窗口（交易日）
pub const DEFAULT_VOLUME_RATIO_PERIOD: usize = 5;

/// 计算量比序列。
///
/// `volume_ratio[i] = volume[i] / mean(volume[i-period..i])`。
/// 前 `period` 个交易日因样本不足记为 `0.0`（视为无效）。
pub fn calculate_volume_ratio_series(volumes: &[f64], period: usize) -> Vec<f64> {
    if period == 0 {
        return vec![0.0; volumes.len()];
    }

    let mut ratios = vec![0.0; volumes.len()];
    for i in period..volumes.len() {
        let window = &volumes[i - period..i];
        let avg = window.iter().sum::<f64>() / period as f64;
        ratios[i] = if avg > 0.0 { volumes[i] / avg } else { 0.0 };
    }
    ratios
}

/// 计算单日换手率（%）。
///
/// `换手率 = 成交额 / 流通市值 × 100`，其中流通市值 = 流通股本 × 当日收盘价。
/// 缺少有效股本/价格时返回 `0.0`。
pub fn calculate_turnover_rate(amount: f64, close: f64, circulating_shares: f64) -> f64 {
    let market_cap = circulating_shares * close;
    if market_cap > 0.0 {
        amount / market_cap * 100.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_ratio_basic() {
        // 前 5 日均量 = 100，第 6 日量 200 → 量比 2.0
        let volumes = vec![100.0, 100.0, 100.0, 100.0, 100.0, 200.0];
        let ratios = calculate_volume_ratio_series(&volumes, 5);
        // 前 5 日无效
        assert_eq!(&ratios[0..5], &[0.0; 5]);
        assert!((ratios[5] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_volume_ratio_insufficient_data() {
        let volumes = vec![100.0, 120.0, 90.0];
        let ratios = calculate_volume_ratio_series(&volumes, 5);
        assert_eq!(ratios, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_volume_ratio_zero_average() {
        let volumes = vec![0.0, 0.0, 50.0];
        let ratios = calculate_volume_ratio_series(&volumes, 2);
        // 索引 2：过去两日均量为 0 → 量比 0.0（防止除零）
        assert_eq!(ratios[2], 0.0);
    }

    #[test]
    fn test_turnover_rate_basic() {
        // 流通股本 1 亿股，价 10 元 → 流通市值 10 亿；成交额 5000 万 → 换手率 5%
        let turnover = calculate_turnover_rate(50_000_000.0, 10.0, 100_000_000.0);
        assert!((turnover - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_turnover_rate_invalid_shares() {
        assert_eq!(calculate_turnover_rate(1000.0, 10.0, 0.0), 0.0);
        assert_eq!(calculate_turnover_rate(1000.0, 0.0, 100.0), 0.0);
    }
}
