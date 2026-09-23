//! 校准区间预测：方向不可测但波动可测。
//!
//! 实证（见 README「关键实证结论」）：单股次日方向无 alpha，引擎点预测已被压成"几乎不动"
//! （这是 MAE 最优且诚实的）。与其伪造无预测力的日间涨跌，不如诚实给出**校准的涨跌区间带**。
//!
//! # 方法
//!
//! 区间边界 = 点预测 + 校准分位 × σ_H，其中 σ_H 是 H 日累计标准差。两处都经
//! `examples/interval_calibration.rs` 在 132 票、h=1/5/10、约 7.7 万个非重叠走步样本、
//! 按时间切 70/30 训练/留出上实测得到。
//!
//! 1. **σ 用「EWMA 向长期方差收缩 + 波动率均值回归期限结构」**，不再用「近 20 日等权已实现
//!    波动 × √d」。旧口径的问题不在边际覆盖率（那本来就是照着它标的），而在**条件覆盖**：
//!    按 σ_now/σ_long 分三档，名义 80% 的带在最高与最低波动档之间差了
//!    12.2pp(h=1) / 15.2pp(h=5) / 16.8pp(h=10)——整体数字看着正常，落到具体某一类票上完全
//!    不可信。新口径把这个跨度压到 1.0pp / 2.2pp / 4.1pp，同时**带更窄、pinball 更低**。
//!
//!    两处根因：√d 假设日收益 IID（而波动率均值回归，当前波动偏离长期水平时 √d 会系统性
//!    地朝一个方向外推错）；以及 EWMA 是"当前波动的估计"而非"下一日波动的预测"——当前很
//!    安静的票，明日风险高于其近期波动所显示的。
//!
//! 2. **分位上下分开取**。实测标准化残差右偏（95% 档下侧 −1.85、上侧 +2.34），
//!    原来的对称 ±2.50 在下侧明显过宽：实际只有 0.6% 的样本跌穿下界，名义该是 2.5%。
//!
//! 分位取自 h=1 的全样本经验分位（7.7 万样本，尾部估计最稳），三个 horizon 通用——实测
//! 各 horizon 的全样本分位几乎一致（80% 档 −1.106/−1.107/−1.108）。详见 [`calibrated_quantiles`]。
//!
//! # 已知局限
//!
//! 写死的常数必然承担 regime drift：全样本标定的分位在留出段（最近约一年）偏窄，
//! 边际覆盖 −0.5pp(h=1) / −2.5pp(h=5) / −4.2pp(h=10)，随 horizon 增大。这不是选错了分位
//! ——留出段自己拟合的分位同样偏窄——而是近期尾部比历史平均更厚。要消掉它只能拿最近一年
//! 去拟合，那等于把验证集当训练集，所以刻意不做。
//!
//! 第 1 天的区间另外按涨跌停限幅截断：越过限幅的那一截对应不可能发生的结果，留着只会
//! 虚增带宽而换不来覆盖率。这一步只收窄带宽、不损失覆盖（当日真实涨跌幅本就在限幅内）。

use crate::prediction::analysis::volatility_forecast::{
    cumulative_sigma_path, estimate_garch_params, ewma_daily_vol, realized_daily_vol_zero_mean,
    EWMA_LAMBDA,
};
use crate::prediction::types::{Prediction, PredictionInterval};
use crate::utils::symbol::price_limit_percent;

/// 长期波动率回看窗口（交易日），同时作为对外报告的 `lookback_days`
const LONG_VOL_WINDOW: usize = 250;

/// 数据不足时的兜底日波动率
const FALLBACK_DAILY_VOL: f64 = 0.02;

/// 起始方差向长期方差收缩的权重：`σ²_start = W·σ²_ewma + (1−W)·σ²_long`。
///
/// 0.8 是实测出来的，不是 ACF 估的持续性 φ（≈0.95）。差别有来源：EWMA 估计本身带测量
/// 误差，最优的一步预测比几何持续性所隐含的收缩得更多。h=1 上 W=1.0（即不收缩）条件覆盖
/// 跨度 7.3pp，W=0.8 降到 1.0pp，pinball 同时取到最低；再往下收缩（W=0.5/0.65）h=10 会
/// 被过度修正，因为期限结构本身已经做了多步回归。
const START_VAR_SHRINK: f64 = 0.8;

/// 估 EWMA / GARCH 持续性所需的最小样本
const MIN_VOL_SAMPLES: usize = 30;

/// 序列化到响应中的稳定方法名。
pub const METHOD: &str = "ewma_term_structure_calibrated";

/// 默认名义覆盖率
pub const DEFAULT_COVERAGE: f64 = 0.80;

/// 校准的标准化残差分位 `(下侧, 上侧)`。
///
/// 取自 `examples/interval_calibration.rs` 在 h=1（约 7.7 万样本，尾部估计最稳）上的全样本
/// 经验分位，σ 口径为 [`horizon_sigma_path`]。上下不对称是实测结果不是笔误：A 股多日简单
/// 收益右偏，下侧尾比上侧短。
///
/// 改动 σ 口径后这组数必须重新标定——两者是配套的，单独改任何一个都会让覆盖率跑掉。
pub fn calibrated_quantiles(confidence: f64) -> (f64, f64) {
    if confidence >= 0.95 {
        (-1.85, 2.34)
    } else if confidence >= 0.90 {
        (-1.47, 1.73)
    } else {
        (-1.11, 1.19) // 默认 80%
    }
}

/// 第 1..=`days` 个预测日的**累计**标准差（小数，如 0.03 = 3%）。
///
/// 三步：零均值 EWMA(λ=0.94) 估当前波动 → 按 [`START_VAR_SHRINK`] 向长期方差收缩得到次日
/// 预测 → 按 φ（取 [`estimate_garch_params`] 的 `alpha + beta`）做均值回归期限结构累加。
/// 长期水平用近 [`LONG_VOL_WINDOW`] 日的零均值已实现波动。数据不足时退回
/// `FALLBACK_DAILY_VOL × √d`。
pub fn horizon_sigma_path(closes: &[f64], days: usize) -> Vec<f64> {
    if days == 0 {
        return Vec::new();
    }
    let fallback = || -> Vec<f64> {
        (1..=days)
            .map(|d| FALLBACK_DAILY_VOL * (d as f64).sqrt())
            .collect()
    };
    if closes.len() < MIN_VOL_SAMPLES {
        return fallback();
    }
    let Some(sigma_now) = ewma_daily_vol(closes, EWMA_LAMBDA) else {
        return fallback();
    };
    let long_start = closes.len().saturating_sub(LONG_VOL_WINDOW);
    let sigma_long = realized_daily_vol_zero_mean(&closes[long_start..]).unwrap_or(sigma_now);
    let params = estimate_garch_params(closes);
    let phi = (params.alpha + params.beta).clamp(0.0, 0.999);

    // EWMA 是当前波动的估计；先收缩一步，把它变成对次日的预测
    let start_var =
        START_VAR_SHRINK * sigma_now * sigma_now + (1.0 - START_VAR_SHRINK) * sigma_long * sigma_long;
    let path = cumulative_sigma_path(start_var.sqrt(), sigma_long, phi, days);
    if path.len() == days && path.iter().all(|s| s.is_finite() && *s > 0.0) {
        path
    } else {
        fallback()
    }
}

/// 为每个预测日填充校准区间带。
///
/// - `closes`：发起日（含）之前的收盘价序列，用于估计波动率
/// - `base_price`：发起日真实收盘价，区间相对它度量累计涨跌
/// - `confidence`：名义覆盖率（默认 [`DEFAULT_COVERAGE`]）
/// - `stock_code`：用于取当日涨跌停限幅，截断第 1 天的区间；`None` 时不截断
///
/// 区间居中于各日点预测价（点预测已近乎不动，区间表达真实不确定性）。
pub fn attach_prediction_intervals(
    predictions: &mut [Prediction],
    closes: &[f64],
    base_price: f64,
    confidence: f64,
    stock_code: Option<&str>,
) {
    if base_price <= 0.0 || !base_price.is_finite() || predictions.is_empty() {
        return;
    }
    let sigma_path = horizon_sigma_path(closes, predictions.len());
    if sigma_path.len() != predictions.len() {
        return;
    }
    // 名字在预测链路上拿不到，ST 只能漏判成主板 ±10%——那只会让带偏宽（过覆盖），
    // 不会造成覆盖不足，是安全的降级方向。
    let daily_limits = stock_code.map(|code| price_limit_percent(code, ""));

    for (idx, prediction) in predictions.iter_mut().enumerate() {
        let day = idx + 1;
        let sigma_h = sigma_path[idx];
        let cum_change = (prediction.predicted_price - base_price) / base_price * 100.0;
        prediction.interval = Some(build_interval(
            base_price,
            cum_change,
            sigma_h,
            day,
            confidence,
            daily_limits,
        ));
        prediction.stress_interval = Some(build_interval(
            base_price,
            cum_change,
            sigma_h,
            day,
            0.95,
            daily_limits,
        ));
    }
}

fn build_interval(
    base_price: f64,
    cumulative_change: f64,
    sigma_h: f64,
    day: usize,
    confidence: f64,
    daily_limits: Option<(f64, f64)>,
) -> PredictionInterval {
    let (lo_q, hi_q) = calibrated_quantiles(confidence);
    let mut lower_change = cumulative_change + lo_q * sigma_h * 100.0;
    let mut upper_change = cumulative_change + hi_q * sigma_h * 100.0;

    // 第 1 天的涨跌幅受当日涨跌停物理约束。超出限幅的那截区间对应的是不可能发生的结果，
    // 留着只虚增带宽、换不来任何覆盖率。
    // 只截第 1 天：涨跌停是单日约束，多日累计涨跌幅不受它约束。
    if let Some((limit_down, limit_up)) = daily_limits {
        if day == 1 {
            lower_change = lower_change.max(limit_down);
            upper_change = upper_change.min(limit_up);
        }
    }

    PredictionInterval {
        confidence,
        lower_change_percent: lower_change,
        upper_change_percent: upper_change,
        lower_price: (base_price * (1.0 + lower_change / 100.0)).max(0.0),
        upper_price: base_price * (1.0 + upper_change / 100.0),
        method: METHOD.to_string(),
        lookback_days: LONG_VOL_WINDOW,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::types::Prediction;

    fn make_predictions(base: f64, daily: f64, days: usize) -> Vec<Prediction> {
        let mut last = base;
        (1..=days)
            .map(|d| {
                last *= 1.0 + daily / 100.0;
                Prediction {
                    target_date: format!("2026-01-{d:02}"),
                    predicted_price: last,
                    predicted_change_percent: daily,
                    confidence: 0.3,
                    trading_signal: None,
                    signal_strength: None,
                    technical_indicators: None,
                    prediction_reason: None,
                    key_factors: None,
                    interval: None,
                    stress_interval: None,
                }
            })
            .collect()
    }

    #[test]
    fn test_interval_widens_with_horizon_and_contains_point() {
        // 恒定 1% 日波动的合成价
        let closes: Vec<f64> = (0..40).map(|i| 100.0 * 1.01_f64.powi(i)).collect();
        let base = *closes.last().unwrap();
        let mut preds = make_predictions(base, 0.0, 5);
        attach_prediction_intervals(&mut preds, &closes, base, DEFAULT_COVERAGE, None);

        let widths: Vec<f64> = preds
            .iter()
            .map(|p| {
                let iv = p.interval.as_ref().unwrap();
                iv.upper_change_percent - iv.lower_change_percent
            })
            .collect();
        // 区间随预测天数单调变宽（√d）
        for w in widths.windows(2) {
            assert!(w[1] > w[0], "区间应随 horizon 变宽: {:?}", widths);
        }
        // 点预测落在自身区间内，且区间对称
        for p in &preds {
            let iv = p.interval.as_ref().unwrap();
            let stress = p.stress_interval.as_ref().unwrap();
            assert!(iv.lower_price < p.predicted_price && p.predicted_price < iv.upper_price);
            assert!((iv.confidence - 0.80).abs() < 1e-9);
            assert!((stress.confidence - 0.95).abs() < 1e-9);
            assert!(stress.lower_price <= iv.lower_price);
            assert!(stress.upper_price >= iv.upper_price);
            assert_eq!(iv.method, METHOD);
            assert_eq!(iv.lookback_days, LONG_VOL_WINDOW);
        }
    }

    #[test]
    fn test_zero_base_price_is_noop() {
        let mut preds = make_predictions(100.0, 0.0, 3);
        attach_prediction_intervals(&mut preds, &[], 0.0, DEFAULT_COVERAGE, None);
        assert!(preds
            .iter()
            .all(|p| p.interval.is_none() && p.stress_interval.is_none()));
    }

    #[test]
    fn test_day_one_band_is_clipped_to_price_limits() {
        // 日波动 5% 的高波动股：95% 压力档 2.5σ = ±12.5%，主板单日打不到
        let closes: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 100.0 } else { 105.0 })
            .collect();
        let base = *closes.last().unwrap();
        let mut preds = make_predictions(base, 0.0, 3);
        attach_prediction_intervals(&mut preds, &closes, base, DEFAULT_COVERAGE, Some("600519"));

        let stress_d1 = preds[0].stress_interval.as_ref().unwrap();
        assert!(
            stress_d1.upper_change_percent <= 10.0 + 1e-9,
            "第 1 天上界不该越过涨停: {}",
            stress_d1.upper_change_percent
        );
        assert!(
            stress_d1.lower_change_percent >= -10.0 - 1e-9,
            "第 1 天下界不该越过跌停: {}",
            stress_d1.lower_change_percent
        );

        // 第 2 天起是累计涨跌幅，不受单日限幅约束，必须仍然能超出 ±10%
        let stress_d2 = preds[1].stress_interval.as_ref().unwrap();
        assert!(
            stress_d2.upper_change_percent > 10.0,
            "多日累计不该被单日限幅截断: {}",
            stress_d2.upper_change_percent
        );
    }

    #[test]
    fn test_growth_board_gets_wider_limit() {
        let closes: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 100.0 } else { 105.0 })
            .collect();
        let base = *closes.last().unwrap();

        let mut main = make_predictions(base, 0.0, 1);
        attach_prediction_intervals(&mut main, &closes, base, DEFAULT_COVERAGE, Some("600519"));
        let mut growth = make_predictions(base, 0.0, 1);
        attach_prediction_intervals(&mut growth, &closes, base, DEFAULT_COVERAGE, Some("300750"));

        let main_up = main[0].stress_interval.as_ref().unwrap().upper_change_percent;
        let growth_up = growth[0].stress_interval.as_ref().unwrap().upper_change_percent;
        assert!(main_up <= 10.0 + 1e-9);
        assert!(growth_up > main_up, "创业板 ±20% 不该被截到主板的 ±10%");
    }

    #[test]
    fn test_no_code_means_no_clipping() {
        let closes: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 100.0 } else { 105.0 })
            .collect();
        let base = *closes.last().unwrap();
        let mut preds = make_predictions(base, 0.0, 1);
        attach_prediction_intervals(&mut preds, &closes, base, DEFAULT_COVERAGE, None);
        assert!(
            preds[0].stress_interval.as_ref().unwrap().upper_change_percent > 10.0,
            "没给代码时不该假设限幅"
        );
    }

    #[test]
    fn test_calibrated_quantile_tiers() {
        let (lo80, hi80) = calibrated_quantiles(0.80);
        let (lo90, hi90) = calibrated_quantiles(0.90);
        let (lo95, hi95) = calibrated_quantiles(0.95);
        // 档位越高带越宽
        assert!(lo95 < lo90 && lo90 < lo80);
        assert!(hi95 > hi90 && hi90 > hi80);
        // 上下不对称是实测结果：上侧尾更长
        for (lo, hi) in [(lo80, hi80), (lo90, hi90), (lo95, hi95)] {
            assert!(lo < 0.0 && hi > 0.0);
            assert!(hi > lo.abs(), "实测右偏，上侧分位应大于下侧绝对值: {lo} / {hi}");
        }
    }

    #[test]
    fn test_sigma_path_falls_back_without_enough_history() {
        let path = horizon_sigma_path(&[100.0, 101.0], 3);
        assert_eq!(path.len(), 3);
        for (i, s) in path.iter().enumerate() {
            assert!((s - FALLBACK_DAILY_VOL * ((i + 1) as f64).sqrt()).abs() < 1e-12);
        }
        assert!(horizon_sigma_path(&[100.0; 60], 0).is_empty());
        // 价格恒定时 EWMA 无解，也要退回兜底而不是给出 0 宽度的带
        let flat = horizon_sigma_path(&[100.0; 60], 2);
        assert_eq!(flat.len(), 2);
        assert!(flat.iter().all(|s| *s > 0.0), "带宽不能为 0");
    }

    #[test]
    fn test_sigma_path_is_mean_reverting_not_sqrt_d() {
        // 前 200 天低波动、最近 20 天高波动 → 当前 σ 远高于长期 σ，
        // 累计 σ 的增长必须慢于 √d，否则就是又回到了 IID 外推
        let mut closes = vec![100.0];
        for i in 0..200 {
            let r = if i % 2 == 0 { 0.004 } else { -0.004 };
            closes.push(closes.last().unwrap() * (1.0 + r));
        }
        for i in 0..20 {
            let r = if i % 2 == 0 { 0.05 } else { -0.05 };
            closes.push(closes.last().unwrap() * (1.0 + r));
        }
        let path = horizon_sigma_path(&closes, 20);
        let naive = path[0] * (20.0_f64).sqrt();
        assert!(
            path[19] < naive,
            "高波动段的累计 σ 应低于 √d 外推: {} vs {naive}",
            path[19]
        );
        assert!(path.windows(2).all(|w| w[1] > w[0]), "累计 σ 仍必须随 horizon 单调变宽");
    }
}
