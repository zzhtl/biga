//! 基于可观测事实的预测风险告警。
//!
//! 本模块只描述当前数据、波动、趋势和模型状态，不输出未经校准的风险概率或综合分。

use crate::prediction::analysis::divergence::DivergenceAnalysis;
use crate::prediction::analysis::market_regime::{MarketRegime, MarketRegimeAnalysis};
use crate::prediction::analysis::signal_confirmation::{
    ConfirmationLevel, SignalConfirmationResult,
};
use crate::prediction::analysis::support_resistance::SupportResistance;
use crate::prediction::analysis::volatility_forecast::{
    VolatilityForecast, VolatilityTrend,
};
use crate::prediction::analysis::volume::VolumePriceSignal;
use crate::prediction::indicators::TechnicalIndicatorValues;
use crate::prediction::types::{
    Prediction, RiskCategory, RiskLevel, RiskMetrics, RiskSummary, RiskWarning,
};

/// 模型自身可用于风险提示的样本外元数据。
#[derive(Debug, Clone, Copy)]
pub struct ModelRiskInput {
    pub direction_accuracy: f64,
    pub test_samples: Option<usize>,
}

/// 构建风险汇总所需的既有分析结果。
pub struct RiskAnalysisInput<'a> {
    pub history_samples: usize,
    pub current_price: f64,
    pub regime: &'a MarketRegimeAnalysis,
    pub volatility_forecast: &'a VolatilityForecast,
    pub signal_confirmation: &'a SignalConfirmationResult,
    pub divergence: &'a DivergenceAnalysis,
    pub support_resistance: &'a SupportResistance,
    pub indicators: &'a TechnicalIndicatorValues,
    pub volume_signal: &'a VolumePriceSignal,
    pub predictions: &'a [Prediction],
    pub model: Option<ModelRiskInput>,
}

/// 生成结构化风险汇总。严重度代表规则触发级别，不是发生概率。
pub fn analyze_prediction_risk(input: RiskAnalysisInput<'_>) -> RiskSummary {
    let support_distance = nearest_distance_below(
        input.current_price,
        &input.support_resistance.support_levels,
    );
    let resistance_distance = nearest_distance_above(
        input.current_price,
        &input.support_resistance.resistance_levels,
    );
    let atr_percent = (input.current_price > 0.0 && input.indicators.atr > 0.0)
        .then_some(input.indicators.atr / input.current_price * 100.0);
    let last_prediction = input.predictions.last();
    let interval_80_width = last_prediction
        .and_then(|prediction| prediction.interval.as_ref())
        .map(|interval| interval.upper_change_percent - interval.lower_change_percent);
    let interval_80_lower = last_prediction
        .and_then(|prediction| prediction.interval.as_ref())
        .map(|interval| interval.lower_change_percent);
    let stress_95_lower = last_prediction
        .and_then(|prediction| prediction.stress_interval.as_ref())
        .map(|interval| interval.lower_change_percent);

    let mut summary = RiskSummary {
        level: RiskLevel::Low,
        level_label: RiskLevel::Low.label().to_string(),
        warnings: Vec::new(),
        metrics: RiskMetrics {
            history_samples: input.history_samples,
            data_staleness_days: None,
            daily_volatility_percent: input.volatility_forecast.current_volatility * 100.0,
            volatility_percentile: input.regime.volatility_percentile,
            interval_80_width_percent: interval_80_width,
            interval_80_lower_percent: interval_80_lower,
            stress_95_lower_percent: stress_95_lower,
            support_distance_percent: support_distance,
            resistance_distance_percent: resistance_distance,
            atr_percent,
        },
    };

    if input.history_samples < 120 {
        push_warning(
            &mut summary,
            warning(
                "DATA_SHORT_HISTORY",
                RiskCategory::Data,
                RiskLevel::Medium,
                "历史样本偏少",
                "风险与波动状态可能不稳定，建议补足历史数据后重新分析。",
                vec![format!("当前仅 {} 根K线，建议至少 120 根", input.history_samples)],
            ),
        );
    }

    if let Some(item) = volatility_warning(input.regime.volatility_percentile) {
        push_warning(&mut summary, item);
    }
    if input.regime.volatility_percentile >= 65.0
        && matches!(
            input.volatility_forecast.volatility_trend,
            VolatilityTrend::Expanding
        )
    {
        push_warning(
            &mut summary,
            warning(
                "VOLATILITY_EXPANDING",
                RiskCategory::Volatility,
                RiskLevel::Medium,
                "波动率正在扩张",
                "近期波动高于前一阶段，价格区间可能继续放大。",
                vec![format!(
                    "当前日波动 {:.2}%，历史百分位 {:.0}%",
                    input.volatility_forecast.current_volatility * 100.0,
                    input.regime.volatility_percentile
                )],
            ),
        );
    }

    if let Some(item) = downside_uncertainty_warning(
        interval_80_lower,
        stress_95_lower,
        input.predictions.len(),
    ) {
        push_warning(&mut summary, item);
    }

    if input.signal_confirmation.is_potential_false_signal
        || matches!(
            input.signal_confirmation.confirmation_level,
            ConfirmationLevel::Invalid
        )
    {
        push_warning(
            &mut summary,
            warning(
                "SIGNAL_CONFLICT",
                RiskCategory::Signal,
                RiskLevel::Medium,
                "技术信号相互冲突",
                "当前信号一致性不足，不应依据单一指标推断方向。",
                vec![
                    format!("冲突级别：{}", input.signal_confirmation.conflict_level),
                    format!(
                        "一致性：{:.0}%",
                        input.signal_confirmation.consistency_score * 100.0
                    ),
                ],
            ),
        );
    }

    match input.regime.regime {
        MarketRegime::StrongDowntrend => {
            let severity = if input.regime.volatility_percentile >= 85.0 {
                RiskLevel::High
            } else {
                RiskLevel::Medium
            };
            push_warning(
                &mut summary,
                warning(
                    "TREND_STRONG_DOWNTREND",
                    RiskCategory::Trend,
                    severity,
                    "当前处于强下跌状态",
                    "该告警描述已发生的趋势状态，不代表未来方向概率。",
                    vec![format!(
                        "趋势强度 {:.0}%，ADX {:.1}",
                        input.regime.trend_strength.abs() * 100.0,
                        input.regime.adx_value
                    )],
                ),
            );
        }
        MarketRegime::PotentialTop => push_warning(
            &mut summary,
            warning(
                "TREND_POTENTIAL_TOP",
                RiskCategory::Trend,
                RiskLevel::Medium,
                "技术状态接近潜在顶部",
                "动量与趋势指标出现顶部状态组合，需关注波动回撤。",
                vec![input.regime.description.clone()],
            ),
        ),
        _ => {}
    }

    if input.divergence.has_divergence && input.divergence.composite_score < -0.2 {
        let severity = if input.divergence.is_triple_divergence {
            RiskLevel::High
        } else {
            RiskLevel::Medium
        };
        push_warning(
            &mut summary,
            warning(
                "SIGNAL_BEARISH_DIVERGENCE",
                RiskCategory::Signal,
                severity,
                "检测到看跌背离",
                "价格与指标走势出现分歧，属于回撤风险信号而非确定性反转预测。",
                vec![format!(
                    "{}，背离数量 {}，综合得分 {:.2}",
                    input.divergence.primary_direction,
                    input.divergence.divergence_count,
                    input.divergence.composite_score
                )],
            ),
        );
    }

    let support_threshold = atr_percent.map_or(5.0, |atr| (atr * 2.0).max(5.0));
    if support_distance.is_none_or(|distance| distance > support_threshold) {
        let evidence = support_distance.map_or_else(
            || vec!["当前价格下方未识别到有效支撑".to_string()],
            |distance| {
                vec![format!(
                    "最近支撑距离 {:.2}%，警示阈值 {:.2}%",
                    distance, support_threshold
                )]
            },
        );
        push_warning(
            &mut summary,
            warning(
                "TREND_SUPPORT_GAP",
                RiskCategory::Trend,
                RiskLevel::Medium,
                "下方支撑缓冲不足",
                "可识别支撑距离较远，压力情景下缺少临近技术缓冲。",
                evidence,
            ),
        );
    }

    let volume_ratio = input.volume_signal.volume_ratio;
    if input.volume_signal.price_trend.contains("下跌") && volume_ratio >= 1.5 {
        let severity = if input.volume_signal.price_trend.contains("强势") && volume_ratio >= 2.0 {
            RiskLevel::High
        } else {
            RiskLevel::Medium
        };
        push_warning(
            &mut summary,
            warning(
                "LIQUIDITY_VOLUME_SELLING",
                RiskCategory::Liquidity,
                severity,
                "下跌伴随成交放大",
                "量价状态显示抛压增加，需关注流动性冲击。",
                vec![format!(
                    "{}，{}，量比 {:.2}",
                    input.volume_signal.price_trend,
                    input.volume_signal.volume_trend,
                    volume_ratio
                )],
            ),
        );
    }

    if let Some(model) = input.model {
        if model.direction_accuracy < 0.5 {
            push_warning(
                &mut summary,
                warning(
                    "MODEL_BELOW_RANDOM_BASELINE",
                    RiskCategory::Model,
                    RiskLevel::High,
                    "模型方向表现低于50%基线",
                    "该模型不应被解释为具有方向预测能力。",
                    vec![format!(
                        "记录的测试集方向准确率 {:.1}%",
                        model.direction_accuracy * 100.0
                    )],
                ),
            );
        }
        if model.test_samples.is_none_or(|samples| samples < 30) {
            push_warning(
                &mut summary,
                warning(
                    "MODEL_SMALL_TEST_SET",
                    RiskCategory::Model,
                    RiskLevel::Medium,
                    "模型测试样本不足",
                    "少量样本下的准确率波动较大，应以走步回测结果为准。",
                    vec![format!(
                        "测试样本 {}",
                        model
                            .test_samples
                            .map_or_else(|| "未知".to_string(), |samples| samples.to_string())
                    )],
                ),
            );
        }
    }

    finalize_summary(&mut summary);
    summary
}

/// 仅在实时预测入口追加数据新鲜度；历史走步回测不调用，避免把历史样本误判为陈旧。
pub fn add_data_staleness(summary: &mut RiskSummary, staleness_days: i64) {
    summary.metrics.data_staleness_days = Some(staleness_days.max(0));
    if staleness_days > 4 {
        let severity = if staleness_days > 10 {
            RiskLevel::High
        } else {
            RiskLevel::Medium
        };
        push_warning(
            summary,
            warning(
                "DATA_STALE",
                RiskCategory::Data,
                severity,
                "历史行情未及时更新",
                "分析基于较旧K线，当前市场状态可能已经变化。",
                vec![format!("距最新K线 {} 个自然日", staleness_days)],
            ),
        );
    }
    finalize_summary(summary);
}

fn volatility_warning(percentile: f64) -> Option<RiskWarning> {
    if percentile < 85.0 {
        return None;
    }
    let severity = if percentile >= 95.0 {
        RiskLevel::High
    } else {
        RiskLevel::Medium
    };
    Some(warning(
        "VOLATILITY_HIGH",
        RiskCategory::Volatility,
        severity,
        "当前波动率处于历史高位",
        "价格区间可能明显放大，应以区间而非点估计作为主要参考。",
        vec![format!("历史波动率百分位 {:.0}%", percentile)],
    ))
}

fn downside_uncertainty_warning(
    interval_80_lower: Option<f64>,
    stress_95_lower: Option<f64>,
    prediction_days: usize,
) -> Option<RiskWarning> {
    let severity = if interval_80_lower.is_some_and(|lower| lower <= -15.0) {
        RiskLevel::High
    } else if interval_80_lower.is_some_and(|lower| lower <= -8.0)
        || stress_95_lower.is_some_and(|lower| lower <= -15.0)
    {
        RiskLevel::Medium
    } else {
        return None;
    };

    let prediction_days = prediction_days.max(1);
    let mut evidence = Vec::new();
    if let Some(lower) = interval_80_lower {
        evidence.push(format!(
            "{}日80%区间下沿 {:+.2}%",
            prediction_days, lower
        ));
    }
    if let Some(lower) = stress_95_lower {
        evidence.push(format!(
            "{}日95%压力下沿 {:+.2}%",
            prediction_days, lower
        ));
    }

    Some(warning(
        "UNCERTAINTY_STRESS_DOWNSIDE",
        RiskCategory::Uncertainty,
        severity,
        "预测区间下行空间较大",
        "80%校准区间用于风险分级；95%压力区间仅提示低概率尾部情景，不单独判为高风险。",
        evidence,
    ))
}

fn nearest_distance_below(current_price: f64, levels: &[f64]) -> Option<f64> {
    if current_price <= 0.0 {
        return None;
    }
    levels
        .iter()
        .copied()
        .filter(|level| *level > 0.0 && *level < current_price)
        .max_by(f64::total_cmp)
        .map(|level| (current_price - level) / current_price * 100.0)
}

fn nearest_distance_above(current_price: f64, levels: &[f64]) -> Option<f64> {
    if current_price <= 0.0 {
        return None;
    }
    levels
        .iter()
        .copied()
        .filter(|level| *level > current_price)
        .min_by(f64::total_cmp)
        .map(|level| (level - current_price) / current_price * 100.0)
}

fn warning(
    code: &str,
    category: RiskCategory,
    severity: RiskLevel,
    title: &str,
    detail: &str,
    evidence: Vec<String>,
) -> RiskWarning {
    RiskWarning {
        code: code.to_string(),
        category,
        severity,
        title: title.to_string(),
        detail: detail.to_string(),
        evidence,
    }
}

fn push_warning(summary: &mut RiskSummary, item: RiskWarning) {
    if !summary.warnings.iter().any(|warning| warning.code == item.code) {
        summary.warnings.push(item);
    }
}

fn finalize_summary(summary: &mut RiskSummary) {
    summary
        .warnings
        .sort_by(|a, b| b.severity.cmp(&a.severity).then_with(|| a.code.cmp(&b.code)));
    summary.level = summary
        .warnings
        .iter()
        .map(|warning| warning.severity)
        .max()
        .unwrap_or(RiskLevel::Low);
    summary.level_label = summary.level.label().to_string();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_summary_has_consistent_low_risk_label() {
        let summary = RiskSummary::default();
        assert_eq!(summary.level, RiskLevel::Low);
        assert_eq!(summary.level_label, "低风险");
    }

    #[test]
    fn volatility_thresholds_are_stable() {
        assert!(volatility_warning(84.9).is_none());
        assert_eq!(volatility_warning(85.0).unwrap().severity, RiskLevel::Medium);
        assert_eq!(volatility_warning(95.0).unwrap().severity, RiskLevel::High);
    }

    #[test]
    fn calibrated_downside_uses_the_primary_interval_for_high_risk() {
        assert!(downside_uncertainty_warning(Some(-7.99), Some(-14.99), 5).is_none());

        let tail_only = downside_uncertainty_warning(Some(-7.0), Some(-15.0), 5).unwrap();
        assert_eq!(tail_only.severity, RiskLevel::Medium);

        for (interval_80, stress_95) in [(-8.74, -17.31), (-10.42, -20.01), (-12.83, -24.56)] {
            assert_eq!(
                downside_uncertainty_warning(Some(interval_80), Some(stress_95), 5)
                    .unwrap()
                    .severity,
                RiskLevel::Medium,
                "95%尾部压力不应单独把整票判为高风险",
            );
        }

        let high = downside_uncertainty_warning(Some(-15.0), Some(-25.0), 5).unwrap();
        assert_eq!(high.severity, RiskLevel::High);
        assert!(high.evidence.iter().any(|item| item.contains("5日80%区间下沿")));
        assert!(high.evidence.iter().any(|item| item.contains("5日95%压力下沿")));
    }

    #[test]
    fn staleness_uses_highest_triggered_severity() {
        let mut summary = RiskSummary::default();
        add_data_staleness(&mut summary, 5);
        assert_eq!(summary.level, RiskLevel::Medium);
        assert_eq!(summary.warnings.len(), 1);

        let mut stale = RiskSummary::default();
        add_data_staleness(&mut stale, 11);
        assert_eq!(stale.level, RiskLevel::High);
        assert_eq!(stale.metrics.data_staleness_days, Some(11));
    }

    #[test]
    fn nearest_level_distances_use_the_closest_valid_level() {
        assert_eq!(nearest_distance_below(100.0, &[80.0, 95.0, 101.0]), Some(5.0));
        assert_eq!(nearest_distance_above(100.0, &[99.0, 108.0, 103.0]), Some(3.0));
    }
}
