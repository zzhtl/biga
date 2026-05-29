//! 调优测量：对多只股票做走步回测，输出聚合方向准确率 / 朴素基准 / 超额(edge)。
//! 用法：cargo run --example tune_backtest -- --symbols=8 --horizons=1,5 --step=20 --diagnostics

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::{get_recent_historical_data_for_symbols, get_symbols_with_min_bars};
use biga_lib::prediction::backtest::metrics::{compute_metrics, BacktestSample};
use biga_lib::prediction::backtest::{run_backtest, BacktestObservation};
use biga_lib::prediction::cross_section::{daily_bias_from_rank, walk_forward_rank_signals};
use biga_lib::prediction::model::inference::MAX_ANALYSIS_DAYS;
use biga_lib::prediction::strategy::professional_engine::get_stock_price_limits;
use chrono::NaiveDate;
use std::collections::HashMap;

type BiasMap = HashMap<(NaiveDate, String), f64>;
const ONE_DAY_MIN_EMPIRICAL_SAMPLES: usize = 20;
const ONE_DAY_EMPIRICAL_WINDOW: usize = 500;
const ONE_DAY_STRONG_UP_RATE: f64 = 0.55;
const ONE_DAY_NEUTRAL_UP_RATE: f64 = 0.5;
const ONE_DAY_MA5_OVERSOLD_GAP: f64 = -1.13;
const ONE_DAY_MA5_OVERBOUGHT_GAP: f64 = 1.13;
const MA5_REVERSION_THRESHOLDS: [f64; 5] = [1.13, 1.5, 2.0, 3.0, 4.0];
const CONFIDENCE_FLIP_THRESHOLDS: [f64; 4] = [0.35, 0.40, 0.45, 0.50];

#[derive(Debug, Clone)]
struct TuneConfig {
    max_symbols: usize,
    horizons: Vec<usize>,
    backtest_step: usize,
    show_diagnostics: bool,
}

impl Default for TuneConfig {
    fn default() -> Self {
        Self {
            max_symbols: 25,
            horizons: vec![1, 5, 7, 10],
            backtest_step: 10,
            show_diagnostics: false,
        }
    }
}

impl TuneConfig {
    fn from_args(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut config = Self::default();
        for arg in args {
            if let Some(value) = arg.strip_prefix("--symbols=") {
                config.max_symbols = parse_positive_usize("symbols", value)?;
            } else if let Some(value) = arg.strip_prefix("--horizons=") {
                config.horizons = parse_horizons(value)?;
            } else if let Some(value) = arg.strip_prefix("--step=") {
                config.backtest_step = parse_positive_usize("step", value)?;
            } else if arg == "--diagnostics" {
                config.show_diagnostics = true;
            } else {
                return Err(format!("未知参数 `{arg}`"));
            }
        }
        Ok(config)
    }
}

fn parse_positive_usize(name: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|e| format!("{name} 不是有效整数: {e}"))?;
    if parsed == 0 {
        return Err(format!("{name} 必须大于0"));
    }
    Ok(parsed)
}

fn parse_horizons(value: &str) -> Result<Vec<usize>, String> {
    let horizons = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize("horizons", part))
        .collect::<Result<Vec<_>, _>>()?;
    if horizons.is_empty() {
        return Err("horizons 至少需要一个周期".to_string());
    }
    Ok(horizons)
}

#[tokio::main]
async fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print_usage();
        return;
    }
    let config = match TuneConfig::from_args(args) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("参数错误: {err}");
            print_usage();
            std::process::exit(2);
        }
    };
    let pool = create_pool().await.expect("创建连接池失败");

    let symbols = get_symbols_with_min_bars(200, &pool)
        .await
        .expect("查询股票失败");

    let data = get_recent_historical_data_for_symbols(&symbols, 3000, &pool)
        .await
        .expect("查询历史数据失败")
        .into_iter()
        .filter(|(_, hist)| hist.len() >= 120)
        .collect::<Vec<_>>();
    let eval_symbols = data
        .iter()
        .take(config.max_symbols)
        .map(|(symbol, _)| symbol.as_str())
        .collect::<Vec<_>>();

    println!("参与截面股票数：{}，参与端到端回测股票数：{}", data.len(), eval_symbols.len());

    for horizon in config.horizons.iter().copied() {
        let show_diagnostics = diagnostics_enabled_for_horizon(&config, horizon);
        let track_cross_hits = horizon == 5;
        let cross_bias = if horizon == 5 {
            cross_section_bias_by_symbol_date(&data, horizon, 120)
        } else {
            HashMap::new()
        };
        let mut acc = 0.0;
        let mut base = 0.0;
        let mut edge = 0.0;
        let mut pred_up = 0.0;
        let mut actual_up = 0.0;
        let mut cross_acc = 0.0;
        let mut cross_edge = 0.0;
        let mut cross_pred_up = 0.0;
        let mut cross_hits = 0usize;
        let mut cross_total = 0usize;
        let mut candidate_acc = 0.0;
        let mut candidate_edge = 0.0;
        let mut candidate_pred_up = 0.0;
        let mut pooled_samples = Vec::new();
        let mut pooled_cross_samples = Vec::new();
        let mut pooled_candidate_samples = Vec::new();
        let mut ma5_reversion_sweeps = MA5_REVERSION_THRESHOLDS
            .iter()
            .map(|threshold| Ma5ReversionSweep {
                threshold: *threshold,
                all_samples: Vec::new(),
                oversold_samples: Vec::new(),
                overbought_samples: Vec::new(),
            })
            .collect::<Vec<_>>();
        let mut confidence_flip_sweeps = CONFIDENCE_FLIP_THRESHOLDS
            .iter()
            .map(|threshold| ConfidenceFlipSweep {
                threshold: *threshold,
                all_sources_samples: Vec::new(),
                no_calibration_samples: Vec::new(),
                affected_all_sources: 0,
                affected_no_calibration: 0,
            })
            .collect::<Vec<_>>();
        let mut skipped_bullish_sweeps = one_day_skipped_bullish_sweeps();
        let mut ma5_bullish_threshold_sweeps = one_day_ma5_bullish_threshold_sweeps();
        let mut seven_mid_reversal_sweep = SevenDayMidReversalSweep::default();
        let cross_sweeps = if cross_sweep_enabled(show_diagnostics, horizon) {
            cross_bias_sweeps_by_symbol_date(&data, horizon, 120, &[0.005, 0.01, 0.02, 0.03])
        } else {
            Vec::new()
        };
        let mut cross_sweep_samples = cross_sweeps
            .iter()
            .map(|(magnitude, _)| (*magnitude, Vec::new()))
            .collect::<Vec<_>>();
        let mut per_symbol_reports = Vec::new();
        let mut confidence_buckets = diagnostic_buckets(&["conf<40", "40<=conf<60", "conf>=60"]);
        let mut magnitude_buckets = diagnostic_buckets(&["|pred|<0.5", "0.5<=|pred|<1.5", "|pred|>=1.5"]);
        let mut direction_buckets = diagnostic_buckets(&["pred_up", "pred_down_or_flat"]);
        let mut calibration_buckets = diagnostic_buckets(&[
            "feature_reversion",
            "empirical_rate",
            "both_calibration",
            "no_calibration",
        ]);
        let mut calibration_direction_buckets = diagnostic_buckets(&[
            "feature_up",
            "feature_down",
            "empirical_up",
            "empirical_down",
            "no_calibration",
            "calibration_unknown",
        ]);
        let mut strategy_buckets = diagnostic_buckets(&[
            "strategy_trend",
            "strategy_mean_reversion",
            "strategy_reversal",
            "strategy_unknown",
        ]);
        let mut magnitude_direction_buckets = diagnostic_buckets(&[
            "up |pred|<0.5",
            "up 0.5<=|pred|<1.5",
            "up |pred|>=1.5",
            "down |pred|<0.5",
            "down 0.5<=|pred|<1.5",
            "down |pred|>=1.5",
        ]);
        let mut n = 0;
        for (symbol, hist) in data
            .iter()
            .filter(|(symbol, _)| eval_symbols.iter().any(|eval| *eval == symbol))
        {
            if let Ok(rep) = run_backtest(symbol, hist, 60, horizon, config.backtest_step) {
                let m = &rep.metrics;
                acc += m.direction_accuracy;
                base += m.baseline_accuracy;
                edge += m.edge();
                pred_up += m.predicted_up_ratio;
                actual_up += m.actual_up_ratio;
                if show_diagnostics {
                    pooled_samples.extend(rep.observations.iter().map(|observation| {
                        BacktestSample {
                            predicted_change: observation.predicted_change,
                            actual_change: observation.actual_change,
                        }
                    }));
                    per_symbol_reports.push((
                        symbol.clone(),
                        m.direction_accuracy,
                        m.baseline_accuracy,
                        m.predicted_up_ratio,
                        m.actual_up_ratio,
                    ));
                    for observation in &rep.observations {
                        let sample = BacktestSample {
                            predicted_change: observation.predicted_change,
                            actual_change: observation.actual_change,
                        };
                        push_bucket(
                            &mut confidence_buckets,
                            confidence_bucket_idx(observation.confidence),
                            sample,
                        );
                        push_bucket(
                            &mut magnitude_buckets,
                            magnitude_bucket_idx(observation.predicted_change),
                            sample,
                        );
                        push_bucket(
                            &mut direction_buckets,
                            usize::from(observation.predicted_change <= 0.0),
                            sample,
                        );
                        push_bucket(
                            &mut calibration_buckets,
                            calibration_bucket_idx(&observation.key_factors),
                            sample,
                        );
                        push_bucket(
                            &mut calibration_direction_buckets,
                            calibration_direction_bucket_idx(&observation.key_factors),
                            sample,
                        );
                        push_bucket(
                            &mut strategy_buckets,
                            strategy_bucket_idx(observation.prediction_reason.as_deref()),
                            sample,
                        );
                        push_bucket(
                            &mut magnitude_direction_buckets,
                            magnitude_direction_bucket_idx(observation.predicted_change),
                            sample,
                        );
                        if horizon == 1 {
                            push_ma5_reversion_sweeps(
                                &mut ma5_reversion_sweeps,
                                hist,
                                observation.prediction_date,
                                observation.actual_change,
                            );
                            push_confidence_flip_sweeps(
                                &mut confidence_flip_sweeps,
                                observation.predicted_change,
                                observation.actual_change,
                                observation.confidence,
                                calibration_bucket_idx(&observation.key_factors) == 3,
                            );
                            push_one_day_skipped_bullish_sweeps(
                                &mut skipped_bullish_sweeps,
                                hist,
                                observation,
                            );
                            push_one_day_ma5_bullish_threshold_sweeps(
                                &mut ma5_bullish_threshold_sweeps,
                                hist,
                                observation,
                            );
                        }
                        if horizon == 7 {
                            push_seven_day_mid_reversal_sweep(
                                &mut seven_mid_reversal_sweep,
                                observation,
                            );
                        }
                    }
                }

                let adjusted = rep
                    .observations
                    .iter()
                    .map(|observation| {
                        let daily_bias = cross_bias
                            .get(&(observation.prediction_date, symbol.clone()))
                            .copied()
                            .unwrap_or(0.0);
                        if track_cross_hits
                            && daily_bias != 0.0
                            && cross_section_bias_aligns_with_prediction(
                                &observation.predicted_daily_changes,
                                daily_bias,
                            )
                        {
                            cross_hits += 1;
                        }
                        if track_cross_hits {
                            cross_total += 1;
                        }
                        BacktestSample {
                            predicted_change: apply_daily_bias_to_observation(
                                observation.base_price,
                                &observation.predicted_daily_changes,
                                daily_bias,
                                symbol,
                            ),
                            actual_change: observation.actual_change,
                        }
                    })
                    .collect::<Vec<_>>();
                if horizon == 5 {
                    for ((_, sweep_bias), (_, sweep_samples)) in
                        cross_sweeps.iter().zip(cross_sweep_samples.iter_mut())
                    {
                        for observation in &rep.observations {
                            let daily_bias = sweep_bias
                                .get(&(observation.prediction_date, symbol.clone()))
                                .copied()
                                .unwrap_or(0.0);
                            sweep_samples.push(BacktestSample {
                                predicted_change: apply_daily_bias_to_observation(
                                    observation.base_price,
                                    &observation.predicted_daily_changes,
                                    daily_bias,
                                    symbol,
                                ),
                                actual_change: observation.actual_change,
                            });
                        }
                    }
                }
                let cm = compute_metrics(&adjusted);
                if show_diagnostics {
                    pooled_cross_samples.extend(adjusted.iter().copied());
                }
                cross_acc += cm.direction_accuracy;
                cross_edge += cm.edge();
                cross_pred_up += cm.predicted_up_ratio;

                if horizon == 1 {
                    let candidate = rep
                        .observations
                        .iter()
                        .map(|observation| BacktestSample {
                            predicted_change: candidate_change(
                                horizon,
                                observation.predicted_change,
                            ),
                            actual_change: observation.actual_change,
                        })
                        .collect::<Vec<_>>();
                    let candidate_metrics = compute_metrics(&candidate);
                    pooled_candidate_samples.extend(candidate.iter().copied());
                    candidate_acc += candidate_metrics.direction_accuracy;
                    candidate_edge += candidate_metrics.edge();
                    candidate_pred_up += candidate_metrics.predicted_up_ratio;
                }
                n += 1;
            }
        }
        if n > 0 {
            let n = n as f64;
            let hit_summary = cross_hit_summary(track_cross_hits, cross_hits, cross_total);
            println!(
                "horizon={horizon}: 引擎={:.1}% 基准={:.1}% 超额={:+.2}% 预测涨占比={:.0}% 真实涨占比={:.0}% | 截面后={:.1}% 超额={:+.2}% 预测涨占比={:.0}% 命中={}（{} 只）",
                acc / n * 100.0,
                base / n * 100.0,
                edge / n * 100.0,
                pred_up / n * 100.0,
                actual_up / n * 100.0,
                cross_acc / n * 100.0,
                cross_edge / n * 100.0,
                cross_pred_up / n * 100.0,
                hit_summary,
                n as usize
            );
            if show_diagnostics {
                print_pooled_metrics("pooled原始", &pooled_samples);
                print_pooled_metrics("pooled截面", &pooled_cross_samples);
                if horizon == 1 {
                    print_pooled_metrics("pooled 1日小幅看涨反向候选", &pooled_candidate_samples);
                    println!(
                        "  {horizon}日反向候选: acc {:.1}% 超额 {:+.2}% 预测涨占比 {:.0}%",
                        candidate_acc / n * 100.0,
                        candidate_edge / n * 100.0,
                        candidate_pred_up / n * 100.0
                    );
                }
                print_horizon_symbol_diagnostics(horizon, &per_symbol_reports);
                print_diagnostic_buckets("置信度分桶", &confidence_buckets);
                print_diagnostic_buckets("预测幅度分桶", &magnitude_buckets);
                print_diagnostic_buckets("预测方向分桶", &direction_buckets);
                print_diagnostic_buckets("校准来源分桶", &calibration_buckets);
                print_diagnostic_buckets("校准方向分桶", &calibration_direction_buckets);
                print_diagnostic_buckets("策略来源分桶", &strategy_buckets);
                print_diagnostic_buckets("幅度方向交叉分桶", &magnitude_direction_buckets);
                if horizon == 1 {
                    print_ma5_reversion_sweeps(&ma5_reversion_sweeps);
                    print_confidence_flip_sweeps(&confidence_flip_sweeps);
                    print_one_day_skipped_bullish_sweeps(&skipped_bullish_sweeps);
                    print_one_day_ma5_bullish_threshold_sweeps(
                        &ma5_bullish_threshold_sweeps,
                    );
                }
                if horizon == 7 {
                    print_seven_day_mid_reversal_sweep(&seven_mid_reversal_sweep);
                }
            }
            if cross_sweep_enabled(show_diagnostics, horizon) {
                print_cross_sweep_metrics(&cross_sweep_samples);
            }
        }
    }
}

fn diagnostics_enabled_for_horizon(config: &TuneConfig, _horizon: usize) -> bool {
    config.show_diagnostics
}

fn cross_sweep_enabled(show_diagnostics: bool, horizon: usize) -> bool {
    show_diagnostics && horizon == 5
}

fn cross_hit_summary(enabled: bool, hits: usize, total: usize) -> String {
    if enabled {
        format!("{hits}/{total}")
    } else {
        "-/-".to_string()
    }
}

fn print_usage() {
    eprintln!("用法: cargo run --example tune_backtest -- --symbols=8 --horizons=1,5 --step=20 --diagnostics");
}

fn print_pooled_metrics(title: &str, samples: &[BacktestSample]) {
    let metrics = compute_metrics(samples);
    println!(
        "  {title}: acc {:.1}% base {:.1}% edge {:+.2}% pred_up {:.0}% actual_up {:.0}% n {}",
        metrics.direction_accuracy * 100.0,
        metrics.baseline_accuracy * 100.0,
        metrics.edge() * 100.0,
        metrics.predicted_up_ratio * 100.0,
        metrics.actual_up_ratio * 100.0,
        metrics.total
    );
}

fn candidate_change(horizon: usize, predicted_change: f64) -> f64 {
    let magnitude = predicted_change.abs();
    let should_flip = horizon == 1 && predicted_change > 0.0 && magnitude < 0.5;
    if should_flip {
        -predicted_change
    } else {
        predicted_change
    }
}

struct DiagnosticBucket {
    label: &'static str,
    samples: Vec<BacktestSample>,
}

struct Ma5ReversionSweep {
    threshold: f64,
    all_samples: Vec<BacktestSample>,
    oversold_samples: Vec<BacktestSample>,
    overbought_samples: Vec<BacktestSample>,
}

struct ConfidenceFlipSweep {
    threshold: f64,
    all_sources_samples: Vec<BacktestSample>,
    no_calibration_samples: Vec<BacktestSample>,
    affected_all_sources: usize,
    affected_no_calibration: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OneDayBullishSignalKind {
    FeatureMa5,
    StrongEmpirical,
    WeakEmpirical,
}

struct OneDayBullishSignal {
    kind: OneDayBullishSignalKind,
    calibrated_change: f64,
}

struct OneDayEmpiricalStats {
    actual_up_ratio: f64,
    average_actual_change: f64,
    average_abs_change: f64,
}

struct OneDaySkippedBullishSweep {
    label: &'static str,
    kind: Option<OneDayBullishSignalKind>,
    all_samples: Vec<BacktestSample>,
    signal_samples: Vec<BacktestSample>,
    matched: usize,
    direction_flips: usize,
}

struct OneDayMa5BullishThresholdSweep {
    threshold: f64,
    all_samples: Vec<BacktestSample>,
    signal_samples: Vec<BacktestSample>,
    matched: usize,
    direction_flips: usize,
}

#[derive(Default)]
struct SevenDayMidReversalSweep {
    production_samples: Vec<BacktestSample>,
    undo_samples: Vec<BacktestSample>,
    affected_production_samples: Vec<BacktestSample>,
    affected_undo_samples: Vec<BacktestSample>,
    matched: usize,
}

fn diagnostic_buckets(labels: &[&'static str]) -> Vec<DiagnosticBucket> {
    labels
        .iter()
        .map(|label| DiagnosticBucket {
            label,
            samples: Vec::new(),
        })
        .collect()
}

fn push_bucket(buckets: &mut [DiagnosticBucket], idx: usize, sample: BacktestSample) {
    if let Some(bucket) = buckets.get_mut(idx) {
        bucket.samples.push(sample);
    }
}

fn confidence_bucket_idx(confidence: f64) -> usize {
    if confidence < 0.40 {
        0
    } else if confidence < 0.60 {
        1
    } else {
        2
    }
}

fn magnitude_bucket_idx(predicted_change: f64) -> usize {
    let magnitude = predicted_change.abs();
    if magnitude < 0.5 {
        0
    } else if magnitude < 1.5 {
        1
    } else {
        2
    }
}

fn magnitude_direction_bucket_idx(predicted_change: f64) -> usize {
    let direction_offset = if predicted_change > 0.0 { 0 } else { 3 };
    direction_offset + magnitude_bucket_idx(predicted_change)
}

fn calibration_bucket_idx(key_factors: &[String]) -> usize {
    let has_feature = key_factors
        .iter()
        .any(|factor| factor.contains("均值回归特征校准"));
    let has_empirical = key_factors
        .iter()
        .any(|factor| factor.contains("已按真实基率校准")
            || factor.contains("已按短周期高基率反向校准")
            || factor.contains("已按1日弱基率校准")
            || factor.contains("已按5日弱基率校准")
            || factor.contains("已按7日高基率反向校准")
            || factor.contains("已按7日中幅方向反向校准")
            || factor.contains("已按7日反转策略偏多校准")
            || factor.contains("已按长周期弱基率校准"));

    match (has_feature, has_empirical) {
        (true, false) => 0,
        (false, true) => 1,
        (true, true) => 2,
        (false, false) => 3,
    }
}

fn calibration_direction_bucket_idx(key_factors: &[String]) -> usize {
    let has_feature = key_factors
        .iter()
        .any(|factor| factor.contains("均值回归特征校准"));
    if has_feature {
        return if key_factors
            .iter()
            .any(|factor| factor.contains("特征偏多"))
        {
            0
        } else if key_factors
            .iter()
            .any(|factor| factor.contains("特征偏空"))
        {
            1
        } else {
            5
        };
    }

    let has_empirical = key_factors
        .iter()
        .any(|factor| factor.contains("已按真实基率校准")
            || factor.contains("已按短周期高基率反向校准")
            || factor.contains("已按1日弱基率校准")
            || factor.contains("已按5日弱基率校准")
            || factor.contains("已按7日高基率反向校准")
            || factor.contains("已按7日中幅方向反向校准")
            || factor.contains("已按7日反转策略偏多校准")
            || factor.contains("已按长周期弱基率校准"));
    if !has_empirical {
        return 4;
    }
    if key_factors
        .iter()
        .any(|factor| factor.contains("上涨基率达到")
            || factor.contains("反向为看涨")
            || factor.contains("反转策略偏多")
            || factor.contains("上涨基率略高于5日")
            || factor.contains("上涨基率略高于"))
    {
        2
    } else if key_factors
        .iter()
        .any(|factor| factor.contains("上涨基率低于")
            || factor.contains("反向为看跌")
            || factor.contains("上涨基率高于短周期过热")
            || factor.contains("上涨基率高于7日过热")
            || factor.contains("上涨基率略低于5日")
            || factor.contains("上涨基率略低于"))
    {
        3
    } else {
        5
    }
}

fn strategy_bucket_idx(prediction_reason: Option<&str>) -> usize {
    let Some(reason) = prediction_reason else {
        return 3;
    };
    if reason.contains("策略:趋势跟踪") {
        0
    } else if reason.contains("策略:均值回归") {
        1
    } else if reason.contains("策略:反转策略") {
        2
    } else {
        3
    }
}

fn print_diagnostic_buckets(title: &str, buckets: &[DiagnosticBucket]) {
    println!("  {title}:");
    for bucket in buckets {
        let metrics = compute_metrics(&bucket.samples);
        println!(
            "    {:<18} n {:<5} acc {:.1}% base {:.1}% edge {:+.1}% pred_up {:.0}% actual_up {:.0}%",
            bucket.label,
            metrics.total,
            metrics.direction_accuracy * 100.0,
            metrics.baseline_accuracy * 100.0,
            metrics.edge() * 100.0,
            metrics.predicted_up_ratio * 100.0,
            metrics.actual_up_ratio * 100.0
        );
    }
}

fn push_ma5_reversion_sweeps(
    sweeps: &mut [Ma5ReversionSweep],
    hist: &[biga_lib::db::models::HistoricalData],
    prediction_date: NaiveDate,
    actual_change: f64,
) {
    let Some(ma5_gap) = ma_gap_at_prediction_date(hist, prediction_date, 5) else {
        return;
    };
    for sweep in sweeps {
        if ma5_gap.abs() < sweep.threshold {
            continue;
        }
        let predicted_change = if ma5_gap < 0.0 { 1.0 } else { -1.0 };
        let sample = BacktestSample {
            predicted_change,
            actual_change,
        };
        sweep.all_samples.push(sample);
        if ma5_gap < 0.0 {
            sweep.oversold_samples.push(sample);
        } else {
            sweep.overbought_samples.push(sample);
        }
    }
}

fn ma_gap_at_prediction_date(
    hist: &[biga_lib::db::models::HistoricalData],
    prediction_date: NaiveDate,
    window: usize,
) -> Option<f64> {
    if window == 0 {
        return None;
    }
    let idx = hist.iter().position(|item| item.date == prediction_date)?;
    let end = idx + 1;
    if end < window {
        return None;
    }
    let start = end - window;
    let average = hist[start..end].iter().map(|item| item.close).sum::<f64>() / window as f64;
    let latest = hist[idx].close;
    if average <= 0.0 || !average.is_finite() || !latest.is_finite() {
        return None;
    }
    Some((latest - average) / average * 100.0)
}

fn print_ma5_reversion_sweeps(sweeps: &[Ma5ReversionSweep]) {
    println!("  MA5偏离均值回归候选:");
    for sweep in sweeps {
        let all_metrics = compute_metrics(&sweep.all_samples);
        let oversold_metrics = compute_metrics(&sweep.oversold_samples);
        let overbought_metrics = compute_metrics(&sweep.overbought_samples);
        println!(
            "    |gap|>={:.2}% all n {:<4} acc {:.1}% base {:.1}% edge {:+.1}% | oversold n {:<4} acc {:.1}% base {:.1}% | overbought n {:<4} acc {:.1}% base {:.1}%",
            sweep.threshold,
            all_metrics.total,
            all_metrics.direction_accuracy * 100.0,
            all_metrics.baseline_accuracy * 100.0,
            all_metrics.edge() * 100.0,
            oversold_metrics.total,
            oversold_metrics.direction_accuracy * 100.0,
            oversold_metrics.baseline_accuracy * 100.0,
            overbought_metrics.total,
            overbought_metrics.direction_accuracy * 100.0,
            overbought_metrics.baseline_accuracy * 100.0
        );
    }
}

fn push_confidence_flip_sweeps(
    sweeps: &mut [ConfidenceFlipSweep],
    predicted_change: f64,
    actual_change: f64,
    confidence: f64,
    no_calibration: bool,
) {
    for sweep in sweeps {
        let flip_all_sources =
            confidence >= sweep.threshold && predicted_change.abs() < 0.5;
        let flip_no_calibration = flip_all_sources && no_calibration;
        if flip_all_sources {
            sweep.affected_all_sources += 1;
        }
        if flip_no_calibration {
            sweep.affected_no_calibration += 1;
        }

        sweep.all_sources_samples.push(BacktestSample {
            predicted_change: if flip_all_sources {
                -predicted_change
            } else {
                predicted_change
            },
            actual_change,
        });
        sweep.no_calibration_samples.push(BacktestSample {
            predicted_change: if flip_no_calibration {
                -predicted_change
            } else {
                predicted_change
            },
            actual_change,
        });
    }
}

fn print_confidence_flip_sweeps(sweeps: &[ConfidenceFlipSweep]) {
    println!("  1日高置信小幅反向候选:");
    for sweep in sweeps {
        let all_metrics = compute_metrics(&sweep.all_sources_samples);
        let no_calibration_metrics = compute_metrics(&sweep.no_calibration_samples);
        println!(
            "    conf>={:.2} all flip {:<4} acc {:.1}% base {:.1}% edge {:+.2}% | no_cal flip {:<4} acc {:.1}% base {:.1}% edge {:+.2}%",
            sweep.threshold,
            sweep.affected_all_sources,
            all_metrics.direction_accuracy * 100.0,
            all_metrics.baseline_accuracy * 100.0,
            all_metrics.edge() * 100.0,
            sweep.affected_no_calibration,
            no_calibration_metrics.direction_accuracy * 100.0,
            no_calibration_metrics.baseline_accuracy * 100.0,
            no_calibration_metrics.edge() * 100.0
        );
    }
}

fn one_day_skipped_bullish_sweeps() -> Vec<OneDaySkippedBullishSweep> {
    [
        ("any_bullish", None),
        ("ma5_feature_up", Some(OneDayBullishSignalKind::FeatureMa5)),
        (
            "strong_empirical_up",
            Some(OneDayBullishSignalKind::StrongEmpirical),
        ),
        (
            "weak_empirical_up",
            Some(OneDayBullishSignalKind::WeakEmpirical),
        ),
    ]
    .into_iter()
    .map(|(label, kind)| OneDaySkippedBullishSweep {
        label,
        kind,
        all_samples: Vec::new(),
        signal_samples: Vec::new(),
        matched: 0,
        direction_flips: 0,
    })
    .collect()
}

fn push_one_day_skipped_bullish_sweeps(
    sweeps: &mut [OneDaySkippedBullishSweep],
    hist: &[HistoricalData],
    observation: &BacktestObservation,
) {
    let signal = one_day_skipped_bullish_signal(hist, observation.prediction_date);
    for sweep in sweeps {
        let calibrated_change = signal
            .as_ref()
            .filter(|signal| sweep.kind.is_none() || sweep.kind == Some(signal.kind))
            .map(|signal| signal.calibrated_change);
        let matches = calibrated_change.is_some();
        let predicted_change = calibrated_change.unwrap_or(observation.predicted_change);
        sweep.all_samples.push(BacktestSample {
            predicted_change,
            actual_change: observation.actual_change,
        });
        if matches {
            sweep.matched += 1;
            if observation.predicted_change <= 0.0 {
                sweep.direction_flips += 1;
            }
            sweep.signal_samples.push(BacktestSample {
                predicted_change,
                actual_change: observation.actual_change,
            });
        }
    }
}

fn one_day_skipped_bullish_signal(
    hist: &[HistoricalData],
    prediction_date: NaiveDate,
) -> Option<OneDayBullishSignal> {
    let visible = visible_history_at_prediction_date(hist, prediction_date)?;
    let stats = one_day_empirical_stats(visible)?;
    let calibrated_change = stats.baseline_magnitude();
    let ma5_gap = ma_gap_at_prediction_date(hist, prediction_date, 5)?;

    if ma5_gap <= ONE_DAY_MA5_OVERSOLD_GAP {
        return Some(OneDayBullishSignal {
            kind: OneDayBullishSignalKind::FeatureMa5,
            calibrated_change,
        });
    }
    if ma5_gap >= ONE_DAY_MA5_OVERBOUGHT_GAP {
        return None;
    }

    const EPS: f64 = 1e-12;
    if stats.actual_up_ratio + EPS >= ONE_DAY_STRONG_UP_RATE {
        Some(OneDayBullishSignal {
            kind: OneDayBullishSignalKind::StrongEmpirical,
            calibrated_change,
        })
    } else if stats.actual_up_ratio > ONE_DAY_NEUTRAL_UP_RATE + EPS {
        Some(OneDayBullishSignal {
            kind: OneDayBullishSignalKind::WeakEmpirical,
            calibrated_change,
        })
    } else {
        None
    }
}

fn visible_history_at_prediction_date(
    hist: &[HistoricalData],
    prediction_date: NaiveDate,
) -> Option<&[HistoricalData]> {
    let end = hist
        .iter()
        .position(|item| item.date == prediction_date)?
        .checked_add(1)?;
    let start = end.saturating_sub(MAX_ANALYSIS_DAYS.max(1));
    Some(&hist[start..end])
}

fn one_day_empirical_stats(hist: &[HistoricalData]) -> Option<OneDayEmpiricalStats> {
    if hist.len() <= ONE_DAY_MIN_EMPIRICAL_SAMPLES + 1 {
        return None;
    }

    let start = hist.len().saturating_sub(ONE_DAY_EMPIRICAL_WINDOW);
    let mut samples = 0usize;
    let mut up = 0usize;
    let mut sum = 0.0;
    let mut abs_sum = 0.0;
    for i in start..hist.len().saturating_sub(1) {
        let base = hist[i].close;
        let future = hist[i + 1].close;
        if base <= 0.0 || future <= 0.0 {
            continue;
        }
        let actual = (future - base) / base * 100.0;
        if actual.abs() < 0.01 || !actual.is_finite() {
            continue;
        }
        samples += 1;
        up += usize::from(actual > 0.0);
        sum += actual;
        abs_sum += actual.abs();
    }

    if samples < ONE_DAY_MIN_EMPIRICAL_SAMPLES {
        return None;
    }

    Some(OneDayEmpiricalStats {
        actual_up_ratio: up as f64 / samples as f64,
        average_actual_change: sum / samples as f64,
        average_abs_change: abs_sum / samples as f64,
    })
}

impl OneDayEmpiricalStats {
    fn baseline_magnitude(&self) -> f64 {
        self.average_actual_change
            .abs()
            .max(self.average_abs_change * 0.35)
            .clamp(0.2, 2.5)
    }
}

fn print_one_day_skipped_bullish_sweeps(sweeps: &[OneDaySkippedBullishSweep]) {
    println!("  1日被跳过偏多校准候选:");
    for sweep in sweeps {
        let all_metrics = compute_metrics(&sweep.all_samples);
        let signal_metrics = compute_metrics(&sweep.signal_samples);
        println!(
            "    {:<20} match {:<4} flip {:<4} all acc {:.1}% base {:.1}% edge {:+.2}% pred_up {:.0}% | signal acc {:.1}% base {:.1}% n {}",
            sweep.label,
            sweep.matched,
            sweep.direction_flips,
            all_metrics.direction_accuracy * 100.0,
            all_metrics.baseline_accuracy * 100.0,
            all_metrics.edge() * 100.0,
            all_metrics.predicted_up_ratio * 100.0,
            signal_metrics.direction_accuracy * 100.0,
            signal_metrics.baseline_accuracy * 100.0,
            signal_metrics.total
        );
    }
}

fn one_day_ma5_bullish_threshold_sweeps() -> Vec<OneDayMa5BullishThresholdSweep> {
    MA5_REVERSION_THRESHOLDS
        .iter()
        .map(|threshold| OneDayMa5BullishThresholdSweep {
            threshold: *threshold,
            all_samples: Vec::new(),
            signal_samples: Vec::new(),
            matched: 0,
            direction_flips: 0,
        })
        .collect()
}

fn push_one_day_ma5_bullish_threshold_sweeps(
    sweeps: &mut [OneDayMa5BullishThresholdSweep],
    hist: &[HistoricalData],
    observation: &BacktestObservation,
) {
    let ma5_gap = ma_gap_at_prediction_date(hist, observation.prediction_date, 5);
    let calibrated_change = visible_history_at_prediction_date(hist, observation.prediction_date)
        .and_then(one_day_empirical_stats)
        .map(|stats| stats.baseline_magnitude());

    for sweep in sweeps {
        let candidate_change = ma5_gap
            .filter(|gap| *gap <= -sweep.threshold)
            .and(calibrated_change);
        let matches = candidate_change.is_some();
        let predicted_change = candidate_change.unwrap_or(observation.predicted_change);
        sweep.all_samples.push(BacktestSample {
            predicted_change,
            actual_change: observation.actual_change,
        });
        if matches {
            sweep.matched += 1;
            if observation.predicted_change <= 0.0 {
                sweep.direction_flips += 1;
            }
            sweep.signal_samples.push(BacktestSample {
                predicted_change,
                actual_change: observation.actual_change,
            });
        }
    }
}

fn print_one_day_ma5_bullish_threshold_sweeps(
    sweeps: &[OneDayMa5BullishThresholdSweep],
) {
    println!("  1日MA5超跌偏多阈值候选:");
    for sweep in sweeps {
        let all_metrics = compute_metrics(&sweep.all_samples);
        let signal_metrics = compute_metrics(&sweep.signal_samples);
        println!(
            "    gap<=-{:.2}% match {:<4} flip {:<4} all acc {:.1}% base {:.1}% edge {:+.2}% pred_up {:.0}% | signal acc {:.1}% base {:.1}% n {}",
            sweep.threshold,
            sweep.matched,
            sweep.direction_flips,
            all_metrics.direction_accuracy * 100.0,
            all_metrics.baseline_accuracy * 100.0,
            all_metrics.edge() * 100.0,
            all_metrics.predicted_up_ratio * 100.0,
            signal_metrics.direction_accuracy * 100.0,
            signal_metrics.baseline_accuracy * 100.0,
            signal_metrics.total
        );
    }
}

fn push_seven_day_mid_reversal_sweep(
    sweep: &mut SevenDayMidReversalSweep,
    observation: &BacktestObservation,
) {
    let production_sample = BacktestSample {
        predicted_change: observation.predicted_change,
        actual_change: observation.actual_change,
    };
    sweep.production_samples.push(production_sample);

    let is_mid_reversal = observation
        .key_factors
        .iter()
        .any(|factor| factor.contains("7日中幅方向反向校准"));
    let undo_predicted_change = if is_mid_reversal {
        compound_change(
            observation.base_price,
            &observation
                .predicted_daily_changes
                .iter()
                .map(|change| -change)
                .collect::<Vec<_>>(),
        )
    } else {
        observation.predicted_change
    };
    let undo_sample = BacktestSample {
        predicted_change: undo_predicted_change,
        actual_change: observation.actual_change,
    };
    sweep.undo_samples.push(undo_sample);

    if is_mid_reversal {
        sweep.matched += 1;
        sweep.affected_production_samples.push(production_sample);
        sweep.affected_undo_samples.push(undo_sample);
    }
}

fn print_seven_day_mid_reversal_sweep(sweep: &SevenDayMidReversalSweep) {
    let production_metrics = compute_metrics(&sweep.production_samples);
    let undo_metrics = compute_metrics(&sweep.undo_samples);
    let affected_production_metrics = compute_metrics(&sweep.affected_production_samples);
    let affected_undo_metrics = compute_metrics(&sweep.affected_undo_samples);

    println!("  7日中幅反向校准撤销对照:");
    println!(
        "    production matched {:<4} acc {:.1}% base {:.1}% edge {:+.2}% pred_up {:.0}% | undo acc {:.1}% edge {:+.2}% pred_up {:.0}%",
        sweep.matched,
        production_metrics.direction_accuracy * 100.0,
        production_metrics.baseline_accuracy * 100.0,
        production_metrics.edge() * 100.0,
        production_metrics.predicted_up_ratio * 100.0,
        undo_metrics.direction_accuracy * 100.0,
        undo_metrics.edge() * 100.0,
        undo_metrics.predicted_up_ratio * 100.0,
    );
    println!(
        "    affected production acc {:.1}% base {:.1}% n {} | affected undo acc {:.1}% base {:.1}% n {}",
        affected_production_metrics.direction_accuracy * 100.0,
        affected_production_metrics.baseline_accuracy * 100.0,
        affected_production_metrics.total,
        affected_undo_metrics.direction_accuracy * 100.0,
        affected_undo_metrics.baseline_accuracy * 100.0,
        affected_undo_metrics.total,
    );
}

fn print_horizon_symbol_diagnostics(horizon: usize, reports: &[(String, f64, f64, f64, f64)]) {
    let mut ranked = reports.to_vec();
    ranked.sort_by(|a, b| (a.1 - a.2).partial_cmp(&(b.1 - b.2)).unwrap());
    println!("  {horizon}日拖累最大的5只:");
    for (symbol, acc, baseline, pred_up, actual_up) in ranked.iter().take(5) {
        println!(
            "    {symbol:<10} acc {:.1}% base {:.1}% edge {:+.1}% pred_up {:.0}% actual_up {:.0}%",
            acc * 100.0,
            baseline * 100.0,
            (acc - baseline) * 100.0,
            pred_up * 100.0,
            actual_up * 100.0
        );
    }
}

fn cross_section_bias_by_symbol_date(
    data: &[(String, Vec<biga_lib::db::models::HistoricalData>)],
    horizon: usize,
    window: usize,
) -> BiasMap {
    walk_forward_rank_signals(data, horizon, window)
        .into_iter()
        .filter_map(|signal| {
            let daily_bias = daily_bias_from_rank(signal.rank, signal.total);
            (daily_bias != 0.0).then_some(((signal.date, signal.symbol), daily_bias))
        })
        .collect()
}

fn cross_bias_sweeps_by_symbol_date(
    data: &[(String, Vec<biga_lib::db::models::HistoricalData>)],
    horizon: usize,
    window: usize,
    magnitudes: &[f64],
) -> Vec<(f64, BiasMap)> {
    let signals = walk_forward_rank_signals(data, horizon, window);
    magnitudes
        .iter()
        .map(|&magnitude| {
            let by_symbol_date = signals
                .iter()
                .filter_map(|signal| {
                    let daily_bias =
                        daily_bias_from_rank_with_magnitude(signal.rank, signal.total, magnitude);
                    (daily_bias != 0.0).then_some(((signal.date, signal.symbol.clone()), daily_bias))
                })
                .collect();
            (magnitude, by_symbol_date)
        })
        .collect()
}

fn daily_bias_from_rank_with_magnitude(rank: usize, total: usize, magnitude: f64) -> f64 {
    if total <= 1 || rank == 0 {
        return 0.0;
    }

    let percentile = (rank - 1) as f64 / (total - 1) as f64;
    if percentile <= 0.20 {
        magnitude
    } else if percentile >= 0.80 {
        -magnitude
    } else {
        0.0
    }
}

fn print_cross_sweep_metrics(samples_by_magnitude: &[(f64, Vec<BacktestSample>)]) {
    println!("  5日截面偏置强度候选:");
    for (magnitude, samples) in samples_by_magnitude {
        let metrics = compute_metrics(samples);
        println!(
            "    bias {:+.3}% acc {:.1}% base {:.1}% edge {:+.2}% pred_up {:.0}% n {}",
            magnitude,
            metrics.direction_accuracy * 100.0,
            metrics.baseline_accuracy * 100.0,
            metrics.edge() * 100.0,
            metrics.predicted_up_ratio * 100.0,
            metrics.total
        );
    }
}

fn apply_daily_bias_to_observation(
    base_price: f64,
    daily_changes: &[f64],
    daily_bias: f64,
    symbol: &str,
) -> f64 {
    if daily_bias == 0.0 || base_price <= 0.0 || !base_price.is_finite() {
        return compound_change(base_price, daily_changes);
    }
    if !cross_section_bias_aligns_with_prediction(daily_changes, daily_bias) {
        return compound_change(base_price, daily_changes);
    }

    let (limit_down, limit_up) = get_stock_price_limits(Some(symbol));
    let mut price = base_price;
    for change in daily_changes {
        let adjusted = (change + daily_bias).clamp(limit_down, limit_up);
        price *= 1.0 + adjusted / 100.0;
    }
    (price - base_price) / base_price * 100.0
}

fn cross_section_bias_aligns_with_prediction(daily_changes: &[f64], daily_bias: f64) -> bool {
    daily_changes
        .first()
        .is_some_and(|change| *change != 0.0 && change.signum() == daily_bias.signum())
}

fn compound_change(base_price: f64, daily_changes: &[f64]) -> f64 {
    if base_price <= 0.0 || !base_price.is_finite() {
        return 0.0;
    }

    let mut price = base_price;
    for change in daily_changes {
        price *= 1.0 + change / 100.0;
    }
    (price - base_price) / base_price * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use biga_lib::db::models::HistoricalData;
    use chrono::Duration;

    fn synthetic_history(closes: &[f64]) -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        closes
            .iter()
            .enumerate()
            .map(|(i, close)| HistoricalData {
                symbol: "test".to_string(),
                date: start + Duration::days(i as i64),
                open: *close,
                close: *close,
                high: *close + 0.5,
                low: *close - 0.5,
                volume: 10_000 + i as i64,
                amount: *close * 10_000.0,
                amplitude: 1.0,
                turnover_rate: 1.0,
                volume_ratio: 1.0,
                change_percent: 0.1,
                change: 0.1,
            })
            .collect()
    }

    #[test]
    fn test_tune_config_defaults_match_full_diagnostic() {
        let config = TuneConfig::from_args(Vec::new()).unwrap();

        assert_eq!(config.max_symbols, 25);
        assert_eq!(config.horizons, vec![1, 5, 7, 10]);
        assert_eq!(config.backtest_step, 10);
        assert!(!config.show_diagnostics);
    }

    #[test]
    fn test_tune_config_parses_quick_diagnostic_args() {
        let config = TuneConfig::from_args([
            "--symbols=8".to_string(),
            "--horizons=1,5".to_string(),
            "--step=20".to_string(),
            "--diagnostics".to_string(),
        ])
        .unwrap();

        assert_eq!(config.max_symbols, 8);
        assert_eq!(config.horizons, vec![1, 5]);
        assert_eq!(config.backtest_step, 20);
        assert!(config.show_diagnostics);
    }

    #[test]
    fn test_tune_config_rejects_invalid_values() {
        assert!(TuneConfig::from_args(["--symbols=0".to_string()]).is_err());
        assert!(TuneConfig::from_args(["--horizons=".to_string()]).is_err());
        assert!(TuneConfig::from_args(["--unknown=1".to_string()]).is_err());
        assert!(TuneConfig::from_args(["--help".to_string()]).is_err());
    }

    #[test]
    fn test_diagnostics_requires_explicit_flag_for_every_horizon() {
        let config = TuneConfig::from_args(Vec::new()).unwrap();

        assert!(!diagnostics_enabled_for_horizon(&config, 1));
        assert!(!diagnostics_enabled_for_horizon(&config, 5));
        assert!(!diagnostics_enabled_for_horizon(&config, 7));
        assert!(!diagnostics_enabled_for_horizon(&config, 10));

        let config = TuneConfig::from_args(["--diagnostics".to_string()]).unwrap();
        assert!(diagnostics_enabled_for_horizon(&config, 7));
    }

    #[test]
    fn test_cross_sweep_only_runs_for_five_day_diagnostics() {
        assert!(!cross_sweep_enabled(false, 5));
        assert!(!cross_sweep_enabled(true, 1));
        assert!(cross_sweep_enabled(true, 5));
        assert!(!cross_sweep_enabled(true, 7));
    }

    #[test]
    fn test_cross_hit_summary_hides_inactive_cross_section_stats() {
        assert_eq!(cross_hit_summary(false, 0, 100), "-/-");
        assert_eq!(cross_hit_summary(true, 12, 100), "12/100");
    }

    #[test]
    fn test_calibration_bucket_idx_classifies_key_factors() {
        assert_eq!(
            calibration_bucket_idx(&["均值回归特征校准: RSI14".to_string()]),
            0
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按真实基率校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按短周期高基率反向校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按1日弱基率校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按5日弱基率校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按7日高基率反向校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按7日中幅方向反向校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按7日反转策略偏多校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&["真实数据校准: 已按长周期弱基率校准".to_string()]),
            1
        );
        assert_eq!(
            calibration_bucket_idx(&[
                "均值回归特征校准: RSI14".to_string(),
                "真实数据校准: 已按真实基率校准".to_string(),
            ]),
            2
        );
        assert_eq!(calibration_bucket_idx(&[]), 3);
    }

    #[test]
    fn test_calibration_direction_bucket_idx_classifies_direction() {
        assert_eq!(
            calibration_direction_bucket_idx(&["均值回归特征校准: RSI14，按3000日真实回测特征偏多".to_string()]),
            0
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["均值回归特征校准: RSI14，按3000日真实回测特征偏空".to_string()]),
            1
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率55%，上涨基率达到短周期基率阈值55%，已按真实基率校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率52%，上涨基率略高于长周期中性线50%，已按长周期弱基率校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率52%，上涨基率略高于5日中性线50%，已按5日弱基率校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率52%，上涨基率略高于1日中性线50%，已按1日弱基率校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率45%，上涨基率低于短周期基率阈值45%，已按真实基率校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率57%，上涨基率高于短周期过热阈值55%，已按短周期高基率反向校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率57%，上涨基率高于7日过热阈值55%，已按7日高基率反向校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 7日中幅预测方向由看跌反向为看涨，已按7日中幅方向反向校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 7日中幅预测方向由看涨反向为看跌，已按7日中幅方向反向校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 7日反转策略历史上涨基率偏高，已按7日反转策略偏多校准".to_string()]),
            2
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率48%，上涨基率略低于5日中性线50%，已按5日弱基率校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率48%，上涨基率略低于1日中性线50%，已按1日弱基率校准".to_string()]),
            3
        );
        assert_eq!(
            calibration_direction_bucket_idx(&["真实数据校准: 近100次历史结果显示上涨基率48%，上涨基率略低于长周期中性线50%，已按长周期弱基率校准".to_string()]),
            3
        );
        assert_eq!(calibration_direction_bucket_idx(&[]), 4);
    }

    #[test]
    fn test_strategy_bucket_idx_classifies_prediction_reason() {
        assert_eq!(strategy_bucket_idx(Some("市场:震荡 | 策略:趋势跟踪")), 0);
        assert_eq!(strategy_bucket_idx(Some("市场:震荡 | 策略:均值回归")), 1);
        assert_eq!(strategy_bucket_idx(Some("市场:震荡 | 策略:反转策略")), 2);
        assert_eq!(strategy_bucket_idx(Some("市场:震荡")), 3);
        assert_eq!(strategy_bucket_idx(None), 3);
    }

    #[test]
    fn test_ma_gap_at_prediction_date_uses_visible_window() {
        let history = synthetic_history(&[100.0, 101.0, 102.0, 103.0, 110.0, 150.0]);
        let gap = ma_gap_at_prediction_date(&history, history[4].date, 5).unwrap();
        let expected = (110.0 - 103.2) / 103.2 * 100.0;

        assert!((gap - expected).abs() < 1e-9);
        assert!(ma_gap_at_prediction_date(&history, history[3].date, 5).is_none());
    }

    #[test]
    fn test_candidate_change_flips_only_known_diagnostic_patterns() {
        assert_eq!(candidate_change(1, 0.2), -0.2);
        assert_eq!(candidate_change(1, 0.6), 0.6);
        assert_eq!(candidate_change(1, -0.2), -0.2);
        assert_eq!(candidate_change(7, 1.0), 1.0);
        assert_eq!(candidate_change(7, -1.0), -1.0);
        assert_eq!(candidate_change(7, 0.2), 0.2);
    }

    #[test]
    fn test_one_day_skipped_bullish_signal_detects_ma5_oversold() {
        let mut closes = (0..24)
            .map(|i| if i % 2 == 0 { 100.0 } else { 100.2 })
            .collect::<Vec<_>>();
        closes.push(96.0);
        let history = synthetic_history(&closes);

        let signal = one_day_skipped_bullish_signal(&history, history.last().unwrap().date)
            .unwrap();

        assert_eq!(signal.kind, OneDayBullishSignalKind::FeatureMa5);
        assert!(signal.calibrated_change > 0.0);
    }

    #[test]
    fn test_one_day_skipped_bullish_signal_ignores_ma5_overbought() {
        let mut closes = (0..24)
            .map(|i| if i % 2 == 0 { 100.0 } else { 100.2 })
            .collect::<Vec<_>>();
        closes.push(104.0);
        let history = synthetic_history(&closes);

        let signal = one_day_skipped_bullish_signal(&history, history.last().unwrap().date);

        assert!(signal.is_none());
    }

    #[test]
    fn test_one_day_ma5_bullish_threshold_sweep_applies_matching_thresholds() {
        let mut closes = (0..24)
            .map(|i| if i % 2 == 0 { 100.0 } else { 100.2 })
            .collect::<Vec<_>>();
        closes.push(96.0);
        let history = synthetic_history(&closes);
        let prediction_date = history.last().unwrap().date;
        let observation = BacktestObservation {
            prediction_date,
            target_date: prediction_date,
            base_price: 100.0,
            predicted_price: 99.8,
            predicted_daily_changes: vec![-0.2],
            actual_price: 101.0,
            predicted_change: -0.2,
            actual_change: 1.0,
            confidence: 0.35,
            key_factors: Vec::new(),
            prediction_reason: None,
        };
        let mut sweeps = one_day_ma5_bullish_threshold_sweeps();

        push_one_day_ma5_bullish_threshold_sweeps(&mut sweeps, &history, &observation);

        assert_eq!(sweeps[0].matched, 1);
        assert_eq!(sweeps[3].matched, 1);
        assert_eq!(sweeps[4].matched, 0);
        assert!(sweeps[0].all_samples[0].predicted_change > 0.0);
        assert_eq!(sweeps[4].all_samples[0].predicted_change, -0.2);
    }

    #[test]
    fn test_seven_day_mid_reversal_sweep_undoes_reversed_daily_changes() {
        let prediction_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let observation = BacktestObservation {
            prediction_date,
            target_date: prediction_date + Duration::days(7),
            base_price: 100.0,
            predicted_price: 99.0,
            predicted_daily_changes: vec![-0.1, -0.2],
            actual_price: 101.0,
            predicted_change: -0.2998,
            actual_change: 1.0,
            confidence: 0.35,
            key_factors: vec![
                "真实数据校准: 7日中幅预测方向由看涨反向为看跌，已按7日中幅方向反向校准".to_string(),
            ],
            prediction_reason: None,
        };
        let mut sweep = SevenDayMidReversalSweep::default();

        push_seven_day_mid_reversal_sweep(&mut sweep, &observation);

        assert_eq!(sweep.matched, 1);
        assert!(sweep.production_samples[0].predicted_change < 0.0);
        assert!(sweep.undo_samples[0].predicted_change > 0.0);
        assert_eq!(sweep.affected_production_samples.len(), 1);
        assert_eq!(sweep.affected_undo_samples.len(), 1);
    }

}
