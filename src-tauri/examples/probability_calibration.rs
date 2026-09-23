//! 概率校准实验：回答「界面上到底该不该出现一个上涨概率」。
//!
//! 用法：cargo run --release --example probability_calibration
//!
//! 两条口径，**都相对气候基率（climatology）打 Brier Skill Score**：
//!
//! - **A. 单股 P(H 日上涨)**：由漂移锚 μ_H 与校准 σ_H，配训练段拟合的经验标准化残差 CDF
//!   得到 `P = 1 − F̂(−μ_H/σ_H)`。参照物是该股自身的历史上涨频率。
//!   实证结论 1 预期这里 **BSS ≤ 0**——那是正确结果，不是失败：它意味着界面上只该出现基率。
//! - **C. 截面 P(H 日跑赢当日中位数)**：实证结论 2 说截面相对强弱有真 alpha（Rank IC ≈ +0.05），
//!   所以这一条才是最可能出现 BSS > 0 的地方。参照物按构造就是 0.5。
//!
//! 所有映射都在**训练段拟合、留出段评估**。同一批样本拟合又自评得到的"技能"是自欺——
//! `calibration` 模块的 `test_isotonic_on_noise_has_no_out_of_sample_skill` 把这条钉死了。

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data_for_symbols;
use biga_lib::prediction::analysis::volatility_forecast::{
    cumulative_sigma_path, estimate_garch_params, ewma_daily_vol, realized_daily_vol_zero_mean,
    EWMA_LAMBDA,
};
use biga_lib::prediction::calibration::{
    isotonic_fit, score_probabilities_against, ProbabilityScore,
};
use biga_lib::prediction::cross_section::{build_panel, walk_forward_rank_signals};
use biga_lib::prediction::model::inference::empirical_horizon_stats;
use chrono::NaiveDate;
use sqlx::Row;
use std::collections::HashMap;

const MIN_LOOKBACK: usize = 260;
const LONG_VOL_WINDOW: usize = 250;
const TRAIN_FRACTION: f64 = 0.7;
const CS_HORIZON: usize = 5;
const CS_WINDOW: usize = 250;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");
    let symbols: Vec<String> = sqlx::query(
        "SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 400 ORDER BY symbol LIMIT 140",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| r.get::<String, _>("symbol"))
    .collect();

    let stocks: Vec<(String, Vec<HistoricalData>)> =
        get_recent_historical_data_for_symbols(&symbols, 900, &pool)
            .await
            .unwrap()
            .into_iter()
            .filter(|(_, h)| h.len() >= MIN_LOOKBACK + 60)
            .collect();
    println!("参与评估的股票数：{}", stocks.len());
    if stocks.is_empty() {
        eprintln!("库里没有足够历史的标的，先补数据。");
        std::process::exit(1);
    }

    for horizon in [1usize, 5, 10] {
        single_stock_up_probability(&stocks, horizon);
    }
    cross_section_beat_median(&stocks);

    println!("\n{}", "=".repeat(78));
    println!("判读：");
    println!("· BSS > 0 才说明这个概率比「直接报历史基率」多带了信息；≤ 0 就该只报基率。");
    println!("· ECE / 可靠性图看的是校准：报 60% 的那批是不是真有 ~60% 发生。");
    println!("· 校准与技能互相独立：报基率的预测完全校准但零技能，两个数必须一起看。");
}

// =============================================================================
// A. 单股 P(H 日上涨)
// =============================================================================

struct UpObs {
    /// 标准化残差 (actual − μ_H) / σ_H，用于拟经验 CDF
    resid: f64,
    /// μ_H / σ_H，唯一能把概率推离中性的量
    signal: f64,
    /// 该股当时的历史上涨基率
    climatology: f64,
    /// 实际是否上涨
    up: bool,
    holdout: bool,
}

fn single_stock_up_probability(stocks: &[(String, Vec<HistoricalData>)], horizon: usize) {
    let mut obs: Vec<UpObs> = Vec::new();

    for (_, hist) in stocks {
        let closes: Vec<f64> = hist.iter().map(|h| h.close).collect();
        let n = closes.len();
        if n < MIN_LOOKBACK + horizon + 1 {
            continue;
        }
        let split_idx = MIN_LOOKBACK + (((n - MIN_LOOKBACK) as f64) * TRAIN_FRACTION) as usize;

        let mut base_idx = MIN_LOOKBACK;
        while base_idx + horizon < n {
            let base = closes[base_idx];
            let future = closes[base_idx + horizon];
            if base > 0.0 && future > 0.0 {
                let actual = (future - base) / base * 100.0; // 百分点，与 μ_H 同单位
                if let (Some(stats), Some(sigma_h)) = (
                    empirical_horizon_stats(&hist[..=base_idx], horizon),
                    sigma_h_pct(&closes[..=base_idx], horizon),
                ) {
                    if sigma_h > 1e-6 && actual.is_finite() && actual.abs() >= 0.01 {
                        obs.push(UpObs {
                            resid: (actual - stats.average_change) / sigma_h,
                            signal: stats.average_change / sigma_h,
                            climatology: stats.up_ratio,
                            up: actual > 0.0,
                            holdout: base_idx >= split_idx,
                        });
                    }
                }
            }
            base_idx += horizon.max(1);
        }
    }

    println!("\n{}", "=".repeat(78));
    println!("A. 单股 P({horizon} 日上涨)");
    if obs.len() < 500 {
        println!("  样本不足（{}），跳过", obs.len());
        return;
    }

    // 训练段拟经验标准化残差 CDF
    let mut train_resid: Vec<f64> = obs.iter().filter(|o| !o.holdout).map(|o| o.resid).collect();
    train_resid.sort_by(f64::total_cmp);
    let holdout: Vec<&UpObs> = obs.iter().filter(|o| o.holdout).collect();
    println!("  训练 {} / 留出 {}", train_resid.len(), holdout.len());
    if train_resid.len() < 200 || holdout.len() < 200 {
        println!("  切分后样本不足，跳过");
        return;
    }

    // P(涨) = P(ε > −μ/σ) = 1 − F̂(−μ/σ)
    let model_pairs: Vec<(f64, bool)> = holdout
        .iter()
        .map(|o| (1.0 - empirical_cdf(&train_resid, -o.signal), o.up))
        .collect();
    let clim_pairs: Vec<(f64, bool)> = holdout.iter().map(|o| (o.climatology, o.up)).collect();

    let realized = holdout.iter().filter(|o| o.up).count() as f64 / holdout.len() as f64;
    let Some(model) = score_probabilities_against(&model_pairs, realized) else {
        return;
    };
    let Some(clim) = score_probabilities_against(&clim_pairs, realized) else {
        return;
    };

    print_comparison("模型 P(涨)", &model, "历史基率", &clim);
    println!(
        "  信号量级：|μ_H/σ_H| 中位数 {:.4}（这个数决定概率能偏离中性多远）",
        median(&mut holdout.iter().map(|o| o.signal.abs()).collect::<Vec<_>>())
    );
    print_reliability(&model);
}

/// H 日累计 σ（百分点）：EWMA + 波动率均值回归期限结构
fn sigma_h_pct(closes: &[f64], horizon: usize) -> Option<f64> {
    let sigma_now = ewma_daily_vol(closes, EWMA_LAMBDA)?;
    let long_start = closes.len().saturating_sub(LONG_VOL_WINDOW);
    let sigma_long = realized_daily_vol_zero_mean(&closes[long_start..])?;
    let params = estimate_garch_params(closes);
    let phi = (params.alpha + params.beta).clamp(0.0, 0.999);
    cumulative_sigma_path(sigma_now, sigma_long, phi, horizon)
        .last()
        .map(|s| s * 100.0)
}

// =============================================================================
// C. 截面 P(H 日跑赢当日中位数)
// =============================================================================

fn cross_section_beat_median(stocks: &[(String, Vec<HistoricalData>)]) {
    println!("\n{}", "=".repeat(78));
    println!("C. 截面 P({CS_HORIZON} 日跑赢当日中位数)");

    // 前向收益：(日期, 标的) -> fwd_return
    let panel = build_panel(stocks, CS_HORIZON);
    let mut fwd: HashMap<(NaiveDate, String), f64> = HashMap::new();
    let mut by_date: HashMap<NaiveDate, Vec<f64>> = HashMap::new();
    for day in &panel {
        for row in day {
            fwd.insert((row.date, row.symbol.clone()), row.fwd_return);
            by_date.entry(row.date).or_default().push(row.fwd_return);
        }
    }
    // 当日中位数
    let medians: HashMap<NaiveDate, f64> = by_date
        .into_iter()
        .map(|(d, mut v)| (d, median(&mut v)))
        .collect();

    let signals = walk_forward_rank_signals(stocks, CS_HORIZON, CS_WINDOW);
    if signals.is_empty() {
        println!("  走步无输出（历史长度或截面宽度不够），跳过");
        return;
    }

    // 排名百分位（1 = 最强）作为分数；按日期切训练/留出
    let mut dates: Vec<NaiveDate> = signals.iter().map(|s| s.date).collect();
    dates.sort_unstable();
    dates.dedup();
    let split_date = dates[((dates.len() as f64) * TRAIN_FRACTION) as usize];

    let mut train: Vec<(f64, bool)> = Vec::new();
    let mut holdout_raw: Vec<(f64, bool)> = Vec::new();
    for s in &signals {
        let (Some(&ret), Some(&med)) = (fwd.get(&(s.date, s.symbol.clone())), medians.get(&s.date))
        else {
            continue;
        };
        if s.total <= 1 || !ret.is_finite() {
            continue;
        }
        let pct = 1.0 - (s.rank - 1) as f64 / (s.total - 1) as f64; // 1=最强
        let beat = ret > med;
        if s.date < split_date {
            train.push((pct, beat));
        } else {
            holdout_raw.push((pct, beat));
        }
    }
    println!(
        "  训练 {} / 留出 {}（切分日 {}）",
        train.len(),
        holdout_raw.len(),
        split_date
    );
    if train.len() < 500 || holdout_raw.len() < 500 {
        println!("  样本不足，跳过");
        return;
    }

    let Some(map) = isotonic_fit(&train) else {
        return;
    };
    println!("  保序回归拟出 {} 个概率档位", map.segments());

    let model_pairs: Vec<(f64, bool)> = holdout_raw
        .iter()
        .map(|&(pct, beat)| (map.apply(pct), beat))
        .collect();
    let realized = holdout_raw.iter().filter(|(_, b)| *b).count() as f64 / holdout_raw.len() as f64;
    // 参照：跑赢中位数按构造就是 ~50%
    let clim_pairs: Vec<(f64, bool)> = holdout_raw.iter().map(|&(_, b)| (realized, b)).collect();

    let (Some(model), Some(clim)) = (
        score_probabilities_against(&model_pairs, realized),
        score_probabilities_against(&clim_pairs, realized),
    ) else {
        return;
    };
    print_comparison("截面 P(跑赢中位数)", &model, "常数基率", &clim);
    println!(
        "  概率跨度：最弱档 {:.1}% → 最强档 {:.1}%",
        map.apply(0.0) * 100.0,
        map.apply(1.0) * 100.0
    );
    print_reliability(&model);
}

// =============================================================================
// 输出
// =============================================================================

fn print_comparison(
    model_label: &str,
    model: &ProbabilityScore,
    clim_label: &str,
    clim: &ProbabilityScore,
) {
    let skill = if clim.brier > 0.0 {
        1.0 - model.brier / clim.brier
    } else {
        f64::NAN
    };
    println!(
        "  {model_label:16} Brier={:.5}  logloss={:.5}  ECE={:.4}  预测均值={:.3}  实际频率={:.3}",
        model.brier, model.log_loss, model.ece, model.mean_predicted, model.realized_freq
    );
    println!(
        "  {clim_label:16} Brier={:.5}  logloss={:.5}  ECE={:.4}  预测均值={:.3}",
        clim.brier, clim.log_loss, clim.ece, clim.mean_predicted
    );
    println!(
        "  → 相对{clim_label}的 BSS = {skill:+.5}   {}",
        if skill > 0.001 {
            "【有技能】"
        } else if skill > -0.001 {
            "【与基率无异 = 无技能】"
        } else {
            "【劣于基率 = 无技能】"
        }
    );
    println!(
        "  → Murphy: reliability={:.5}（越小越校准） resolution={:.5}（越大越有分辨力） uncertainty={:.5}",
        model.reliability, model.resolution, model.uncertainty
    );
}

fn print_reliability(s: &ProbabilityScore) {
    println!("  可靠性图（只列样本数 ≥ 100 的桶）:");
    let mut shown = 0;
    for b in &s.bins {
        if b.count < 100 {
            continue;
        }
        shown += 1;
        println!(
            "    [{:.1},{:.1})  n={:>6}  预测均值={:.3}  实际频率={:.3}  缺口={:+.3}",
            b.lo,
            b.hi,
            b.count,
            b.mean_predicted,
            b.realized_freq,
            b.realized_freq - b.mean_predicted
        );
    }
    if shown == 0 {
        println!("    （概率全部集中在极窄区间，没有可分辨的桶——这本身说明预测几乎是常数）");
    }
    if let Some(gap) = s.max_gap_over(200) {
        println!("    样本 ≥200 的桶中最大校准缺口: {:.3}", gap);
    }
}

fn empirical_cdf(sorted: &[f64], x: f64) -> f64 {
    let idx = sorted.partition_point(|&v| v <= x);
    (idx as f64 / sorted.len() as f64).clamp(0.0, 1.0)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}
