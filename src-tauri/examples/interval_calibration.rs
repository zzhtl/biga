//! 区间预测校准实验：方向不可测但波动可测——验证 H 日涨跌区间带是否"校准"。
//!
//! 用法：cargo run --release --example interval_calibration
//!
//! 与旧版相比，这一版回答的是三个此前没回答过的问题：
//!
//! 1. **条件覆盖**。旧版只报池化的边际覆盖率。边际 80% 完全可以是「高波动段覆盖 65% +
//!    低波动段覆盖 93%」凑出来的——用户在高波动股上体感"完全不准"，而报表上一切正常。
//!    这里按 horizon × 波动状态分组，逐格报覆盖率。
//! 2. **带宽是否值这个覆盖率**。只看覆盖率会被"无限放宽"骗过去，所以一律配 pinball loss
//!    （见 `prediction::calibration::score_intervals`）。
//! 3. **σ 估计器选哪个**。对比四种：生产在用的 realized20、EWMA、EWMA+均值回归期限结构、
//!    GARCH。σ 一变，配套的 z 就得重新定，所以两者必须一起评。
//!
//! **所有分位都在训练段拟合、在留出段评估**。同一批样本拟合又自评，得到的"校准"是自欺。

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data_for_symbols;
use biga_lib::prediction::analysis::prediction_interval;
use biga_lib::prediction::analysis::volatility_forecast::{
    calculate_realized_volatility, cumulative_sigma_path, estimate_garch_params, ewma_daily_vol,
    realized_daily_vol_zero_mean, GarchForecaster, EWMA_LAMBDA,
};
use biga_lib::prediction::calibration::score_intervals;
use biga_lib::prediction::model::inference::empirical_horizon_stats;
use biga_lib::utils::math::empirical_quantile;
use sqlx::Row;

/// 估长期波动与 GARCH 参数需要的最小历史
const MIN_LOOKBACK: usize = 260;
/// 长期波动的回看窗口
const LONG_VOL_WINDOW: usize = 250;
/// 训练段占比，其余留作样本外评估
const TRAIN_FRACTION: f64 = 0.7;
/// 分组报告的最小样本数，低于它的格子只打印不判定
const MIN_CELL: usize = 300;
/// 条件一致性验收：波动率最高档与最低档的覆盖率之差上限（百分点）。
///
/// 这是本次改造要解决的核心问题——同一条名义 80% 的带，在不同波动状态的票上可不可信。
/// 改造前 realized20 在 h=10 上是 20.0pp。
const SPREAD_TOLERANCE: f64 = 0.05;

/// 边际水平验收：整体覆盖率偏离名义值的上限（百分点）。
///
/// 比条件一致性松，因为写死的常数必然承担 regime drift：全样本标定出来的分位在最近
/// 一年（留出段）上偏窄约 2~4pp，且随 horizon 增大。收紧到 ±3pp 就只能拿最近一年去拟合，
/// 那是拿验证集当训练集。
const LEVEL_TOLERANCE: f64 = 0.05;

/// 参与对比的 σ 估计器。
///
/// `production` 直接调生产函数；其余是对照。重新调参时可以往里加 `blendNN`（起始方差
/// 收缩权重）或 `phiNN`（均值回归速度下限）——两者都由名字里的数字参数化。
/// 已测结论：`phiNN` 抬高 φ 只会让带更窄、覆盖更差（σ_now/σ_long 多数时候 <1，
/// 因为方差分布右偏、250 日均值高于典型值），不是可行方向。
const METHODS: [&str; 6] = [
    "production", "realized20", "ewma", "ewma_ts", "garch", "blend65",
];
const HORIZONS: [usize; 3] = [1, 5, 10];

/// 一次走步观测
#[derive(Clone, Copy)]
struct Obs {
    /// 实际 H 日收益（小数）
    actual: f64,
    /// 漂移锚 μ_H（小数）。生产的区间带就是以它为中心，这里必须对齐，
    /// 否则拟合出来的"上下不对称"混进的是漂移而不是真偏度。
    mu_h: f64,
    /// 该方法给出的 H 日累计 σ
    sigma_h: f64,
    /// σ_now / σ_long，刻画当前处在什么波动状态
    vol_ratio: f64,
    /// 是否属于留出段
    holdout: bool,
}

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    let symbols: Vec<String> = sqlx::query(
        "SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 400 ORDER BY symbol LIMIT 120",
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
    println!("参与校准的股票数：{}", stocks.len());
    if stocks.is_empty() {
        eprintln!(
            "库里没有满足 >= {} 根 K 线的标的。先跑 fetch_universe / batch_refresh 补数据。",
            MIN_LOOKBACK + 60
        );
        std::process::exit(1);
    }

    let mut production_passed = true;
    let mut checked_cells = 0usize;
    let mut summaries: Vec<Summary> = Vec::new();

    for horizon in HORIZONS {
        for method in METHODS {
            let obs = collect(&stocks, method, horizon);
            if obs.len() < 200 {
                println!("\n[h={horizon} 方法={method}] 样本不足（{}），跳过", obs.len());
                continue;
            }
            let (pass, cells) = report(&obs, method, horizon, &mut summaries);
            if method == "production" {
                production_passed &= pass;
                checked_cells += cells;
            }
        }
    }

    print_summary_table(&summaries);

    println!("\n{}", "=".repeat(78));
    println!("判读：");
    println!(
        "· 「生产口径」用的是 prediction_interval::calibrated_z 当前写死的 z。某一格覆盖率"
    );
    println!("  显著偏离名义值，说明那类样本上的区间不可信——这正是「概率不准」的来源。");
    println!(
        "· 「留出段拟合口径」是在训练段取经验分位、在留出段评估的结果。它比生产口径好很多，"
    );
    println!("  就说明把 z 拆成分组表是值得做的；差不多，就说明当前的单常数已经够用。");
    println!("· pinball 越小越好，可用来在四种 σ 估计器之间做选择；不要只比覆盖率。");

    assert!(checked_cells > 0, "生产区间没有任何可判定的分组，样本不足");
    assert!(
        production_passed,
        "生产区间未通过验收：条件一致性跨度或边际覆盖水平超限"
    );
}

/// 走步收集观测：非重叠抽样，每个标的按时间先后切训练/留出段。
fn collect(stocks: &[(String, Vec<HistoricalData>)], method: &str, horizon: usize) -> Vec<Obs> {
    let mut out = Vec::new();
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
                let actual = (future - base) / base;
                if let (Some((sigma_h, sigma_now, sigma_long)), Some(stats)) = (
                    sigma_for(method, &closes, base_idx, horizon),
                    empirical_horizon_stats(&hist[..=base_idx], horizon),
                ) {
                    if sigma_h > 1e-6 && actual.is_finite() && sigma_long > 1e-9 {
                        out.push(Obs {
                            actual,
                            mu_h: stats.average_change / 100.0,
                            sigma_h,
                            vol_ratio: sigma_now / sigma_long,
                            holdout: base_idx >= split_idx,
                        });
                    }
                }
            }
            base_idx += horizon.max(1);
        }
    }
    out
}

/// 返回 (H 日累计 σ, 当日条件 σ, 长期 σ)
fn sigma_for(
    method: &str,
    closes: &[f64],
    base_idx: usize,
    horizon: usize,
) -> Option<(f64, f64, f64)> {
    let visible = &closes[..=base_idx];
    let long_start = visible.len().saturating_sub(LONG_VOL_WINDOW);
    let sigma_long = realized_daily_vol_zero_mean(&visible[long_start..])?;
    let h = horizon as f64;

    match method {
        // 生产真正在用的那条路径：直接调生产函数，杜绝 harness 与生产实现漂移
        "production" => {
            let path = prediction_interval::horizon_sigma_path(visible, horizon);
            let sigma_h = *path.last()?;
            let sigma_now = *path.first()?;
            (sigma_h > 1e-9).then_some((sigma_h, sigma_now, sigma_long))
        }
        // 改造前的老口径：近 20 日等权、减样本均值的已实现波动 × √H
        "realized20" => {
            let start = visible.len().saturating_sub(20);
            let sd = calculate_realized_volatility(&visible[start..]);
            (sd > 1e-9).then_some((sd * h.sqrt(), sd, sigma_long))
        }
        // 换成零均值 EWMA，仍按 √H 外推
        "ewma" => {
            let sd = ewma_daily_vol(visible, EWMA_LAMBDA)?;
            Some((sd * h.sqrt(), sd, sigma_long))
        }
        // EWMA + 波动率均值回归的期限结构，替掉 √H
        "ewma_ts" => {
            let sd = ewma_daily_vol(visible, EWMA_LAMBDA)?;
            let params = estimate_garch_params(visible);
            let phi = (params.alpha + params.beta).clamp(0.0, 0.999);
            let path = cumulative_sigma_path(sd, sigma_long, phi, horizon);
            path.last().map(|&s| (s, sd, sigma_long))
        }
        // 与 production 同构，只是给均值回归速度 φ 设下限。
        // 动机：ACF(1) 估出的 φ 偏低（波动率持续性被低估是已知现象），远期回归得太快，
        // 高波动股在 h=10 上的带就会偏窄。
        m if m.starts_with("phi") => {
            let floor: f64 = m.trim_start_matches("phi").parse::<f64>().ok()? / 100.0;
            let ewma = ewma_daily_vol(visible, EWMA_LAMBDA)?;
            let sd = (0.8 * ewma * ewma + 0.2 * sigma_long * sigma_long).sqrt();
            let params = estimate_garch_params(visible);
            let phi = (params.alpha + params.beta).clamp(0.0, 0.999).max(floor);
            let path = cumulative_sigma_path(sd, sigma_long, phi, horizon);
            path.last().map(|&s| (s, sd, sigma_long))
        }
        // EWMA 先按权重 w 向长期方差收缩，再走期限结构。
        // 动机：EWMA 是"当前波动的估计"，不是"下一日波动的预测"。波动率均值回归意味着
        // 当前很安静的票，下一日的风险高于它近期波动所显示的——这正是低波动段覆盖不足的根因。
        m if m.starts_with("blend") => {
            let w: f64 = m.trim_start_matches("blend").parse::<f64>().ok()? / 100.0;
            let ewma = ewma_daily_vol(visible, EWMA_LAMBDA)?;
            let sd = (w * ewma * ewma + (1.0 - w) * sigma_long * sigma_long).sqrt();
            let params = estimate_garch_params(visible);
            let phi = (params.alpha + params.beta).clamp(0.0, 0.999);
            let path = cumulative_sigma_path(sd, sigma_long, phi, horizon);
            path.last().map(|&s| (s, sd, sigma_long))
        }
        // 研究对照：现成的简化 GJR-GARCH
        "garch" => {
            let f = GarchForecaster::from_prices(visible);
            let daily = f.forecast(horizon).forecast;
            let sigma_h = daily.iter().map(|s| s * s).sum::<f64>().sqrt();
            let sigma_now = *daily.first()?;
            (sigma_h > 1e-9).then_some((sigma_h, sigma_now, sigma_long))
        }
        _ => None,
    }
}

/// 一个 (horizon, method) 组合的汇总，用于最后的选型表
struct Summary {
    horizon: usize,
    method: &'static str,
    nominal: f64,
    /// 生产 z 下留出段的整体覆盖
    prod_coverage: f64,
    /// 生产 z 下条件覆盖的最大偏差（绝对值，百分点）
    prod_max_gap: f64,
    /// 条件覆盖最好与最差格子的跨度（百分点）——「概率准不准」最直观的一个数
    prod_spread: f64,
    prod_pinball: f64,
    prod_width: f64,
    /// 全样本拟合的上下分位（给生产写常数用）
    full_lo: f64,
    full_hi: f64,
}

/// 打印一个 (horizon, method) 组合的完整报告
fn report(
    obs: &[Obs],
    method: &'static str,
    horizon: usize,
    out: &mut Vec<Summary>,
) -> (bool, usize) {
    let train: Vec<Obs> = obs.iter().copied().filter(|o| !o.holdout).collect();
    let holdout: Vec<Obs> = obs.iter().copied().filter(|o| o.holdout).collect();

    println!("\n{}", "-".repeat(78));
    println!(
        "[h={horizon} 方法={method}] 训练 {} / 留出 {}",
        train.len(),
        holdout.len()
    );
    if train.len() < 100 || holdout.len() < 100 {
        println!("  样本太少，跳过");
        return (true, 0);
    }

    // 训练段拟合的经验分位（上下分开取，不再强制对称）
    let mut std_resid: Vec<f64> = train
        .iter()
        .map(|o| (o.actual - o.mu_h) / o.sigma_h)
        .collect();
    std_resid.sort_by(f64::total_cmp);
    let fitted = |nominal: f64| -> (f64, f64) {
        let tail = (1.0 - nominal) / 2.0;
        (
            empirical_quantile(&std_resid, tail).unwrap_or(-1.28),
            empirical_quantile(&std_resid, 1.0 - tail).unwrap_or(1.28),
        )
    };

    // 全样本分位：生产要写死一个常数时，用全样本比用 70% 训练段更稳
    let mut full_resid: Vec<f64> = obs.iter().map(|o| (o.actual - o.mu_h) / o.sigma_h).collect();
    full_resid.sort_by(f64::total_cmp);

    let mut all_pass = true;
    let mut cells = 0usize;

    for nominal in [0.80_f64, 0.90, 0.95] {
        let (prod_lo, prod_hi) = prediction_interval::calibrated_quantiles(nominal);
        let (lo_q, hi_q) = fitted(nominal);
        println!(
            "  名义 {:.0}%：生产分位 下={prod_lo:.2} 上={prod_hi:.2}；训练段拟合分位 下={lo_q:.3} 上={hi_q:.3}{}",
            nominal * 100.0,
            if (lo_q.abs() - hi_q.abs()).abs() > 0.15 {
                "  ← 上下明显不对称，对称带在此处必然有一侧不准"
            } else {
                ""
            }
        );

        let prod = band_score(
            &holdout,
            |o| (o.mu_h + prod_lo * o.sigma_h, o.mu_h + prod_hi * o.sigma_h),
            nominal,
        );
        let fit = band_score(
            &holdout,
            |o| (o.mu_h + lo_q * o.sigma_h, o.mu_h + hi_q * o.sigma_h),
            nominal,
        );
        if let (Some(p), Some(f)) = (prod, fit) {
            println!(
                "    留出整体  生产: 覆盖 {:>5.1}% 带宽 ±{:>5.2}% pinball {:.5}  |  拟合: 覆盖 {:>5.1}% 带宽 ±{:>5.2}% pinball {:.5}",
                p.coverage * 100.0,
                p.mean_width * 50.0,
                p.pinball,
                f.coverage * 100.0,
                f.mean_width * 50.0,
                f.pinball,
            );
            println!(
                "    越界分布  生产: 下方 {:.1}% / 上方 {:.1}%",
                p.below_rate * 100.0,
                p.above_rate * 100.0
            );
        }

        // 条件覆盖：按训练段定出的波动状态三分位切分留出段
        let mut ratios: Vec<f64> = train.iter().map(|o| o.vol_ratio).collect();
        ratios.sort_by(f64::total_cmp);
        let t1 = empirical_quantile(&ratios, 1.0 / 3.0).unwrap_or(1.0);
        let t2 = empirical_quantile(&ratios, 2.0 / 3.0).unwrap_or(1.0);
        println!("    条件覆盖（按 σ_now/σ_long 三分位，切点 {t1:.2} / {t2:.2}）:");
        let mut cell_coverages: Vec<f64> = Vec::new();
        for (label, lo, hi) in [
            ("低波动段", f64::NEG_INFINITY, t1),
            ("中波动段", t1, t2),
            ("高波动段", t2, f64::INFINITY),
        ] {
            let cell: Vec<Obs> = holdout
                .iter()
                .copied()
                .filter(|o| o.vol_ratio >= lo && o.vol_ratio < hi)
                .collect();
            let Some(sc) = band_score(
                &cell,
                |o| (o.mu_h + prod_lo * o.sigma_h, o.mu_h + prod_hi * o.sigma_h),
                nominal,
            ) else {
                println!("      {label}: 无样本");
                continue;
            };
            let judged = sc.n >= MIN_CELL;
            let ok = sc.coverage_gap().abs() <= LEVEL_TOLERANCE;
            if judged {
                cells += 1;
                cell_coverages.push(sc.coverage);
            }
            println!(
                "      {label}: n={:>5} 覆盖 {:>5.1}% (名义 {:.0}%, 偏差 {:+.1}pp) {}",
                sc.n,
                sc.coverage * 100.0,
                nominal * 100.0,
                sc.coverage_gap() * 100.0,
                if !judged {
                    "样本不足，不判定"
                } else if ok {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }

        // 两条独立判据：条件一致性（跨度）与边际水平
        if !cell_coverages.is_empty() {
            let hi = cell_coverages.iter().copied().fold(f64::MIN, f64::max);
            let lo = cell_coverages.iter().copied().fold(f64::MAX, f64::min);
            let spread_ok = (hi - lo) <= SPREAD_TOLERANCE;
            all_pass &= spread_ok;
            println!(
                "    → 条件一致性：跨度 {:.1}pp {}（上限 {:.0}pp）",
                (hi - lo) * 100.0,
                if spread_ok { "PASS" } else { "FAIL" },
                SPREAD_TOLERANCE * 100.0
            );
        }
        if let Some(p) = band_score(
            &holdout,
            |o| (o.mu_h + prod_lo * o.sigma_h, o.mu_h + prod_hi * o.sigma_h),
            nominal,
        ) {
            let level_ok = p.coverage_gap().abs() <= LEVEL_TOLERANCE;
            all_pass &= level_ok;
            println!(
                "    → 边际水平：{:+.1}pp {}（上限 ±{:.0}pp）",
                p.coverage_gap() * 100.0,
                if level_ok { "PASS" } else { "FAIL" },
                LEVEL_TOLERANCE * 100.0
            );
        }

        let tail = (1.0 - nominal) / 2.0;
        let prod = band_score(
            &holdout,
            |o| (o.mu_h + prod_lo * o.sigma_h, o.mu_h + prod_hi * o.sigma_h),
            nominal,
        );
        if let (Some(p), false) = (prod, cell_coverages.is_empty()) {
            let worst = cell_coverages
                .iter()
                .map(|c| (c - nominal).abs())
                .fold(0.0_f64, f64::max);
            let hi_cell = cell_coverages.iter().copied().fold(f64::MIN, f64::max);
            let lo_cell = cell_coverages.iter().copied().fold(f64::MAX, f64::min);
            out.push(Summary {
                horizon,
                method,
                nominal,
                prod_coverage: p.coverage,
                prod_max_gap: worst * 100.0,
                prod_spread: (hi_cell - lo_cell) * 100.0,
                prod_pinball: p.pinball,
                prod_width: p.mean_width * 50.0,
                full_lo: empirical_quantile(&full_resid, tail).unwrap_or(f64::NAN),
                full_hi: empirical_quantile(&full_resid, 1.0 - tail).unwrap_or(f64::NAN),
            });
        }
    }
    (all_pass, cells)
}

/// 选型表：一眼看出该用哪个 σ 估计器、z 该定多少
fn print_summary_table(rows: &[Summary]) {
    for nominal in [0.80_f64, 0.90, 0.95] {
        println!("\n{}", "=".repeat(96));
        println!("选型表（名义 {:.0}%，全部为留出段）", nominal * 100.0);
        println!(
            "{:<4} {:<12} {:>8} {:>10} {:>10} {:>10} {:>9} {:>16}",
            "h", "方法", "覆盖率", "最大偏差", "高低跨度", "pinball", "半宽", "全样本分位(下/上)"
        );
        println!("{}", "-".repeat(96));
        for r in rows.iter().filter(|r| (r.nominal - nominal).abs() < 1e-9) {
            println!(
                "{:<4} {:<12} {:>7.1}% {:>9.1}pp {:>9.1}pp {:>10.5} {:>8.2}% {:>7.3} /{:>6.3}",
                r.horizon,
                r.method,
                r.prod_coverage * 100.0,
                r.prod_max_gap,
                r.prod_spread,
                r.prod_pinball,
                r.prod_width,
                r.full_lo,
                r.full_hi
            );
        }
        println!(
            "「高低跨度」= 波动率最高与最低那一档覆盖率之差。它接近 0 才说明区间在各种波动状态下\n             都一样可信；生产当前的 realized20 在这一列上最差，这正是「概率不准」的直接来源。"
        );
    }
}

fn band_score(
    obs: &[Obs],
    band: impl Fn(&Obs) -> (f64, f64),
    nominal: f64,
) -> Option<biga_lib::prediction::calibration::IntervalScore> {
    let rows: Vec<(f64, f64, f64)> = obs
        .iter()
        .map(|o| {
            let (l, u) = band(o);
            (o.actual, l, u)
        })
        .collect();
    score_intervals(&rows, nominal)
}
