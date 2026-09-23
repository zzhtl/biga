//! 周期阶段模型实证：把「箱体 → 主升 → 高位横盘 → 下跌 → 箱体」的每一条说法拿到全库上核对，
//! 并产出 `prediction::analysis::cycle_phase::outlook` 里写死的常量。
//!
//! 用法：cargo run --release --example cycle_phase_study
//!
//! 口径：
//! - 2000-01-01 起的前复权日线（`cycle_phase::adjust`）。每只票前 250 根是状态机预热期
//!   （非典型下跌要回看 250 日高点），不计入阶段占比与前向收益。
//! - 训练 / 留出按 2016-01-01 切：训练段只收在切分日前**已有结果**的样本，留出段只收在切分日
//!   后**才开始**的样本，跨界的两边都不收。
//! - 状态机就是生产代码那一份（`track_phases`），这里不重新实现任何规则。
//! - 前向收益按 H 日步长取不重叠样本，否则相邻样本高度相关，样本量是虚的。

use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::{get_historical_data, get_symbols_with_min_bars};
use biga_lib::prediction::analysis::cycle_phase::{
    forward_adjust, track_phases, AdjustedBar, CycleCore, CycleKind, CycleParams, CyclePhase,
    CycleState, DrawdownEpisode, EpisodeOutcome, PhaseTrack,
};
use biga_lib::prediction::calibration::score_probabilities_against;
use biga_lib::utils::symbol::canonical_stock_symbol;
use chrono::NaiveDate;
use std::collections::{BTreeMap, BTreeSet};

const START_DATE: &str = "2000-01-01";
const MIN_BARS: usize = 500;
const WARMUP_BARS: usize = 250;
/// 回撤阈值网格：生产代码按它插值条件概率
const DRAWDOWN_GRID: [f64; 15] = [
    0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
];

fn split_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2016, 1, 1).expect("合法日期")
}

struct Stock {
    bars: Vec<AdjustedBar>,
    track: PhaseTrack,
}

impl Stock {
    fn date(&self, index: usize) -> NaiveDate {
        self.bars[index].date
    }
}

/// 比对外阶段更细的口径：把「无上下文箱体」和「下跌后的试探性底部」分开
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Fine {
    Box,
    Base,
    Markup,
    Top,
    Markdown,
}

impl Fine {
    fn of(state: &CycleState) -> Self {
        match state {
            CycleState::Box => Self::Box,
            CycleState::Base { .. } => Self::Base,
            CycleState::Rally { stalled: false, .. } => Self::Markup,
            CycleState::Rally { stalled: true, .. } => Self::Top,
            CycleState::Decline { .. } => Self::Markdown,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Box => "箱体(无上下文)",
            Self::Base => "箱体(下跌后试探底)",
            Self::Markup => "主升浪",
            Self::Top => "高位横盘",
            Self::Markdown => "下跌通道",
        }
    }

    const ALL: [Fine; 5] = [
        Self::Box,
        Self::Base,
        Self::Markup,
        Self::Top,
        Self::Markdown,
    ];
}

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");
    let raw_symbols = get_symbols_with_min_bars(MIN_BARS as i64, &pool)
        .await
        .expect("读取股票列表失败");
    // 000002 与 000002.SZ 是同一只票，按规范代码去重；取数时由仓库层解析最全的那个变体
    let symbols: BTreeSet<String> = raw_symbols
        .iter()
        .map(|s| canonical_stock_symbol(s))
        .collect();

    let mut stocks = Vec::new();
    let (mut total_bars, mut corporate_actions, mut unadjusted) = (0usize, 0usize, 0usize);
    let (mut first, mut last) = (NaiveDate::MAX, NaiveDate::MIN);
    for symbol in &symbols {
        let history = get_historical_data(symbol, START_DATE, "2099-12-31", &pool)
            .await
            .expect("读取历史数据失败");
        let series = forward_adjust(&history);
        if series.bars.len() < MIN_BARS {
            continue;
        }
        total_bars += series.bars.len();
        corporate_actions += series.corporate_actions;
        unadjusted += usize::from(!series.adjusted);
        first = first.min(series.bars[0].date);
        last = last.max(series.bars[series.bars.len() - 1].date);
        let track = track_phases(&series.bars, &CycleParams::default());
        stocks.push(Stock {
            bars: series.bars,
            track,
        });
    }

    println!("{}", "=".repeat(78));
    println!("A. 数据");
    println!(
        "  股票 {} 只（去重前 {}），K 线 {} 根，{} ~ {}",
        stocks.len(),
        raw_symbols.len(),
        total_bars,
        first,
        last
    );
    println!("  识别并复权的除权除息 {corporate_actions} 次；change 字段不可信而未复权的 {unadjusted} 只");
    if stocks.is_empty() {
        eprintln!("库里没有足够历史的标的，先补数据。");
        std::process::exit(1);
    }

    time_share(&stocks);
    let rally = rally_stats(&stocks);
    let top = run_resolution(&stocks, Fine::Top);
    let markup = run_resolution(&stocks, Fine::Markup);
    let survival = drawdown_depth(&stocks);
    let decline_days = decline_duration(&stocks);
    let base = run_resolution(&stocks, Fine::Base);
    forward_direction(&stocks);
    sensitivity(&stocks);
    print_constants(&survival, &top, &markup, &base, &rally, &decline_days);
}

// =============================================================================
// B. 时间占比
// =============================================================================

fn time_share(stocks: &[Stock]) {
    println!("\n{}", "=".repeat(78));
    println!("B. 各阶段时间占比（模型说法：大多数时间在箱体里）");
    let mut counts: BTreeMap<Fine, usize> = BTreeMap::new();
    for stock in stocks {
        for state in stock.track.states.iter().skip(WARMUP_BARS) {
            *counts.entry(Fine::of(state)).or_default() += 1;
        }
    }
    let total: usize = counts.values().sum();
    for fine in Fine::ALL {
        let n = counts.get(&fine).copied().unwrap_or(0);
        println!("  {:18} {:>6.1}%", fine.label(), pct(n, total));
    }
    let boxed = counts.get(&Fine::Box).copied().unwrap_or(0)
        + counts.get(&Fine::Base).copied().unwrap_or(0);
    println!("  → 对外口径「箱体震荡」合计 {:.1}%", pct(boxed, total));
}

// =============================================================================
// C. 主升
// =============================================================================

/// 交易日与涨幅的四分位 (q25, q50, q75)
struct Quartiles {
    days: [f64; 3],
    gain: [f64; 3],
}

impl Quartiles {
    fn of(sorted_days: &[f64], sorted_gains: &[f64]) -> Self {
        let q = |v: &[f64]| [quantile(v, 0.25), quantile(v, 0.5), quantile(v, 0.75)];
        Self {
            days: q(sorted_days),
            gain: q(sorted_gains),
        }
    }
}

struct RallyStats {
    full: Quartiles,
    first_leg: Quartiles,
}

/// 完整周期 = 典型周期里结局不是「洗盘」的回撤事件，其 base→peak 就是整段主升
fn completed_typical(stocks: &[Stock]) -> impl Iterator<Item = (&Stock, &DrawdownEpisode)> {
    stocks.iter().flat_map(|stock| {
        stock
            .track
            .episodes
            .iter()
            .filter(|ep| ep.kind == CycleKind::Markup)
            .filter(|ep| matches!(ep.end, Some((_, outcome)) if outcome != EpisodeOutcome::NewHigh))
            .map(move |ep| (stock, ep))
    })
}

fn rally_stats(stocks: &[Stock]) -> RallyStats {
    println!("\n{}", "=".repeat(78));
    println!("C. 主升（模型说法：疯狂上涨几周甚至 1-2 个月，几周概率最大）");
    let mut days = Vec::new();
    let mut gains = Vec::new();
    for (_, ep) in completed_typical(stocks) {
        let Some(base) = ep.base else { continue };
        days.push((ep.peak.index - base.index) as f64);
        gains.push(ep.peak.price / base.price - 1.0);
    }
    days.sort_by(f64::total_cmp);
    gains.sort_by(f64::total_cmp);
    let n = days.len();
    println!("  完整周期 {n} 轮（起涨点 → 最终峰值，含中途洗盘）");
    println!(
        "  主升交易日  q10/q25/q50/q75/q90 = {}",
        fmt_quantiles(&days, &[0.1, 0.25, 0.5, 0.75, 0.9], 0)
    );
    println!(
        "  主升涨幅    q10/q25/q50/q75/q90 = {}",
        fmt_quantiles_pct(&gains, &[0.1, 0.25, 0.5, 0.75, 0.9])
    );
    let buckets = [
        (0.0, 20.0, "≤4 周"),
        (20.0, 40.0, "1-2 个月"),
        (40.0, 80.0, "2-4 个月"),
        (80.0, f64::MAX, "4 个月以上"),
    ];
    for (lo, hi, label) in buckets {
        let c = days.iter().filter(|&&d| d > lo && d <= hi).count();
        println!("    {label:10} {:>5.1}%", pct(c, n));
    }
    RallyStats {
        full: Quartiles::of(&days, &gains),
        first_leg: first_leg(stocks),
    }
}

/// 首段主升：新一轮周期从起涨点到第一次转入横盘或跌破 −20% 时的峰值。
///
/// 上面的「完整主升」把横盘后再创新高、洗盘后再创新高的多段行情合成一段；模型说的
/// "疯狂上涨几周"更接近这里的第一段。
fn first_leg(stocks: &[Stock]) -> Quartiles {
    let mut days = Vec::new();
    let mut gains = Vec::new();
    for stock in stocks {
        let mut open_leg = false;
        let mut prev_base = None;
        for state in &stock.track.states {
            let base = match state {
                CycleState::Rally { cycle, .. }
                | CycleState::Decline { cycle, .. }
                | CycleState::Base { cycle, .. } => cycle.base,
                CycleState::Box => None,
            };
            if let CycleState::Rally { cycle, stalled } = state {
                // 起涨点变了（或之前没有周期）= 新一轮
                if cycle.base.is_some() && cycle.base != prev_base {
                    open_leg = true;
                }
                if open_leg && *stalled {
                    record_leg(cycle, &mut days, &mut gains);
                    open_leg = false;
                }
            } else if let (true, CycleState::Decline { cycle, .. }) = (open_leg, state) {
                record_leg(cycle, &mut days, &mut gains);
                open_leg = false;
            }
            prev_base = base;
        }
    }
    days.sort_by(f64::total_cmp);
    gains.sort_by(f64::total_cmp);
    let n = days.len();
    println!("\n  首段主升 {n} 段（起涨点 → 第一次横盘或跌破时的峰值）");
    println!(
        "  首段交易日  q10/q25/q50/q75/q90 = {}",
        fmt_quantiles(&days, &[0.1, 0.25, 0.5, 0.75, 0.9], 0)
    );
    println!(
        "  首段涨幅    q10/q25/q50/q75/q90 = {}",
        fmt_quantiles_pct(&gains, &[0.1, 0.25, 0.5, 0.75, 0.9])
    );
    let buckets = [
        (0.0, 20.0, "≤4 周"),
        (20.0, 40.0, "1-2 个月"),
        (40.0, 80.0, "2-4 个月"),
        (80.0, f64::MAX, "4 个月以上"),
    ];
    for (lo, hi, label) in buckets {
        let c = days.iter().filter(|&&d| d > lo && d <= hi).count();
        println!("    {label:10} {:>5.1}%", pct(c, n));
    }
    Quartiles::of(&days, &gains)
}

fn record_leg(cycle: &CycleCore, days: &mut Vec<f64>, gains: &mut Vec<f64>) {
    if let Some(base) = cycle.base {
        days.push((cycle.peak.index - base.index) as f64);
        gains.push(cycle.peak.price / base.price - 1.0);
    }
}

// =============================================================================
// D / F. 阶段结局
// =============================================================================

/// 某阶段的一段连续运行结束后转去了哪里：(全样本计数, 训练段计数, 留出段计数)
type Resolution = BTreeMap<Fine, (usize, usize, usize)>;

fn run_resolution(stocks: &[Stock], of: Fine) -> Resolution {
    let mut res: Resolution = BTreeMap::new();
    let mut still_open = 0usize;
    for stock in stocks {
        let states = &stock.track.states;
        let mut start = 0usize;
        for t in 1..=states.len() {
            if t < states.len() && Fine::of(&states[t]) == Fine::of(&states[start]) {
                continue;
            }
            if Fine::of(&states[start]) == of {
                if t == states.len() {
                    still_open += 1;
                } else {
                    let next = Fine::of(&states[t]);
                    let entry = res.entry(next).or_default();
                    entry.0 += 1;
                    if stock.date(t) < split_date() {
                        entry.1 += 1;
                    } else if stock.date(start) >= split_date() {
                        entry.2 += 1;
                    }
                }
            }
            start = t;
        }
    }

    println!("\n{}", "=".repeat(78));
    match of {
        Fine::Top => println!("D. 高位横盘的结局（模型说法：横盘突破不了就快速下跌）"),
        Fine::Markup => println!("D'. 主升的结局（先横盘，还是不经横盘直接跌破 −20%）"),
        Fine::Base => println!("F. 下跌后试探性底部的结局（模型说法：下跌一波接着一波）"),
        _ => println!("{} 的结局", of.label()),
    }
    let totals = res
        .values()
        .fold((0, 0, 0), |a, v| (a.0 + v.0, a.1 + v.1, a.2 + v.2));
    for (next, (all, train, test)) in &res {
        let meaning = match (of, next) {
            (Fine::Top, Fine::Markup) => "先创新高（横盘向上突破）",
            (Fine::Top, Fine::Markdown) => "先跌破峰值 −20%",
            (Fine::Markup, Fine::Top) => "转入高位横盘",
            (Fine::Markup, Fine::Markdown) => "不经横盘直接跌破 −20%",
            (Fine::Base, Fine::Markdown) => "底部被跌破，同一轮下跌继续",
            (Fine::Base, Fine::Markup) => "底部里起新一轮主升",
            (Fine::Base, Fine::Box) => "底部守住（守满期限或收复前高），本轮结束",
            _ => "",
        };
        println!(
            "  → {:14} 全样本 {:>5.1}% ({all:>5})  训练 {:>5.1}% ({train:>4})  留出 {:>5.1}% ({test:>4})  {meaning}",
            next.label(),
            pct(*all, totals.0),
            pct(*train, totals.1),
            pct(*test, totals.2),
        );
    }
    println!("  截至数据末尾仍在该阶段的 {still_open} 段未计入");
    res
}

// =============================================================================
// E. 下跌深度
// =============================================================================

/// 典型回撤事件在各阈值上「最终最大回撤达到该阈值」的事件数（全样本，已结束的事件）
struct Survival {
    markup: Vec<usize>,
}

fn reached(episodes: &[&DrawdownEpisode], threshold: f64) -> usize {
    // 1e-9：0.20 这类阈值在浮点上恰好取等时算"达到"
    episodes
        .iter()
        .filter(|ep| -ep.max_drawdown() >= threshold - 1e-9)
        .count()
}

fn drawdown_depth(stocks: &[Stock]) -> Survival {
    println!("\n{}", "=".repeat(78));
    println!("E. 下跌深度（模型说法：跌 50%-70% 大概率，90% 小概率）");
    let split = split_date();
    let mut survival = Survival { markup: Vec::new() };
    for kind in [CycleKind::Markup, CycleKind::RangeHigh] {
        let closed: Vec<(&Stock, &DrawdownEpisode)> = stocks
            .iter()
            .flat_map(|s| s.track.episodes.iter().map(move |ep| (s, ep)))
            .filter(|(_, ep)| ep.kind == kind && ep.end.is_some())
            .collect();
        let all: Vec<&DrawdownEpisode> = closed.iter().map(|(_, ep)| *ep).collect();
        let train: Vec<&DrawdownEpisode> = closed
            .iter()
            .filter(|(s, ep)| ep.end.is_some_and(|(end, _)| s.date(end) < split))
            .map(|(_, ep)| *ep)
            .collect();
        let test: Vec<&DrawdownEpisode> = closed
            .iter()
            .filter(|(s, ep)| s.date(ep.start) >= split)
            .map(|(_, ep)| *ep)
            .collect();
        let label = match kind {
            CycleKind::Markup => "典型（暴涨后）",
            CycleKind::RangeHigh => "非典型（未经暴涨）",
        };
        let mut depths: Vec<f64> = all.iter().map(|ep| -ep.max_drawdown()).collect();
        depths.sort_by(f64::total_cmp);
        println!(
            "\n  {label}回撤事件 {} 次（训练 {} / 留出 {}），最终最大回撤 q10/q25/q50/q75/q90 = {}",
            all.len(),
            train.len(),
            test.len(),
            fmt_quantiles_pct(&depths, &[0.1, 0.25, 0.5, 0.75, 0.9])
        );
        let entry = match kind {
            CycleKind::Markup => 0.20,
            CycleKind::RangeHigh => 0.30,
        };
        println!(
            "  事件内（含洗盘）: 跌满 50% {:.1}%，跌满 70% {:.1}%，跌满 90% {:.2}%",
            pct(reached(&all, 0.5), reached(&all, entry)),
            pct(reached(&all, 0.7), reached(&all, entry)),
            pct(reached(&all, 0.9), reached(&all, entry)),
        );
        if kind == CycleKind::Markup {
            let no_shakeout: Vec<&DrawdownEpisode> = all
                .iter()
                .copied()
                .filter(|ep| !matches!(ep.end, Some((_, EpisodeOutcome::NewHigh))))
                .collect();
            println!(
                "  对照（错误口径：剔除洗盘）: 跌满 50% {:.1}%，跌满 70% {:.1}% ← 分母漏掉洗盘会高估深跌",
                pct(reached(&no_shakeout, 0.5), no_shakeout.len()),
                pct(reached(&no_shakeout, 0.7), no_shakeout.len()),
            );
        }
        println!("  条件概率 P(最终跌满 Y | 已跌 X)：训练 → 留出");
        for x in [entry, 0.30, 0.40, 0.50, 0.60] {
            if x < entry {
                continue;
            }
            let mut line = format!("    已跌 {:>2.0}%:", x * 100.0);
            for y in [0.5, 0.7, 0.9] {
                if y <= x + 1e-9 {
                    continue;
                }
                let tr = ratio(reached(&train, y), reached(&train, x));
                let te = ratio(reached(&test, y), reached(&test, x));
                line.push_str(&format!(
                    "  →{:.0}%: {:.3} → {:.3} (n={}/{})",
                    y * 100.0,
                    tr,
                    te,
                    reached(&train, x),
                    reached(&test, x)
                ));
            }
            println!("{line}");
        }
        if kind == CycleKind::Markup {
            survival.markup = DRAWDOWN_GRID.iter().map(|&g| reached(&all, g)).collect();
        }
    }
    survival
}

/// 返回 (峰值 → 确认下跌, 峰值 → 谷底) 两组交易日四分位
fn decline_duration(stocks: &[Stock]) -> ([f64; 3], [f64; 3]) {
    let mut to_break: Vec<f64> = completed_typical(stocks)
        .map(|(_, ep)| (ep.start - ep.peak.index) as f64)
        .collect();
    to_break.sort_by(f64::total_cmp);
    println!(
        "\n  典型完整周期 峰值 → 确认下跌(−20%)交易日 q10/q25/q50/q75/q90 = {}",
        fmt_quantiles(&to_break, &[0.1, 0.25, 0.5, 0.75, 0.9], 0)
    );
    let mut days: Vec<f64> = completed_typical(stocks)
        .map(|(_, ep)| (ep.trough.index - ep.peak.index) as f64)
        .collect();
    days.sort_by(f64::total_cmp);
    println!(
        "\n  典型完整周期 峰值 → 谷底交易日 q10/q25/q50/q75/q90 = {}",
        fmt_quantiles(&days, &[0.1, 0.25, 0.5, 0.75, 0.9], 0)
    );
    let q = |v: &[f64]| [quantile(v, 0.25), quantile(v, 0.5), quantile(v, 0.75)];
    (q(&to_break), q(&days))
}

// =============================================================================
// G. 阶段对未来涨跌方向的预测力
// =============================================================================

fn forward_direction(stocks: &[Stock]) {
    println!("\n{}", "=".repeat(78));
    println!(
        "G. 阶段对未来涨跌方向有没有预测力（训练段各阶段上涨频率 → 留出段打 Brier Skill Score）"
    );
    let split = split_date();
    for horizon in [5usize, 20, 60] {
        let mut train: BTreeMap<Fine, (usize, usize)> = BTreeMap::new();
        let mut test: Vec<(Fine, bool)> = Vec::new();
        for stock in stocks {
            let mut t = WARMUP_BARS;
            while t + horizon < stock.bars.len() {
                let ret = stock.bars[t + horizon].close / stock.bars[t].close - 1.0;
                if ret.is_finite() && ret.abs() >= 1e-4 {
                    let fine = Fine::of(&stock.track.states[t]);
                    if stock.date(t + horizon) < split {
                        let e = train.entry(fine).or_default();
                        e.0 += usize::from(ret > 0.0);
                        e.1 += 1;
                    } else if stock.date(t) >= split {
                        test.push((fine, ret > 0.0));
                    }
                }
                t += horizon;
            }
        }
        let (ups, n) = train.values().fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));
        let climatology = ratio(ups, n);
        let rate = |fine: Fine| {
            train
                .get(&fine)
                .filter(|(_, n)| *n > 0)
                .map_or(climatology, |(u, n)| ratio(*u, *n))
        };
        let model: Vec<(f64, bool)> = test.iter().map(|&(f, up)| (rate(f), up)).collect();
        let clim: Vec<(f64, bool)> = test.iter().map(|&(_, up)| (climatology, up)).collect();
        let realized = ratio(test.iter().filter(|(_, up)| *up).count(), test.len());
        let (Some(m), Some(c)) = (
            score_probabilities_against(&model, realized),
            score_probabilities_against(&clim, realized),
        ) else {
            println!("  H={horizon}: 样本不足");
            continue;
        };
        let bss = 1.0 - m.brier / c.brier;
        println!(
            "\n  H={horizon:>2}  训练基率 {:.3}  留出实际 {:.3}  BSS(阶段 vs 训练基率) = {bss:+.4}  {}",
            climatology,
            realized,
            if bss > 0.001 { "【有技能】" } else { "【无技能】" }
        );
        for fine in Fine::ALL {
            let (tu, tn) = train.get(&fine).copied().unwrap_or((0, 0));
            let te: Vec<bool> = test
                .iter()
                .filter(|(f, _)| *f == fine)
                .map(|(_, up)| *up)
                .collect();
            println!(
                "    {:18} 训练 P(涨)={:.3} (n={tn:>5})  留出 P(涨)={:.3} (n={:>5})",
                fine.label(),
                ratio(tu, tn),
                ratio(te.iter().filter(|u| **u).count(), te.len()),
                te.len()
            );
        }
    }
}

// =============================================================================
// H. 参数敏感性
// =============================================================================

fn sensitivity(stocks: &[Stock]) {
    println!("\n{}", "=".repeat(78));
    println!("H. 参数敏感性（结论若随参数翻转，就不该写死）");
    println!(
        "  {:>6} {:>6} | {:>7} {:>9} {:>11} {:>11} {:>11}",
        "主升", "下跌", "箱体占比", "横盘先跌", "P(50|入跌)", "P(70|入跌)", "P(50|已跌40)"
    );
    for surge_gain in [0.4, 0.5, 0.6] {
        for breakdown in [0.15, 0.20, 0.25] {
            let params = CycleParams {
                surge_gain,
                breakdown,
                ..CycleParams::default()
            };
            let (mut boxed, mut total) = (0usize, 0usize);
            let (mut top_down, mut top_all) = (0usize, 0usize);
            let mut episodes = Vec::new();
            for stock in stocks {
                let track = track_phases(&stock.bars, &params);
                for s in track.states.iter().skip(WARMUP_BARS) {
                    total += 1;
                    boxed += usize::from(s.phase() == CyclePhase::Box);
                }
                for w in track.states.windows(2) {
                    if w[0].phase() == CyclePhase::Top && w[1].phase() != CyclePhase::Top {
                        top_all += 1;
                        top_down += usize::from(w[1].phase() == CyclePhase::Markdown);
                    }
                }
                episodes.extend(
                    track
                        .episodes
                        .into_iter()
                        .filter(|ep| ep.kind == CycleKind::Markup && ep.end.is_some()),
                );
            }
            let refs: Vec<&DrawdownEpisode> = episodes.iter().collect();
            println!(
                "  {:>5.0}% {:>5.0}% | {:>7.1}% {:>8.1}% {:>11.3} {:>11.3} {:>11.3}",
                surge_gain * 100.0,
                breakdown * 100.0,
                pct(boxed, total),
                pct(top_down, top_all),
                ratio(reached(&refs, 0.5), reached(&refs, breakdown)),
                ratio(reached(&refs, 0.7), reached(&refs, breakdown)),
                ratio(reached(&refs, 0.5), reached(&refs, 0.4)),
            );
        }
    }
}

// =============================================================================
// I. 常量
// =============================================================================

fn print_constants(
    survival: &Survival,
    top: &Resolution,
    markup: &Resolution,
    base: &Resolution,
    rally: &RallyStats,
    decline_days: &([f64; 3], [f64; 3]),
) {
    let get = |r: &Resolution, f: Fine| r.get(&f).map_or(0, |v| v.0);
    println!("\n{}", "=".repeat(78));
    println!("I. 写入 cycle_phase/outlook.rs 的常量（全样本；非典型下跌样本外不稳定，不写入）");
    let grid = DRAWDOWN_GRID
        .iter()
        .zip(&survival.markup)
        .map(|(g, c)| format!("({g:.2}, {c})"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("const TYPICAL_DRAWDOWN_REACHED: [(f64, usize); 15] = [{grid}];");
    println!(
        "const TOP_NEW_HIGH_FIRST: usize = {};",
        get(top, Fine::Markup)
    );
    println!(
        "const TOP_BREAKDOWN_FIRST: usize = {};",
        get(top, Fine::Markdown)
    );
    println!("const MARKUP_TO_TOP: usize = {};", get(markup, Fine::Top));
    println!(
        "const MARKUP_DIRECT_BREAKDOWN: usize = {};",
        get(markup, Fine::Markdown)
    );
    println!(
        "const BASE_REBROKEN: usize = {};",
        get(base, Fine::Markdown)
    );
    // 底部里起的新主升可能当根就已满足"横盘"条件，两者都算新一轮
    println!(
        "const BASE_NEW_MARKUP: usize = {};",
        get(base, Fine::Markup) + get(base, Fine::Top)
    );
    println!("const BASE_HELD: usize = {};", get(base, Fine::Box));
    let arr = |v: &[f64; 3], decimals: usize| {
        v.iter()
            .map(|x| format!("{x:.decimals$}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "const FIRST_LEG_DAYS: [f64; 3] = [{}];",
        arr(&rally.first_leg.days, 1)
    );
    println!(
        "const FIRST_LEG_GAIN: [f64; 3] = [{}];",
        arr(&rally.first_leg.gain, 2)
    );
    println!(
        "const FULL_RALLY_DAYS: [f64; 3] = [{}];",
        arr(&rally.full.days, 1)
    );
    println!(
        "const FULL_RALLY_GAIN: [f64; 3] = [{}];",
        arr(&rally.full.gain, 2)
    );
    println!(
        "const PEAK_TO_BREAKDOWN_DAYS: [f64; 3] = [{}];",
        arr(&decline_days.0, 1)
    );
    println!(
        "const DECLINE_DAYS: [f64; 3] = [{}];",
        arr(&decline_days.1, 1)
    );
}

// =============================================================================
// 工具
// =============================================================================

fn pct(part: usize, whole: usize) -> f64 {
    ratio(part, whole) * 100.0
}

fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        f64::NAN
    } else {
        part as f64 / whole as f64
    }
}

/// 与原型一致的最近秩分位：`sorted[min(n−1, ⌊q·n⌋)]`
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn fmt_quantiles(sorted: &[f64], qs: &[f64], decimals: usize) -> String {
    qs.iter()
        .map(|&q| format!("{:.*}", decimals, quantile(sorted, q)))
        .collect::<Vec<_>>()
        .join(" / ")
}

fn fmt_quantiles_pct(sorted: &[f64], qs: &[f64]) -> String {
    qs.iter()
        .map(|&q| format!("{:.0}%", quantile(sorted, q) * 100.0))
        .collect::<Vec<_>>()
        .join(" / ")
}
