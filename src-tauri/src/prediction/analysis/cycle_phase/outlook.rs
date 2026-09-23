//! 按周期模型推演：当前阶段的关键价位、下一步的触发条件，以及全库历史上的结局频率。
//!
//! # 常量出处
//!
//! 本文件的计数与分位全部是 `examples/cycle_phase_study.rs` 「I.」节的输出：本库 131 只股票、
//! 2000-01 ~ 2026-05 前复权日线，状态机即 [`super::track_phases`] 的默认参数。改了状态机的
//! 规则或参数必须重跑并整段替换——常量与规则是配套的，单改一边数字就不再对应任何东西。
//!
//! # 为什么只给这几类概率
//!
//! - **阶段本身不预测涨跌方向。** 用训练段（2016 年前）各阶段的上涨频率去报留出段，5/20/60 日
//!   Brier Skill Score 为 −0.0001 / −0.0002 / +0.0003；主升、横盘在训练段偏涨、留出段偏跌，
//!   方向整个翻了。所以这里没有"某阶段上涨概率"，阶段也不进点预测。
//! - **下跌深度的条件频率是稳的。** P(最终跌满 50% | 已跌 40%) 训练 0.668 / 留出 0.638，
//!   P(跌满 70% | 已跌 50%) 0.289 / 0.224。留出段普遍低 1～11pp（2016 年后深跌更少），全样本数
//!   偏向高估深跌——对持仓者是保守的方向。
//! - **试探性底部被再次跌破的比例最稳**：训练 45.4% / 留出 45.4%。
//! - **非典型下跌（未经暴涨）不给深度概率。** 训练 / 留出差得太远（已跌 30% 再跌满 50%：
//!   48% / 31%），写死只会误导。
//! - 参数敏感性（主升 +40/50/60% × 下跌 −15/20/25%）下，上面每条结论的方向都不变。

use super::adjust::{AdjustedBar, AdjustedSeries};
use super::machine::{
    CycleCore, CycleKind, CycleParams, CycleState, DrawdownEpisode, Episode, EpisodeOutcome,
    PhaseTrack, PricePoint,
};
use super::{CycleAnalysis, CycleEpisode, CycleLevel, CycleOdds};

/// 典型回撤事件（暴涨后跌破峰值 −20%）中，最终最大回撤达到各阈值的事件数。
///
/// **分母含洗盘**（跌破 −20% 后又再创新高的事件）：站在 −25% 分不清是洗盘还是大跌的开头，
/// 把洗盘排除在外会把「跌满 50%」从 30.8% 高估成 46.5%。
const TYPICAL_DRAWDOWN_REACHED: [(f64, usize); 15] = [
    (0.20, 1183),
    (0.25, 1019),
    (0.30, 818),
    (0.35, 667),
    (0.40, 552),
    (0.45, 447),
    (0.50, 364),
    (0.55, 279),
    (0.60, 206),
    (0.65, 131),
    (0.70, 88),
    (0.75, 53),
    (0.80, 25),
    (0.85, 9),
    (0.90, 0),
];
const DEPTH_STABILITY: &str =
    "留出段（2016 年后）比训练段普遍低 1～11 个百分点，全样本数偏向高估深跌";

/// 高位横盘的结局：先再创新高 / 先跌破峰值 −20%
const TOP_NEW_HIGH_FIRST: usize = 1008;
const TOP_BREAKDOWN_FIRST: usize = 723;
const TOP_STABILITY: &str = "先跌破的比例：训练段 38.5% → 留出段 46.2%，近年横盘后下跌更常见";

/// 主升的结局：转入高位横盘 / 不经横盘直接跌破峰值 −20%
const MARKUP_TO_TOP: usize = 1716;
const MARKUP_DIRECT_BREAKDOWN: usize = 497;
const MARKUP_STABILITY: &str = "训练段 21.6% → 留出段 23.2%";

/// 下跌后试探性底部的结局：被再次跌破 / 起新一轮主升 / 守住（守满期限或收复前高）
const BASE_REBROKEN: usize = 679;
const BASE_NEW_MARKUP: usize = 199;
const BASE_HELD: usize = 621;
const BASE_STABILITY: &str = "被跌破的比例：训练段 45.4% → 留出段 45.4%";

/// 首段主升（起涨点 → 第一次横盘或跌破时的峰值）交易日与涨幅的 q25/q50/q75
const FIRST_LEG_DAYS: [f64; 3] = [26.0, 37.0, 45.0];
const FIRST_LEG_GAIN: [f64; 3] = [0.62, 0.71, 0.90];
/// 整轮主升（含横盘后续涨、洗盘后续涨）
const FULL_RALLY_DAYS: [f64; 3] = [40.0, 68.0, 148.0];
const FULL_RALLY_GAIN: [f64; 3] = [0.71, 1.04, 2.05];
/// 典型完整周期：峰值 → 确认下跌（−20%）、峰值 → 最终谷底 的交易日 q25/q50/q75
const PEAK_TO_BREAKDOWN_DAYS: [f64; 3] = [7.0, 13.0, 25.0];
const DECLINE_DAYS: [f64; 3] = [39.0, 108.0, 227.0];

/// 箱体区间的回看窗口（约半年）
const BOX_WINDOW: usize = 120;
/// 本股历史回撤事件最多展示几条
const MAX_HISTORY: usize = 8;
/// 模型的下跌目标：峰值 −50% / −70% / −90%
const MODEL_TARGETS: [(f64, &str); 3] = [
    (0.5, "模型下跌目标区上沿（模型认为大概率到达）"),
    (0.7, "模型下跌目标区下沿"),
    (0.9, "模型认为的小概率极端情形"),
];

pub(super) fn build(
    series: &AdjustedSeries,
    track: &PhaseTrack,
    params: &CycleParams,
) -> Option<CycleAnalysis> {
    let last = series.bars.len().checked_sub(1)?;
    let state = *track.states.get(last)?;
    let since = track.current_phase_start()?;
    let mut report = Report {
        bars: &series.bars,
        p: params,
        last,
        price: series.bars[last].close,
        levels: Vec::new(),
        odds: Vec::new(),
        facts: Vec::new(),
    };
    let (cycle_kind, summary) = match state {
        CycleState::Box => (None, report.plain_box()),
        CycleState::Base {
            cycle,
            episode,
            since: base_since,
        } => (
            Some(cycle.kind),
            report.tentative_base(cycle, episode, base_since),
        ),
        CycleState::Rally {
            cycle,
            stalled: false,
        } => (Some(cycle.kind), report.markup(cycle)),
        CycleState::Rally {
            cycle,
            stalled: true,
        } => (Some(cycle.kind), report.top(cycle)),
        CycleState::Decline { cycle, episode } => {
            (Some(cycle.kind), report.markdown(cycle, episode))
        }
    };

    let phase = state.phase();
    Some(CycleAnalysis {
        phase,
        phase_label: phase.label().to_string(),
        phase_since: report.date(since),
        days_in_phase: last - since + 1,
        cycle_kind,
        tentative_base: matches!(state, CycleState::Base { .. }),
        summary,
        history: history(&series.bars, &track.episodes),
        key_levels: report.levels,
        odds: report.odds,
        facts: report.facts,
        price_basis: price_basis(series),
        method_note: method_note(params),
    })
}

struct Report<'a> {
    bars: &'a [AdjustedBar],
    p: &'a CycleParams,
    last: usize,
    price: f64,
    levels: Vec<CycleLevel>,
    odds: Vec<CycleOdds>,
    facts: Vec<String>,
}

impl Report<'_> {
    fn date(&self, index: usize) -> String {
        format_date(self.bars, index)
    }

    fn level(&mut self, label: &str, price: f64, meaning: impl Into<String>) {
        self.levels.push(CycleLevel {
            label: label.to_string(),
            price,
            distance_percent: (price / self.price - 1.0) * 100.0,
            meaning: meaning.into(),
        });
    }

    fn odds(&mut self, label: impl Into<String>, hits: usize, samples: usize, note: &str) {
        if samples == 0 {
            return;
        }
        let note = if hits == 0 {
            "样本中从未出现；本库以行业龙头为主，小盘股、ST 股的极端跌幅会更常见".to_string()
        } else {
            note.to_string()
        };
        self.odds.push(CycleOdds {
            label: label.into(),
            probability: hits as f64 / samples as f64,
            hits,
            samples,
            note,
        });
    }

    /// 下一根收盘达到多少即确认主升。状态机用的窗口含当根，这里假设下一根不再创新低。
    fn surge_trigger(&self) -> f64 {
        let from = (self.last + 1).saturating_sub(self.p.surge_window);
        lowest_low(self.bars, from, self.last) * (1.0 + self.p.surge_gain)
    }

    fn surge_meaning(&self) -> String {
        format!(
            "收盘站上即按模型确认主升（近 {} 个交易日最低价 +{:.0}%）",
            self.p.surge_window,
            self.p.surge_gain * 100.0
        )
    }

    /// 非典型下跌的触发价：收盘须同时不高于一年内高点 × (1 − range_breakdown)，且低于近
    /// `new_low_window` 日的最低收盘。
    fn range_warning(&self) -> Option<f64> {
        let p = self.p;
        if self.last + 1 < p.range_window.max(p.new_low_window) {
            return None;
        }
        let high = highest_high(self.bars, self.last + 1 - p.range_window, self.last);
        let low_close = self.bars[self.last + 1 - p.new_low_window..=self.last]
            .iter()
            .map(|bar| bar.close)
            .fold(f64::INFINITY, f64::min);
        Some((high * (1.0 - p.range_breakdown)).min(low_close))
    }

    fn plain_box(&mut self) -> String {
        let from = (self.last + 1).saturating_sub(BOX_WINDOW);
        let days = self.last - from + 1;
        let top = highest_high(self.bars, from, self.last);
        let bottom = lowest_low(self.bars, from, self.last);
        let position = if top > bottom {
            (self.price - bottom) / (top - bottom)
        } else {
            0.5
        };
        let trigger = self.surge_trigger();
        self.level("箱体上沿", top, format!("近 {days} 个交易日最高价"));
        self.level("箱体下沿", bottom, format!("近 {days} 个交易日最低价"));
        self.level("主升确认价", trigger, self.surge_meaning());
        let warning = self.range_warning();
        if let Some(warning) = warning {
            self.level(
                "深跌警戒价",
                warning,
                format!(
                    "收盘跌破即按模型转入下跌通道（较一年内高点 −{:.0}% 且创 {} 日收盘新低）",
                    self.p.range_breakdown * 100.0,
                    self.p.new_low_window
                ),
            );
        }
        self.facts.push(format!(
            "近 {days} 个交易日振幅 {:.0}%，现价位于箱体 {:.0}% 分位",
            (top / bottom - 1.0) * 100.0,
            position * 100.0
        ));
        let downside = warning.map_or(String::new(), |w| {
            format!("；反过来，收盘跌破 {w:.2} 则按模型转入下跌通道")
        });
        format!(
            "箱体震荡：近 {days} 个交易日在 {bottom:.2}～{top:.2} 之间运行，现价处在箱体 {:.0}% 分位。\
             按模型，箱体是主升前的蓄势期，但突破时点无法预判——收盘站上 {trigger:.2}（距现价 {:+.1}%）\
             才确认进入主升浪{downside}。",
            position * 100.0,
            (trigger / self.price - 1.0) * 100.0,
        )
    }

    fn tentative_base(&mut self, cycle: CycleCore, episode: Episode, base_since: usize) -> String {
        let (peak, trough) = (cycle.peak, episode.trough);
        let max_drawdown = 1.0 - trough.price / peak.price;
        let quiet_days = self.last - trough.index;
        let remaining = (base_since + self.p.base_expiry_days).saturating_sub(self.last);
        let trigger = self.surge_trigger();
        self.level(
            "本轮峰值",
            peak.price,
            format!("{} 的周期顶，回撤从这里量", self.date(peak.index)),
        );
        self.level(
            "谷底（破位价）",
            trough.price,
            "收盘跌破即回到下跌通道，本轮回撤继续",
        );
        self.level("主升确认价", trigger, self.surge_meaning());
        self.targets(peak, trough);

        let total = BASE_REBROKEN + BASE_NEW_MARKUP + BASE_HELD;
        self.odds(
            "底部被再次跌破、下跌继续",
            BASE_REBROKEN,
            total,
            BASE_STABILITY,
        );
        self.odds("底部里起新一轮主升", BASE_NEW_MARKUP, total, "");
        self.odds(
            "底部守住（守满一年或收复前高）、本轮结束",
            BASE_HELD,
            total,
            "",
        );
        // 刻意不给"最终跌满 50%/70%"的深度频率：它只按已跌幅度分档、不看底部守了多久。
        // 实例：已跌 47%、守了 223 天的底会被报成"81% 再跌满 50%"，而这类底部整体被跌破的
        // 比例只有 45%。底部阶段该看的是上面这组结局频率。

        self.facts.push(format!(
            "谷底 {:.2}（{}）以来 {quiet_days} 个交易日未创新低",
            trough.price,
            self.date(trough.index)
        ));
        self.facts.push(format!(
            "自峰值最深回撤 {:.1}%，现价距峰值 {:+.1}%",
            max_drawdown * 100.0,
            (self.price / peak.price - 1.0) * 100.0
        ));
        self.facts.push(format!(
            "试探性底部已 {} 个交易日，再守 {remaining} 个交易日算本轮结束",
            self.last - base_since + 1
        ));
        format!(
            "下跌后的试探性底部：本轮自 {} 峰值 {:.2} 最深回撤 {:.0}%（谷底 {:.2}），此后 {quiet_days} 个交易日\
             未创新低，按模型进入新的箱体。但历史上这类底部有 {:.0}% 被再次跌破、下跌继续（{BASE_REBROKEN}/{total} 例）\
             ——收盘跌破 {:.2} 即回到下跌通道；收盘站上 {trigger:.2} 才确认新一轮主升。",
            self.date(peak.index),
            peak.price,
            max_drawdown * 100.0,
            trough.price,
            share(BASE_REBROKEN, total),
            trough.price,
        )
    }

    fn markup(&mut self, cycle: CycleCore) -> String {
        let peak = cycle.peak;
        let breakdown = peak.price * (1.0 - self.p.breakdown);
        let stall_left = self.p.stall_days.saturating_sub(self.last - peak.index);
        let mut origin = String::new();
        if let Some(base) = cycle.base {
            let gain = (peak.price / base.price - 1.0) * 100.0;
            let days = self.last - base.index;
            let base_date = self.date(base.index);
            self.facts.push(format!(
                "自起涨点 {:.2}（{base_date}）最高已涨 {gain:.0}%，历时 {days} 个交易日",
                base.price
            ));
            origin = format!(
                "自 {base_date} 起涨点 {:.2} 最高已涨 {gain:.0}%（{days} 个交易日），",
                base.price
            );
            self.level("起涨点", base.price, format!("{base_date} 主升起点"));
        }
        self.level("峰值", peak.price, "再创新高主升延续");
        self.level(
            "下跌确认价",
            breakdown,
            format!(
                "收盘跌破即确认下跌（峰值 −{:.0}%）",
                self.p.breakdown * 100.0
            ),
        );

        let total = MARKUP_TO_TOP + MARKUP_DIRECT_BREAKDOWN;
        self.odds(
            format!(
                "主升不经横盘、直接跌破峰值 −{:.0}%",
                self.p.breakdown * 100.0
            ),
            MARKUP_DIRECT_BREAKDOWN,
            total,
            MARKUP_STABILITY,
        );
        self.facts.push(format!(
            "历史首段主升（起涨 → 第一次横盘）交易日四分位 {:.0} / {:.0} / {:.0}，涨幅中位 {:.0}%",
            FIRST_LEG_DAYS[0],
            FIRST_LEG_DAYS[1],
            FIRST_LEG_DAYS[2],
            FIRST_LEG_GAIN[1] * 100.0
        ));
        self.facts.push(format!(
            "历史整轮主升（含横盘、洗盘后的续涨）交易日中位 {:.0}，涨幅中位 {:.0}%（四分位 {:.0}%～{:.0}%）",
            FULL_RALLY_DAYS[1],
            FULL_RALLY_GAIN[1] * 100.0,
            FULL_RALLY_GAIN[0] * 100.0,
            FULL_RALLY_GAIN[2] * 100.0
        ));
        format!(
            "主升浪：{origin}峰值 {:.2}。按模型，主升之后是高位横盘：再有 {stall_left} 个交易日不创新高即转入横盘；\
             收盘跌破 {breakdown:.2}（峰值 −{:.0}%）则直接确认下跌——历史上 {:.0}% 的主升不经横盘直接跌破。\
             历史首段主升中位 {:.0} 个交易日、涨幅 {:.0}%。",
            peak.price,
            self.p.breakdown * 100.0,
            share(MARKUP_DIRECT_BREAKDOWN, total),
            FIRST_LEG_DAYS[1],
            FIRST_LEG_GAIN[1] * 100.0,
        )
    }

    fn top(&mut self, cycle: CycleCore) -> String {
        let peak = cycle.peak;
        let breakdown = peak.price * (1.0 - self.p.breakdown);
        let stalled_days = self.last - peak.index;
        let from_peak = (self.price / peak.price - 1.0) * 100.0;
        self.level("峰值（突破价）", peak.price, "再创新高即回到主升");
        self.level(
            "下跌确认价",
            breakdown,
            format!(
                "收盘跌破即确认下跌（峰值 −{:.0}%）",
                self.p.breakdown * 100.0
            ),
        );
        if let Some(base) = cycle.base {
            self.level(
                "起涨点",
                base.price,
                format!("{} 主升起点", self.date(base.index)),
            );
        }

        let total = TOP_NEW_HIGH_FIRST + TOP_BREAKDOWN_FIRST;
        self.odds(
            format!("高位横盘先跌破峰值 −{:.0}%", self.p.breakdown * 100.0),
            TOP_BREAKDOWN_FIRST,
            total,
            TOP_STABILITY,
        );
        self.odds(
            "高位横盘先再创新高",
            TOP_NEW_HIGH_FIRST,
            total,
            TOP_STABILITY,
        );
        self.facts.push(format!(
            "峰值 {:.2}（{}）后 {stalled_days} 个交易日未创新高，现价距峰值 {from_peak:+.1}%",
            peak.price,
            self.date(peak.index)
        ));
        self.facts.push(format!(
            "历史典型周期从峰值到确认下跌的交易日四分位 {:.0} / {:.0} / {:.0}",
            PEAK_TO_BREAKDOWN_DAYS[0], PEAK_TO_BREAKDOWN_DAYS[1], PEAK_TO_BREAKDOWN_DAYS[2]
        ));
        format!(
            "高位横盘：峰值 {:.2}（{}）后已 {stalled_days} 个交易日未创新高，现价距峰值 {from_peak:+.1}%。\
             按模型，横盘突破不了就会快速下跌；但历史上高位横盘先跌破峰值 −{:.0}% 的占 {:.0}%，先再创新高的占 {:.0}%\
             （{total} 例），横盘不必然是顶。向上看突破 {:.2}，向下看收盘跌破 {breakdown:.2}——一旦见顶，\
             历史上从峰值到确认下跌中位只要 {:.0} 个交易日。",
            peak.price,
            self.date(peak.index),
            self.p.breakdown * 100.0,
            share(TOP_BREAKDOWN_FIRST, total),
            share(TOP_NEW_HIGH_FIRST, total),
            peak.price,
            PEAK_TO_BREAKDOWN_DAYS[1],
        )
    }

    fn markdown(&mut self, cycle: CycleCore, episode: Episode) -> String {
        let (peak, trough) = (cycle.peak, episode.trough);
        let max_drawdown = 1.0 - trough.price / peak.price;
        let from_peak = (self.price / peak.price - 1.0) * 100.0;
        let since_trough = self.last - trough.index;
        let base_left = self.p.base_days.saturating_sub(since_trough);
        self.level(
            "本轮峰值",
            peak.price,
            format!("{} 的周期顶，回撤从这里量", self.date(peak.index)),
        );
        let quiet = if since_trough == 0 {
            "今日刚创新低".to_string()
        } else {
            format!(
                "{} 以来 {since_trough} 个交易日未创新低",
                self.date(trough.index)
            )
        };
        self.level(
            "谷底",
            trough.price,
            format!("{quiet}；连续 {} 日不破即转入试探性底部", self.p.base_days),
        );
        self.targets(peak, trough);
        let mut reversal = String::new();
        if since_trough < self.p.surge_window {
            let v_left = self.p.surge_window - since_trough;
            let price = trough.price * (1.0 + self.p.surge_gain);
            self.level(
                "V 型反转确认价",
                price,
                format!("{v_left} 个交易日内收盘站上即视为 V 型反转、开启新一轮主升"),
            );
            reversal = format!(
                "若 {v_left} 个交易日内自谷底反弹 {:.0}%（收盘 ≥ {price:.2}），则视为 V 型反转、开启新一轮主升。",
                self.p.surge_gain * 100.0
            );
        }
        self.facts.push(format!(
            "自峰值 {:.2}（{}）最深回撤 {:.1}%，现价距峰值 {from_peak:+.1}%",
            peak.price,
            self.date(peak.index),
            max_drawdown * 100.0
        ));
        self.facts.push(format!(
            "历史典型周期从峰值到最终谷底的交易日四分位 {:.0} / {:.0} / {:.0}，本轮已 {} 个交易日",
            DECLINE_DAYS[0],
            DECLINE_DAYS[1],
            DECLINE_DAYS[2],
            self.last - peak.index
        ));

        let targets = format!(
            "模型预设的下跌目标区为峰值 −50%～−70%（{:.2}～{:.2}），−90%（{:.2}）为小概率",
            peak.price * 0.5,
            peak.price * 0.3,
            peak.price * 0.1
        );
        let evidence = match cycle.kind {
            CycleKind::Markup => match self.depth_odds(max_drawdown) {
                Some(DepthEvidence {
                    given_at,
                    given,
                    reached,
                }) => {
                    let parts: Vec<String> = reached
                        .iter()
                        .map(|&(depth, hits)| {
                            format!(
                                "跌满 {:.0}% 的 {hits} 次（{:.0}%）",
                                depth * 100.0,
                                share(hits, given)
                            )
                        })
                        .collect();
                    format!(
                        "历史实测：已跌 ≥{:.0}% 的 {given} 次回撤中，最终{}。",
                        given_at * 100.0,
                        parts.join("、")
                    )
                }
                None => "本轮回撤已超出历史统计范围，没有可比样本。".to_string(),
            },
            CycleKind::RangeHigh => {
                "这是未经暴涨的非典型下跌，历史深度统计在样本内外差异过大，不给概率。".to_string()
            }
        };
        let kind = match cycle.kind {
            CycleKind::Markup => "",
            CycleKind::RangeHigh => "（非典型：之前没有暴涨，是从一年内高点深跌）",
        };
        format!(
            "下跌通道{kind}：自 {} 峰值 {:.2} 最深已回撤 {:.0}%（谷底 {:.2}），现价距峰值 {from_peak:+.1}%。\
             {targets}。{evidence}中途反弹不改变趋势：再有 {base_left} 个交易日不创新低（不破 {:.2}）才转入试探性底部。{}",
            self.date(peak.index),
            peak.price,
            max_drawdown * 100.0,
            trough.price,
            trough.price,
            reversal,
        )
    }

    /// 模型的下跌目标价：峰值 −50% / −70% / −90%
    fn targets(&mut self, peak: PricePoint, trough: PricePoint) {
        for (depth, meaning) in MODEL_TARGETS {
            let price = peak.price * (1.0 - depth);
            let meaning = if trough.price <= price {
                "本轮已触及"
            } else {
                meaning
            };
            self.level(&format!("峰值 −{:.0}%", depth * 100.0), price, meaning);
        }
    }

    /// 典型回撤事件：已跌到 `max_drawdown` 时，最终跌满各模型目标的历史频率。写进 `odds`，
    /// 同时返回给结论引用；没有可比样本时为 `None`。
    ///
    /// 条件取不超过当前回撤的最近网格点（如已跌 43% 按"已跌 ≥40%"），分子分母都是整数、可核对；
    /// 代价是比插值略低估更深一档的概率。
    fn depth_odds(&mut self, max_drawdown: f64) -> Option<DepthEvidence> {
        let &(given_at, given) = TYPICAL_DRAWDOWN_REACHED
            .iter()
            .rev()
            .find(|(grid, _)| *grid <= max_drawdown + 1e-9)?;
        let mut reached = Vec::new();
        for (depth, _) in MODEL_TARGETS {
            if depth <= max_drawdown || given == 0 {
                continue;
            }
            let Some(&(_, hits)) = TYPICAL_DRAWDOWN_REACHED
                .iter()
                .find(|(grid, _)| (grid - depth).abs() < 1e-9)
            else {
                continue;
            };
            self.odds(
                format!(
                    "已跌 ≥{:.0}% 的回撤最终跌满 {:.0}%",
                    given_at * 100.0,
                    depth * 100.0
                ),
                hits,
                given,
                DEPTH_STABILITY,
            );
            reached.push((depth, hits));
        }
        (!reached.is_empty()).then_some(DepthEvidence {
            given_at,
            given,
            reached,
        })
    }
}

/// 深度条件频率：已跌 ≥`given_at` 的 `given` 次回撤中，各模型目标 `(深度, 命中数)`。
struct DepthEvidence {
    given_at: f64,
    given: usize,
    reached: Vec<(f64, usize)>,
}

fn history(bars: &[AdjustedBar], episodes: &[DrawdownEpisode]) -> Vec<CycleEpisode> {
    episodes
        .iter()
        .rev()
        .take(MAX_HISTORY)
        .map(|ep| CycleEpisode {
            kind: ep.kind,
            base_date: ep.base.map(|b| format_date(bars, b.index)),
            base_price: ep.base.map(|b| b.price),
            peak_date: format_date(bars, ep.peak.index),
            peak_price: ep.peak.price,
            trough_date: format_date(bars, ep.trough.index),
            trough_price: ep.trough.price,
            rally_percent: ep.base.map(|b| (ep.peak.price / b.price - 1.0) * 100.0),
            rally_days: ep.base.map(|b| ep.peak.index - b.index),
            max_drawdown_percent: ep.max_drawdown() * 100.0,
            decline_days: ep.trough.index - ep.peak.index,
            outcome: ep.end.map(|(_, outcome)| outcome),
            outcome_label: outcome_label(ep).to_string(),
        })
        .collect()
}

fn outcome_label(ep: &DrawdownEpisode) -> &'static str {
    match (ep.end.map(|(_, outcome)| outcome), ep.kind) {
        (None, _) => "进行中",
        (Some(EpisodeOutcome::NewHigh), CycleKind::Markup) => "再创新高（洗盘）",
        (Some(EpisodeOutcome::NewHigh), CycleKind::RangeHigh) => "再创新高（下跌被证伪）",
        (Some(EpisodeOutcome::VReversal), _) => "V 型反转，开启新一轮",
        (Some(EpisodeOutcome::NewMarkup), _) => "底部起新一轮主升",
        (Some(EpisodeOutcome::Recovered), _) => "底部抬高收复前高，本轮结束",
        (Some(EpisodeOutcome::BaseHeld), _) => "底部守满一年，本轮结束",
    }
}

fn price_basis(series: &AdjustedSeries) -> String {
    match (series.adjusted, series.corporate_actions) {
        (true, 0) => "前复权（区间内未发现除权除息）".to_string(),
        (true, n) => format!("前复权（以最新收盘为基准，已处理 {n} 次除权除息）"),
        (false, _) => "未复权：涨跌额字段缺失，无法识别除权，送转会被误判成下跌".to_string(),
    }
}

fn method_note(p: &CycleParams) -> String {
    format!(
        "阶段按周期模型逐日推进，只用当日及以前的数据：{} 个交易日内较最低价 +{:.0}% 确认主升，\
         {} 日不创新高为高位横盘，收盘较峰值 −{:.0}% 确认下跌，{} 日不创新低为试探性底部。\
         概率与分位数是本库 131 只股票 2000–2026 年前复权日线上的历史频率（以行业龙头为主），\
         不是对本股的保证。实证显示阶段对未来 5/20/60 日涨跌方向没有样本外预测力，因此不参与点预测。",
        p.surge_window,
        p.surge_gain * 100.0,
        p.stall_days,
        p.breakdown * 100.0,
        p.base_days
    )
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

fn format_date(bars: &[AdjustedBar], index: usize) -> String {
    bars[index].date.format("%Y-%m-%d").to_string()
}

fn lowest_low(bars: &[AdjustedBar], from: usize, to: usize) -> f64 {
    bars[from..=to]
        .iter()
        .map(|bar| bar.low)
        .fold(f64::INFINITY, f64::min)
}

fn highest_high(bars: &[AdjustedBar], from: usize, to: usize) -> f64 {
    bars[from..=to]
        .iter()
        .map(|bar| bar.high)
        .fold(f64::NEG_INFINITY, f64::max)
}
