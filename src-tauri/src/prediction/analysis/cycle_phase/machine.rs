//! 周期状态机：逐根 K 线推进，只读当根及以前的数据（无未来函数）。
//!
//! # 转移规则
//!
//! - 箱体 → 主升：收盘 ≥ 近 `surge_window` 日最低价 × (1 + `surge_gain`)。
//! - 主升 → 高位横盘：连续 `stall_days` 日未创峰值新高。
//! - 高位横盘 → 主升：重新创出峰值新高（横盘向上突破）。
//! - 主升 / 高位横盘 → 下跌通道：收盘 ≤ 峰值 × (1 − `breakdown`)。
//! - 下跌通道 → 主升（同一轮）：重新创出峰值新高——那次回撤只是洗盘。
//! - 下跌通道 → 主升（新一轮）：谷底后 `surge_window` 日内收盘较谷底涨 `surge_gain`（V 型反转）。
//! - 下跌通道 → 试探性底部：连续 `base_days` 日不创新低。对外仍报"箱体"，但保留本轮峰值与谷底。
//! - 试探性底部 → 下跌通道：收盘跌破谷底。底部失败，**同一次回撤继续**，回撤仍从原峰值量。
//! - 试探性底部里收复本轮峰值：本轮周期结束，回到无上下文的箱体。收复得快会先满足主升条件
//!   （算新一轮），走到这一步说明是箱体慢慢抬高，不是模型说的暴涨。
//! - 试探性底部守住 `base_expiry_days` 日：本轮周期结束，回到无上下文的箱体。
//! - 箱体（无上下文）→ 下跌通道（非典型）：收盘较近 `range_window` 日最高价回撤
//!   ≥ `range_breakdown`，且创 `new_low_window` 日收盘新低。模型里下跌都跟在主升之后，但慢牛
//!   见顶、箱体破位同样会深跌；把它们留在"箱体"是错标——实测旧口径下被标成箱体的交易日里，
//!   26% 已较一年内高点跌去 30% 以上。
//!
//! # 回撤事件
//!
//! 进入下跌通道开始一个回撤事件，直到出现峰值新高（洗盘）、V 型反转、试探性底部里再起主升、
//! 底部收复前高、或底部守满期限为止。**洗盘也是一个完整的回撤事件**：站在 −25% 的位置分不清
//! 是洗盘还是大跌的开始，统计"最终会跌多深"时把洗盘排除在分母外，会把深跌概率高估约一半。

use serde::{Deserialize, Serialize};

use super::adjust::AdjustedBar;

/// 周期阶段（对外口径）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CyclePhase {
    /// 箱体震荡（含下跌后的试探性底部）
    Box,
    /// 主升浪
    Markup,
    /// 高位横盘
    Top,
    /// 下跌通道
    Markdown,
}

impl CyclePhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Box => "箱体震荡",
            Self::Markup => "主升浪",
            Self::Top => "高位横盘",
            Self::Markdown => "下跌通道",
        }
    }
}

/// 一轮周期的来历。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CycleKind {
    /// 箱体 → 主升 → 见顶：模型的典型形态
    Markup,
    /// 没有合格主升，从一年内高点深跌（慢牛见顶、箱体破位）
    RangeHigh,
}

/// 回撤事件的结局。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    /// 重新创出峰值新高：典型周期里是洗盘，主升继续；非典型下跌则被证伪
    NewHigh,
    /// 谷底后短期内反弹达到主升标准，开启新一轮
    VReversal,
    /// 试探性底部里出现新一轮主升
    NewMarkup,
    /// 试探性底部慢慢抬高、收复本轮峰值，本轮周期结束
    Recovered,
    /// 试探性底部守满期限，本轮周期结束
    BaseHeld,
}

/// 状态机参数。
///
/// 默认值逐条来自周期模型的文字描述，**不是在本库数据上寻优的结果**：
/// "疯狂上涨几周甚至 1-2 个月"→ 40 个交易日内 +50%；"高位横盘一段时间"→ 两周不创新高；
/// "快速下跌"的确认用通行的 −20% 熊市口径；"又开始在箱体中上下"→ 三个月不创新低。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycleParams {
    /// 主升确认的回看窗口（交易日）
    pub surge_window: usize,
    /// 主升确认涨幅（0.5 = 50%）
    pub surge_gain: f64,
    /// 连续多少日不创新高算横盘
    pub stall_days: usize,
    /// 收盘较峰值回撤多少确认下跌
    pub breakdown: f64,
    /// 下跌中连续多少日不创新低算试探性底部
    pub base_days: usize,
    /// 试探性底部守住多少日算本轮结束
    pub base_expiry_days: usize,
    /// 非典型下跌：参照高点的回看窗口
    pub range_window: usize,
    /// 非典型下跌：较参照高点回撤多少
    pub range_breakdown: f64,
    /// 非典型下跌：须创多少日收盘新低
    pub new_low_window: usize,
}

impl Default for CycleParams {
    fn default() -> Self {
        Self {
            surge_window: 40,
            surge_gain: 0.50,
            stall_days: 10,
            breakdown: 0.20,
            base_days: 60,
            base_expiry_days: 250,
            range_window: 250,
            range_breakdown: 0.30,
            new_low_window: 60,
        }
    }
}

/// 序列中的一个价格点。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PricePoint {
    pub index: usize,
    pub price: f64,
}

/// 一轮周期的骨架：起涨点与峰值。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycleCore {
    pub kind: CycleKind,
    /// 起涨点（主升前的最低价）；非典型下跌为 `None`
    pub base: Option<PricePoint>,
    pub peak: PricePoint,
}

/// 进行中的回撤事件。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Episode {
    /// 进入下跌通道的那根 K 线
    pub start: usize,
    /// 峰值以来的最低价
    pub trough: PricePoint,
}

/// 状态机在某根 K 线收盘后的完整状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CycleState {
    /// 箱体震荡，无周期上下文
    Box,
    /// 主升（`stalled == false`）或高位横盘（`stalled == true`）
    Rally { cycle: CycleCore, stalled: bool },
    /// 下跌通道
    Decline { cycle: CycleCore, episode: Episode },
    /// 下跌后的试探性底部（对外是"箱体"）
    Base {
        cycle: CycleCore,
        episode: Episode,
        since: usize,
    },
}

impl CycleState {
    pub fn phase(&self) -> CyclePhase {
        match self {
            Self::Box | Self::Base { .. } => CyclePhase::Box,
            Self::Rally { stalled: false, .. } => CyclePhase::Markup,
            Self::Rally { stalled: true, .. } => CyclePhase::Top,
            Self::Decline { .. } => CyclePhase::Markdown,
        }
    }
}

/// 一次回撤事件（已结束或截至最后一根仍在进行）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawdownEpisode {
    pub kind: CycleKind,
    pub base: Option<PricePoint>,
    pub peak: PricePoint,
    pub start: usize,
    pub trough: PricePoint,
    /// `None` 表示截至最后一根仍未结束
    pub end: Option<(usize, EpisodeOutcome)>,
}

impl DrawdownEpisode {
    /// 峰值到谷底的最大回撤（负数，如 −0.45）
    pub fn max_drawdown(&self) -> f64 {
        self.trough.price / self.peak.price - 1.0
    }
}

/// 整段序列的阶段轨迹。
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseTrack {
    /// 每根 K 线收盘后的状态
    pub states: Vec<CycleState>,
    /// 按时间顺序的回撤事件；末尾可能有一个未结束的
    pub episodes: Vec<DrawdownEpisode>,
}

impl PhaseTrack {
    pub fn phase_at(&self, index: usize) -> Option<CyclePhase> {
        self.states.get(index).map(CycleState::phase)
    }

    /// 最后一根所处阶段的起点下标。
    pub fn current_phase_start(&self) -> Option<usize> {
        let last = self.states.last()?.phase();
        let run = self
            .states
            .iter()
            .rev()
            .take_while(|state| state.phase() == last)
            .count();
        Some(self.states.len() - run)
    }
}

/// 从头推进状态机。`bars` 必须时间正序且价格为正（[`super::adjust::forward_adjust`] 的输出即可）。
pub fn track_phases(bars: &[AdjustedBar], params: &CycleParams) -> PhaseTrack {
    let mut machine = Machine {
        bars,
        params: *params,
        episodes: Vec::new(),
    };
    let mut state = CycleState::Box;
    let mut states = Vec::with_capacity(bars.len());
    for t in 0..bars.len() {
        state = machine.step(state, t);
        states.push(state);
    }
    if let CycleState::Decline { cycle, episode } | CycleState::Base { cycle, episode, .. } = state
    {
        machine.episodes.push(episode_record(cycle, episode, None));
    }
    PhaseTrack {
        states,
        episodes: machine.episodes,
    }
}

struct Machine<'a> {
    bars: &'a [AdjustedBar],
    params: CycleParams,
    episodes: Vec<DrawdownEpisode>,
}

impl Machine<'_> {
    fn step(&mut self, state: CycleState, t: usize) -> CycleState {
        let bar = self.bars[t];
        let p = self.params;
        match state {
            CycleState::Box => {
                if let Some(cycle) = self.surge(t) {
                    return self.rally(cycle, t);
                }
                self.range_breakdown(t).unwrap_or(CycleState::Box)
            }
            CycleState::Rally { mut cycle, .. } => {
                if bar.high > cycle.peak.price {
                    cycle.peak = PricePoint {
                        index: t,
                        price: bar.high,
                    };
                }
                if bar.close <= cycle.peak.price * (1.0 - p.breakdown) {
                    let episode = Episode {
                        start: t,
                        trough: self.lowest_low(cycle.peak.index, t),
                    };
                    return CycleState::Decline { cycle, episode };
                }
                self.rally(cycle, t)
            }
            CycleState::Decline {
                mut cycle,
                mut episode,
            } => {
                if bar.low < episode.trough.price {
                    episode.trough = PricePoint {
                        index: t,
                        price: bar.low,
                    };
                }
                if bar.high > cycle.peak.price {
                    self.close(cycle, episode, t, EpisodeOutcome::NewHigh);
                    return match cycle.kind {
                        CycleKind::Markup => {
                            cycle.peak = PricePoint {
                                index: t,
                                price: bar.high,
                            };
                            self.rally(cycle, t)
                        }
                        CycleKind::RangeHigh => CycleState::Box,
                    };
                }
                let trough = episode.trough;
                if t - trough.index <= p.surge_window
                    && bar.close >= trough.price * (1.0 + p.surge_gain)
                {
                    self.close(cycle, episode, t, EpisodeOutcome::VReversal);
                    let next = CycleCore {
                        kind: CycleKind::Markup,
                        base: Some(trough),
                        peak: self.highest_high(trough.index, t),
                    };
                    return self.rally(next, t);
                }
                if t - trough.index >= p.base_days {
                    return CycleState::Base {
                        cycle,
                        episode,
                        since: t,
                    };
                }
                CycleState::Decline { cycle, episode }
            }
            CycleState::Base {
                cycle,
                mut episode,
                since,
            } => {
                if let Some(next) = self.surge(t) {
                    self.close(cycle, episode, t, EpisodeOutcome::NewMarkup);
                    return self.rally(next, t);
                }
                // 先比较再更新谷底：跌破的是"此前"的底
                let broken = bar.close < episode.trough.price;
                if bar.low < episode.trough.price {
                    episode.trough = PricePoint {
                        index: t,
                        price: bar.low,
                    };
                }
                if broken {
                    return CycleState::Decline { cycle, episode };
                }
                if bar.high > cycle.peak.price {
                    self.close(cycle, episode, t, EpisodeOutcome::Recovered);
                    return CycleState::Box;
                }
                if t - since >= p.base_expiry_days {
                    self.close(cycle, episode, t, EpisodeOutcome::BaseHeld);
                    return CycleState::Box;
                }
                CycleState::Base {
                    cycle,
                    episode,
                    since,
                }
            }
        }
    }

    fn rally(&self, cycle: CycleCore, t: usize) -> CycleState {
        CycleState::Rally {
            cycle,
            stalled: t - cycle.peak.index >= self.params.stall_days,
        }
    }

    /// 主升确认：收盘较近 `surge_window` 日最低价涨够 `surge_gain`。
    fn surge(&self, t: usize) -> Option<CycleCore> {
        let p = self.params;
        if t < p.surge_window {
            return None;
        }
        let base = self.lowest_low(t - p.surge_window, t);
        (self.bars[t].close >= base.price * (1.0 + p.surge_gain)).then(|| CycleCore {
            kind: CycleKind::Markup,
            base: Some(base),
            peak: self.highest_high(base.index, t),
        })
    }

    /// 非典型下跌：较一年内高点深跌且创收盘新低。
    fn range_breakdown(&self, t: usize) -> Option<CycleState> {
        let p = self.params;
        if t < p.range_window.max(p.new_low_window) {
            return None;
        }
        let close = self.bars[t].close;
        let peak = self.highest_high(t - p.range_window, t);
        if close > peak.price * (1.0 - p.range_breakdown) {
            return None;
        }
        let prior_low_close = self.bars[t - p.new_low_window..t]
            .iter()
            .map(|bar| bar.close)
            .fold(f64::INFINITY, f64::min);
        (close < prior_low_close).then(|| CycleState::Decline {
            cycle: CycleCore {
                kind: CycleKind::RangeHigh,
                base: None,
                peak,
            },
            episode: Episode {
                start: t,
                trough: self.lowest_low(peak.index, t),
            },
        })
    }

    fn close(&mut self, cycle: CycleCore, episode: Episode, t: usize, outcome: EpisodeOutcome) {
        self.episodes
            .push(episode_record(cycle, episode, Some((t, outcome))));
    }

    /// `[from, to]` 内最低价，并列取最早一根（与逐根更新时的严格小于一致）。
    fn lowest_low(&self, from: usize, to: usize) -> PricePoint {
        let mut best = PricePoint {
            index: from,
            price: self.bars[from].low,
        };
        for (index, bar) in self.bars.iter().enumerate().take(to + 1).skip(from + 1) {
            if bar.low < best.price {
                best = PricePoint {
                    index,
                    price: bar.low,
                };
            }
        }
        best
    }

    /// `[from, to]` 内最高价，并列取最早一根。
    fn highest_high(&self, from: usize, to: usize) -> PricePoint {
        let mut best = PricePoint {
            index: from,
            price: self.bars[from].high,
        };
        for (index, bar) in self.bars.iter().enumerate().take(to + 1).skip(from + 1) {
            if bar.high > best.price {
                best = PricePoint {
                    index,
                    price: bar.high,
                };
            }
        }
        best
    }
}

fn episode_record(
    cycle: CycleCore,
    episode: Episode,
    end: Option<(usize, EpisodeOutcome)>,
) -> DrawdownEpisode {
    DrawdownEpisode {
        kind: cycle.kind,
        base: cycle.base,
        peak: cycle.peak,
        start: episode.start,
        trough: episode.trough,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    /// 分段对数线性路径，叠加 ±`wiggle` 的正弦扰动避免整段等高。
    fn path(start: f64, segments: &[(usize, f64)], wiggle: f64) -> Vec<f64> {
        let mut out = Vec::new();
        let mut from = start;
        for &(len, to) in segments {
            for i in 1..=len {
                let frac = i as f64 / len as f64;
                out.push((from.ln() + (to.ln() - from.ln()) * frac).exp());
            }
            from = to;
        }
        out.iter()
            .enumerate()
            .map(|(i, c)| c * (1.0 + wiggle * (i as f64 * 0.7).sin()))
            .collect()
    }

    fn bars(closes: &[f64]) -> Vec<AdjustedBar> {
        let day0 = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| AdjustedBar {
                date: day0 + chrono::Duration::days(i as i64),
                open: close,
                high: close * 1.005,
                low: close * 0.995,
                close,
            })
            .collect()
    }

    fn first_index_of(track: &PhaseTrack, phase: CyclePhase) -> Option<usize> {
        track.states.iter().position(|s| s.phase() == phase)
    }

    /// 箱体 300 日 → 30 日翻倍 → 横盘 25 日 → 100 日跌到 6 → 底部 150 日。
    /// 底部略微抬升，免得扰动项在谷底附近反复刷出新低、让底部迟迟确认不了。
    fn full_cycle() -> Vec<f64> {
        path(
            10.0,
            &[(300, 10.0), (30, 20.0), (25, 19.5), (100, 6.0), (150, 6.3)],
            0.02,
        )
    }

    #[test]
    fn walks_through_the_whole_cycle_in_order() {
        let closes = full_cycle();
        let track = track_phases(&bars(&closes), &CycleParams::default());
        let markup = first_index_of(&track, CyclePhase::Markup).expect("应识别出主升");
        let top = first_index_of(&track, CyclePhase::Top).expect("应识别出高位横盘");
        let markdown = first_index_of(&track, CyclePhase::Markdown).expect("应识别出下跌");
        assert!(track.states[..markup].iter().all(|s| *s == CycleState::Box));
        assert!(
            (300..330).contains(&markup),
            "主升应在拉升段内确认: {markup}"
        );
        assert!(markup < top && top < markdown, "{markup} {top} {markdown}");
        // 最后停在下跌后的试探性底部（对外报箱体），且保留本轮峰值
        let last = *track.states.last().unwrap();
        assert!(matches!(last, CycleState::Base { .. }), "{last:?}");
        assert_eq!(last.phase(), CyclePhase::Box);

        // 回撤事件：一个、未结束、典型周期、回撤约 −70%
        assert_eq!(track.episodes.len(), 1);
        let ep = track.episodes[0];
        assert_eq!(ep.kind, CycleKind::Markup);
        assert!(ep.end.is_none(), "底部还没守满 250 日: {:?}", ep.end);
        assert!(
            (-0.73..-0.66).contains(&ep.max_drawdown()),
            "{}",
            ep.max_drawdown()
        );
        let base = ep.base.expect("典型周期有起涨点");
        assert!(base.index < 330 && base.price < 10.0);
    }

    #[test]
    fn labels_never_look_ahead() {
        // 前缀上跑出的阶段必须与全量跑出的同一位置完全相同——这是"无未来函数"的定义
        let all = bars(&full_cycle());
        let full = track_phases(&all, &CycleParams::default());
        for k in [1, 39, 40, 250, 301, 320, 340, 360, 420, 500, 560, all.len()] {
            let prefix = track_phases(&all[..k], &CycleParams::default());
            assert_eq!(
                prefix.states[..],
                full.states[..k],
                "前缀长度 {k} 处出现分歧"
            );
        }
    }

    #[test]
    fn dip_that_recovers_to_a_new_high_is_a_shakeout() {
        // 箱体 → 翻倍 → 回撤 25% → 再创新高
        let closes = path(
            10.0,
            &[(300, 10.0), (30, 20.0), (15, 15.0), (20, 24.0)],
            0.0,
        );
        let track = track_phases(&bars(&closes), &CycleParams::default());
        assert_eq!(track.episodes.len(), 1);
        let ep = track.episodes[0];
        assert_eq!(ep.end.map(|(_, o)| o), Some(EpisodeOutcome::NewHigh));
        assert!(
            (-0.27..-0.2).contains(&ep.max_drawdown()),
            "{}",
            ep.max_drawdown()
        );
        // 洗盘后仍是同一轮：起涨点不变
        match *track.states.last().unwrap() {
            CycleState::Rally { cycle, .. } => assert_eq!(cycle.base, ep.base),
            other => panic!("洗盘后应回到主升: {other:?}"),
        }
    }

    #[test]
    fn failed_base_resumes_the_same_decline() {
        // 下跌到 8 → 横 80 日（进入试探性底部）→ 再跌到 5
        let closes = path(
            10.0,
            &[(300, 10.0), (30, 20.0), (60, 8.0), (80, 8.2), (60, 5.0)],
            0.0,
        );
        let track = track_phases(&bars(&closes), &CycleParams::default());
        let phases: Vec<CyclePhase> = track.states.iter().map(CycleState::phase).collect();
        let base_at = track
            .states
            .iter()
            .position(|s| matches!(s, CycleState::Base { .. }))
            .expect("应先进入试探性底部");
        assert!(
            phases[base_at..].contains(&CyclePhase::Markdown),
            "底部跌破后应回到下跌"
        );
        // 同一次回撤：只有一个事件，且谷底取到第二段的低点，回撤从原峰值量
        assert_eq!(track.episodes.len(), 1);
        let ep = track.episodes[0];
        assert!(ep.trough.index > base_at);
        assert!(
            (-0.77..-0.73).contains(&ep.max_drawdown()),
            "{}",
            ep.max_drawdown()
        );
    }

    #[test]
    fn sharp_rebound_from_the_trough_starts_a_new_cycle() {
        // 翻倍 → 跌到 10 → 20 日内反弹到 16（+60%）
        let closes = path(
            10.0,
            &[(300, 10.0), (30, 20.0), (40, 10.0), (20, 16.0)],
            0.0,
        );
        let track = track_phases(&bars(&closes), &CycleParams::default());
        let closed: Vec<&DrawdownEpisode> =
            track.episodes.iter().filter(|e| e.end.is_some()).collect();
        assert_eq!(closed.len(), 1);
        assert_eq!(
            closed[0].end.map(|(_, o)| o),
            Some(EpisodeOutcome::VReversal)
        );
        match *track.states.last().unwrap() {
            CycleState::Rally { cycle, .. } => {
                assert_eq!(
                    cycle.base,
                    Some(closed[0].trough),
                    "新一轮起涨点就是上一轮谷底"
                );
            }
            other => panic!("V 型反转后应是主升: {other:?}"),
        }
    }

    #[test]
    fn deep_decline_without_a_surge_is_an_atypical_markdown() {
        // 慢牛一年 +40%（从未 40 日 +50%）→ 半年跌 45%
        let closes = path(10.0, &[(300, 10.0), (250, 14.0), (120, 7.7)], 0.01);
        let track = track_phases(&bars(&closes), &CycleParams::default());
        assert!(
            first_index_of(&track, CyclePhase::Markup).is_none(),
            "慢牛不是主升"
        );
        match *track.states.last().unwrap() {
            CycleState::Decline { cycle, .. } => {
                assert_eq!(cycle.kind, CycleKind::RangeHigh);
                assert!(cycle.base.is_none());
            }
            other => panic!("深跌应标为下跌通道: {other:?}"),
        }
    }

    #[test]
    fn base_that_holds_long_enough_closes_the_cycle() {
        let closes = path(
            10.0,
            &[(300, 10.0), (30, 20.0), (60, 8.0), (330, 8.3)],
            0.01,
        );
        let track = track_phases(&bars(&closes), &CycleParams::default());
        assert_eq!(
            *track.states.last().unwrap(),
            CycleState::Box,
            "底部守满后上下文应清空"
        );
        assert_eq!(track.episodes.len(), 1);
        assert_eq!(
            track.episodes[0].end.map(|(_, o)| o),
            Some(EpisodeOutcome::BaseHeld)
        );
    }

    #[test]
    fn base_that_climbs_back_above_the_peak_ends_the_cycle() {
        // 翻倍 → 跌到 12 → 用 200 日慢慢爬回 21：从未 40 日 +50%，不算新一轮主升
        let closes = path(
            10.0,
            &[(300, 10.0), (30, 20.0), (40, 12.0), (200, 21.0)],
            0.0,
        );
        let track = track_phases(&bars(&closes), &CycleParams::default());
        assert!(
            track
                .states
                .iter()
                .any(|s| matches!(s, CycleState::Base { .. })),
            "应先进入试探性底部"
        );
        assert_eq!(track.episodes.len(), 1);
        let ep = track.episodes[0];
        assert_eq!(ep.end.map(|(_, o)| o), Some(EpisodeOutcome::Recovered));
        assert!(
            (-0.42..-0.38).contains(&ep.max_drawdown()),
            "{}",
            ep.max_drawdown()
        );
        assert_eq!(
            *track.states.last().unwrap(),
            CycleState::Box,
            "收复前高后不应再挂着旧峰值与谷底"
        );
    }

    #[test]
    fn short_or_flat_history_stays_in_the_box() {
        let flat = bars(&path(10.0, &[(600, 10.0)], 0.03));
        let track = track_phases(&flat, &CycleParams::default());
        assert!(track.states.iter().all(|s| *s == CycleState::Box));
        assert!(track.episodes.is_empty());

        let short = bars(&path(10.0, &[(20, 30.0)], 0.0));
        let track = track_phases(&short, &CycleParams::default());
        assert!(
            track.states.iter().all(|s| *s == CycleState::Box),
            "不足回看窗口不能确认主升"
        );
        assert_eq!(track_phases(&[], &CycleParams::default()).states.len(), 0);
    }

    #[test]
    fn current_phase_start_finds_the_last_run() {
        let closes = full_cycle();
        let track = track_phases(&bars(&closes), &CycleParams::default());
        let start = track.current_phase_start().unwrap();
        assert!(track.states[start..]
            .iter()
            .all(|s| s.phase() == CyclePhase::Box));
        assert_ne!(track.states[start - 1].phase(), CyclePhase::Box);
    }
}
