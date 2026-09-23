//! 周期阶段模型：箱体震荡 → 主升浪 → 高位横盘 → 下跌通道 → 箱体，循环往复。
//!
//! # 判断标准
//!
//! 大多数股票长期在箱体里上下震荡；某天突然疯狂上涨，持续几周到一两个月；随后高位横盘，
//! 突破不了就快速下跌，一波接一波、中途小反弹但趋势向下，从峰值跌去 50%～70%（小概率 90%）；
//! 之后回到新的箱体，周而复始。
//!
//! 落成可计算的规则见 [`machine`]：40 个交易日内较最低价 +50% 确认主升；两周不创新高为高位
//! 横盘；收盘较峰值 −20% 确认下跌；三个月不创新低为试探性底部，底部被跌破则同一轮下跌继续。
//! 判断只用当日及以前的数据；价格先做前复权（[`adjust`]），否则送转会被当成腰斩。
//!
//! # 实证（`examples/cycle_phase_study.rs`，本库 131 只股票 2000–2026）
//!
//! | 模型说法 | 实测 | 结论 |
//! |---|---|---|
//! | 大多数时间在箱体 | 箱体 55%（其中一半是下跌后的试探底），下跌通道 34% | 箱体为主，但下跌占时远超"一成" |
//! | 主升几周到一两个月 | 首段主升中位 37 个交易日，62% 在两个月内 | 成立 |
//! | 横盘突破不了就快速下跌 | 横盘先跌破 −20% 的 42%、先再创新高的 58%；见顶后到 −20% 中位 13 日 | "快"成立，"横盘即顶"不成立 |
//! | 下跌一波接一波 | 下跌后的底部 45% 被再次跌破；峰值到谷底中位 108 日 | 成立 |
//! | 跌 50%～70% 是大概率 | 跌破 −20% 后最终跌满 50% 的 31%、满 70% 的 7%；已跌 40% 再到 50% 的 66% | 不成立，除非已经深跌 |
//! | 跌 90% 是小概率 | 1183 次回撤中 0 次 | 本库（以龙头为主）未出现 |
//!
//! 阶段对未来 5/20/60 日涨跌**方向**没有样本外预测力（BSS≈0），所以本模块只输出阶段、
//! 关键价位与结局频率，**不参与点预测**。各概率的样本外稳定性见 [`outlook`]。

pub mod adjust;
pub mod machine;
mod outlook;

use serde::{Deserialize, Serialize};

use crate::db::models::HistoricalData;

pub use adjust::{forward_adjust, AdjustedBar, AdjustedSeries};
pub use machine::{
    track_phases, CycleCore, CycleKind, CycleParams, CyclePhase, CycleState, DrawdownEpisode,
    Episode, EpisodeOutcome, PhaseTrack, PricePoint,
};

/// 周期分析所需的最少 K 线：非典型下跌要回看 250 日高点，更短的历史只看得出主升。
pub const MIN_CYCLE_BARS: usize = 250;

/// 周期阶段分析：当前阶段、按模型推演的关键价位与全库历史结局频率。价格均为前复权口径。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleAnalysis {
    pub phase: CyclePhase,
    pub phase_label: String,
    /// 本阶段起始日
    pub phase_since: String,
    /// 本阶段已持续的交易日数（含起始日）
    pub days_in_phase: usize,
    /// 所处周期的来历；纯箱体（无周期上下文）时为 `None`
    pub cycle_kind: Option<CycleKind>,
    /// 是否为下跌后的试探性底部（对外阶段仍是箱体，但保留本轮峰值与谷底）
    pub tentative_base: bool,
    /// 按模型推理出的结论
    pub summary: String,
    pub key_levels: Vec<CycleLevel>,
    /// 全库历史结局频率——不是对本股的预测
    pub odds: Vec<CycleOdds>,
    /// 支撑结论的可核验事实
    pub facts: Vec<String>,
    /// 本股历史回撤事件，最近的在前
    pub history: Vec<CycleEpisode>,
    /// 价格口径（前复权 / 未复权）
    pub price_basis: String,
    /// 规则、样本与局限
    pub method_note: String,
}

impl CycleAnalysis {
    /// 附到每条预测 `key_factors` 里的一行摘要
    pub fn key_factor(&self) -> String {
        let label = if self.tentative_base {
            "箱体震荡（下跌后的试探性底部）"
        } else {
            self.phase_label.as_str()
        };
        format!(
            "周期阶段: {label}，{} 起第 {} 个交易日",
            self.phase_since, self.days_in_phase
        )
    }
}

/// 按模型推演出的关键价位。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleLevel {
    pub label: String,
    pub price: f64,
    /// 相对最新收盘的距离（%）
    pub distance_percent: f64,
    /// 价格到达后按模型意味着什么
    pub meaning: String,
}

/// 一条历史结局频率：`hits / samples`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleOdds {
    pub label: String,
    pub probability: f64,
    pub hits: usize,
    pub samples: usize,
    /// 口径与样本外稳定性
    pub note: String,
}

/// 本股的一次历史回撤事件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleEpisode {
    pub kind: CycleKind,
    pub base_date: Option<String>,
    pub base_price: Option<f64>,
    pub peak_date: String,
    pub peak_price: f64,
    pub trough_date: String,
    pub trough_price: f64,
    /// 起涨点 → 峰值涨幅（%）
    pub rally_percent: Option<f64>,
    /// 起涨点 → 峰值交易日数
    pub rally_days: Option<usize>,
    /// 峰值 → 谷底最大回撤（%，负数）
    pub max_drawdown_percent: f64,
    /// 峰值 → 谷底交易日数
    pub decline_days: usize,
    /// `None` 表示仍在进行
    pub outcome: Option<EpisodeOutcome>,
    pub outcome_label: String,
}

/// 对时间正序的日线做周期阶段分析。有效 K 线不足 [`MIN_CYCLE_BARS`] 时返回 `None`。
pub fn analyze_cycle(history: &[HistoricalData]) -> Option<CycleAnalysis> {
    let series = forward_adjust(history);
    if series.bars.len() < MIN_CYCLE_BARS {
        return None;
    }
    let params = CycleParams::default();
    let track = track_phases(&series.bars, &params);
    outlook::build(&series, &track, &params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    /// 分段对数线性收盘价，`change` 按真实涨跌填，模拟不复权接口的正常日子
    fn history(start: f64, segments: &[(usize, f64)]) -> Vec<HistoricalData> {
        let mut closes = vec![start];
        for &(len, to) in segments {
            let from = *closes.last().unwrap();
            for i in 1..=len {
                let frac = i as f64 / len as f64;
                closes.push((from.ln() + (to.ln() - from.ln()) * frac).exp());
            }
        }
        let day0 = NaiveDate::from_ymd_opt(2018, 1, 1).unwrap();
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| {
                let prev = if i == 0 { close } else { closes[i - 1] };
                HistoricalData {
                    symbol: "600000".to_string(),
                    date: day0 + chrono::Duration::days(i as i64),
                    open: close,
                    close,
                    high: close * 1.005,
                    low: close * 0.995,
                    volume: 1000,
                    amount: 0.0,
                    amplitude: 0.0,
                    turnover_rate: 0.0,
                    volume_ratio: 0.0,
                    change_percent: (close / prev - 1.0) * 100.0,
                    change: close - prev,
                }
            })
            .collect()
    }

    fn level<'a>(analysis: &'a CycleAnalysis, label: &str) -> &'a CycleLevel {
        analysis
            .key_levels
            .iter()
            .find(|l| l.label == label)
            .unwrap_or_else(|| panic!("缺少关键价位「{label}」: {:?}", analysis.key_levels))
    }

    fn assert_well_formed(analysis: &CycleAnalysis) {
        assert!(!analysis.summary.is_empty());
        assert!(!analysis.method_note.is_empty());
        for l in &analysis.key_levels {
            assert!(l.price.is_finite() && l.price > 0.0, "{l:?}");
            assert!(l.distance_percent.is_finite(), "{l:?}");
        }
        for o in &analysis.odds {
            assert!(o.samples > 0 && o.hits <= o.samples, "{o:?}");
            assert!((o.probability - o.hits as f64 / o.samples as f64).abs() < 1e-12);
        }
        assert!(!analysis.summary.contains("NaN") && !analysis.summary.contains("inf"));
    }

    #[test]
    fn plain_box_reports_range_and_triggers() {
        let analysis = analyze_cycle(&history(10.0, &[(150, 11.0), (150, 10.0)])).unwrap();
        assert_well_formed(&analysis);
        assert_eq!(analysis.phase, CyclePhase::Box);
        assert!(analysis.cycle_kind.is_none() && !analysis.tentative_base);
        let trigger = level(&analysis, "主升确认价");
        assert!(
            trigger.distance_percent > 30.0,
            "箱体里主升确认价应远在现价之上: {trigger:?}"
        );
        assert!(level(&analysis, "箱体上沿").price > level(&analysis, "箱体下沿").price);
        assert!(analysis.odds.is_empty(), "纯箱体没有经过校验的概率可给");
        assert_eq!(analysis.days_in_phase, 301);
    }

    #[test]
    fn top_phase_reports_breakout_and_breakdown_levels() {
        // 箱体 → 30 日翻倍 → 横 15 日
        let analysis =
            analyze_cycle(&history(10.0, &[(300, 10.0), (30, 20.0), (15, 19.0)])).unwrap();
        assert_well_formed(&analysis);
        assert_eq!(analysis.phase, CyclePhase::Top);
        assert_eq!(analysis.cycle_kind, Some(CycleKind::Markup));
        let peak = level(&analysis, "峰值（突破价）").price;
        let breakdown = level(&analysis, "下跌确认价").price;
        assert!((breakdown / peak - 0.8).abs() < 1e-9);
        let total: f64 = analysis.odds.iter().map(|o| o.probability).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "横盘两种结局应互补: {:?}",
            analysis.odds
        );
        assert!(analysis.key_factor().contains("高位横盘"));
    }

    #[test]
    fn markdown_reports_model_targets_with_conditional_odds() {
        // 翻倍后跌去约 45%
        let analysis =
            analyze_cycle(&history(10.0, &[(300, 10.0), (30, 20.0), (40, 11.0)])).unwrap();
        assert_well_formed(&analysis);
        assert_eq!(analysis.phase, CyclePhase::Markdown);
        let peak = level(&analysis, "本轮峰值").price;
        for (label, factor) in [("峰值 −50%", 0.5), ("峰值 −70%", 0.3), ("峰值 −90%", 0.1)]
        {
            assert!((level(&analysis, label).price / peak - factor).abs() < 1e-9);
        }
        // 已跌 ~45%：条件取「已跌 ≥45%」这一档，只报尚未到达的目标
        let labels: Vec<&str> = analysis.odds.iter().map(|o| o.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "已跌 ≥45% 的回撤最终跌满 50%",
                "已跌 ≥45% 的回撤最终跌满 70%",
                "已跌 ≥45% 的回撤最终跌满 90%"
            ]
        );
        assert_eq!(
            (analysis.odds[0].hits, analysis.odds[0].samples),
            (364, 447)
        );
        assert_eq!(analysis.odds[2].hits, 0);
        assert!(
            analysis.odds[2].note.contains("从未出现"),
            "0 例必须说明口径"
        );
        // 谷底后不久：还在 V 型反转窗口内
        assert!(analysis
            .key_levels
            .iter()
            .any(|l| l.label == "V 型反转确认价"));
        assert_eq!(analysis.history.len(), 1);
        assert_eq!(analysis.history[0].outcome_label, "进行中");
    }

    #[test]
    fn tentative_base_keeps_the_cycle_context() {
        let analysis = analyze_cycle(&history(
            10.0,
            &[(300, 10.0), (30, 20.0), (60, 8.0), (80, 8.3)],
        ))
        .unwrap();
        assert_well_formed(&analysis);
        assert_eq!(analysis.phase, CyclePhase::Box);
        assert!(analysis.tentative_base);
        assert_eq!(analysis.cycle_kind, Some(CycleKind::Markup));
        assert!(analysis.key_factor().contains("试探性底部"));
        assert!(level(&analysis, "谷底（破位价）").distance_percent < 0.0);
        assert!(level(&analysis, "峰值 −50%").meaning.contains("已触及"));
        assert!(analysis.odds.iter().any(|o| o.label.contains("再次跌破")));
        assert!(
            analysis.odds.iter().all(|o| !o.label.contains("最终跌满")),
            "深度频率不看底部守了多久，放在底部阶段会误导: {:?}",
            analysis.odds
        );
    }

    #[test]
    fn markup_phase_reports_rally_origin() {
        let analysis = analyze_cycle(&history(10.0, &[(300, 10.0), (25, 17.0)])).unwrap();
        assert_well_formed(&analysis);
        assert_eq!(analysis.phase, CyclePhase::Markup);
        assert!(level(&analysis, "起涨点").price < 10.0);
        assert!(analysis.summary.contains("主升浪"));
    }

    #[test]
    fn bonus_share_split_is_not_mistaken_for_a_crash() {
        // 平稳箱体中间做一次 10 送 10：不复权的收盘价腰斩，但真实涨跌为 0
        let mut bars = history(10.0, &[(400, 10.0)]);
        for bar in bars.iter_mut().skip(200) {
            bar.close /= 2.0;
            bar.open /= 2.0;
            bar.high /= 2.0;
            bar.low /= 2.0;
            bar.change /= 2.0;
        }
        let analysis = analyze_cycle(&bars).unwrap();
        assert_eq!(analysis.phase, CyclePhase::Box, "{}", analysis.summary);
        assert!(
            analysis.history.is_empty(),
            "送转不能被当成一次回撤: {:?}",
            analysis.history
        );
        assert!(
            analysis.price_basis.contains("1 次除权除息"),
            "{}",
            analysis.price_basis
        );
    }

    #[test]
    fn short_history_yields_nothing() {
        assert!(analyze_cycle(&history(10.0, &[(100, 12.0)])).is_none());
    }
}
