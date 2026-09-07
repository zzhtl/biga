//! 纪律阈值配置。
//!
//! # 这些数字从哪来
//!
//! 全部是行业约定俗成的经验值，**不是从本项目数据优化出来的**。本模块不提供参数寻优——
//! 在几十笔交易样本上寻优等于过拟合，会得到一组在历史上完美、在未来无意义的数字。
//! 这与 `risk_warning.rs` 的「严重度代表规则触发级别，不是发生概率」是同一个姿态。
//!
//! # 默认值（标准档）的数学含义
//!
//! - 单笔最大亏损锁死在总资产 **2%**：连输 10 笔才亏 20%，本金还在，还能继续。
//! - 盈亏比门槛 **2:1**：胜率 40% 即正期望（0.4 × 2 − 0.6 × 1 = +0.2）。
//! - 这两条合起来就是「赚大钱亏小钱」的全部数学。其余规则都是为了让这两条不被人性绕过。
//!
//! # `atr_mult` 为什么是 3.5
//!
//! 止损取四个候选里最严的一个（见 [`crate::discipline::stop`]），所以 ATR 线与固定线
//! 的相对位置由 `atr_mult × ATR%` 和 `fixed_stop_pct` 的大小决定：
//!
//! ```text
//! atr_mult × ATR% < 8%  →  ATR 线更严，它说了算
//! 交叉点：ATR% = 8 / atr_mult
//! ```
//!
//! `atr_mult = 2.0` 时交叉点是 ATR% = 4%，而 A 股日均 ATR% 多在 2–4%，等于绝大多数票
//! 都由 ATR 线接管，止损被压到成本下方 4–6%——比标称的 8% 紧一大截，低波动股会被
//! 正常震荡反复扫出。`atr_mult = 3.5` 把交叉点拉到 **ATR% = 2.29%**：
//!
//! - ATR% ≥ 2.29%（多数 A 股）→ **固定 8% 是主力线**
//! - ATR% < 2.29%（低波动股）→ ATR 线仍接管，但落在 −7% 附近，与固定线只差 1 个点
//!
//! ATR 项因此从「日常止损线」退回它该有的角色：只在真正安静的票上稍作收紧。
//! 想让 8% 在任何波动率下都当家，把 `atr_mult` 提到 4.0（交叉点 ATR% = 2%）以上。

use serde::{Deserialize, Serialize};

use crate::prediction::types::RiskLevel;

/// 全部纪律阈值。序列化后存进 `discipline_account.rules_json`。
///
/// 每个字段都标了 `#[serde(default = ...)]`，所以新增规则时旧的 rules_json 仍能反序列化，
/// 缺失字段自动取默认值 —— 不需要写数据迁移。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DisciplineRules {
    // —— 止损三口径 ——
    /// 固定百分比止损：成本 × (1 − x)
    #[serde(default = "d_fixed_stop_pct")]
    pub fixed_stop_pct: f64,
    /// ATR 倍数：成本 − n × ATR
    #[serde(default = "d_atr_mult")]
    pub atr_mult: f64,
    #[serde(default = "d_atr_period")]
    pub atr_period: usize,
    /// 支撑位缓冲：支撑 × (1 − x)
    #[serde(default = "d_support_buffer_pct")]
    pub support_buffer_pct: f64,
    /// 买入时允许的最大止损宽度；超过说明这笔的技术位太远，不该做
    #[serde(default = "d_max_stop_pct")]
    pub max_stop_pct: f64,

    // —— 止盈 ——
    /// 移动止盈启动线：浮盈达到 x 后才开始跟踪回撤
    #[serde(default = "d_trail_arm_pct")]
    pub trail_arm_pct: f64,
    /// 从持仓最高价回撤 x 即清仓
    #[serde(default = "d_trail_pct")]
    pub trail_pct: f64,
    /// 分批止盈档位（浮盈百分比）
    #[serde(default = "d_scale_out_tiers")]
    pub scale_out_tiers: [f64; 2],
    /// 每档减仓比例
    #[serde(default = "d_scale_out_fraction")]
    pub scale_out_fraction: f64,

    // —— 时间止损 ——
    /// 持有满 n 个交易日
    #[serde(default = "d_time_stop_bars")]
    pub time_stop_bars: i64,
    /// 且浮盈不足 x 即换股（机会成本）
    #[serde(default = "d_time_stop_min_gain_pct")]
    pub time_stop_min_gain_pct: f64,

    // —— 买入准入 ——
    /// 单笔最大亏损占总资产比例
    #[serde(default = "d_max_risk_pct")]
    pub max_risk_pct: f64,
    /// 最低盈亏比
    #[serde(default = "d_min_rr")]
    pub min_rr: f64,
    /// 允许买入的最高风险等级；超过即拒绝。只用于否决，不用于放行。
    #[serde(default = "d_max_entry_risk_level")]
    pub max_entry_risk_level: RiskLevel,

    // —— 仓位上限 ——
    #[serde(default = "d_max_single_pct")]
    pub max_single_pct: f64,
    #[serde(default = "d_max_total_pct")]
    pub max_total_pct: f64,
    #[serde(default = "d_max_holdings")]
    pub max_holdings: usize,

    // —— 冷静期 / 熔断 ——
    /// 某股止损离场后 n 个交易日内禁买（防报复性交易）
    #[serde(default = "d_cooldown_bars")]
    pub cooldown_bars: i64,
    /// 连续亏损 n 笔触发熔断
    #[serde(default = "d_breaker_losses")]
    pub breaker_losses: i64,
    /// 熔断后停手 n 个交易日
    #[serde(default = "d_breaker_bars")]
    pub breaker_bars: i64,

    // —— 数据可用性 ——
    /// 距最新 K 线超过 n 个自然日即提示数据陈旧（对齐 `risk_warning::add_data_staleness`）
    #[serde(default = "d_stale_days")]
    pub stale_days: i64,
}

fn d_fixed_stop_pct() -> f64 { 8.0 }
fn d_atr_mult() -> f64 { 3.5 }
fn d_atr_period() -> usize { 14 }
fn d_support_buffer_pct() -> f64 { 1.0 }
fn d_max_stop_pct() -> f64 { 10.0 }
fn d_trail_arm_pct() -> f64 { 10.0 }
fn d_trail_pct() -> f64 { 8.0 }
fn d_scale_out_tiers() -> [f64; 2] { [15.0, 30.0] }
fn d_scale_out_fraction() -> f64 { 0.5 }
fn d_time_stop_bars() -> i64 { 20 }
fn d_time_stop_min_gain_pct() -> f64 { 3.0 }
fn d_max_risk_pct() -> f64 { 2.0 }
fn d_min_rr() -> f64 { 2.0 }
fn d_max_entry_risk_level() -> RiskLevel { RiskLevel::Medium }
fn d_max_single_pct() -> f64 { 25.0 }
fn d_max_total_pct() -> f64 { 80.0 }
fn d_max_holdings() -> usize { 5 }
fn d_cooldown_bars() -> i64 { 10 }
fn d_breaker_losses() -> i64 { 3 }
fn d_breaker_bars() -> i64 { 5 }
fn d_stale_days() -> i64 { 4 }

impl Default for DisciplineRules {
    fn default() -> Self {
        Self {
            fixed_stop_pct: d_fixed_stop_pct(),
            atr_mult: d_atr_mult(),
            atr_period: d_atr_period(),
            support_buffer_pct: d_support_buffer_pct(),
            max_stop_pct: d_max_stop_pct(),
            trail_arm_pct: d_trail_arm_pct(),
            trail_pct: d_trail_pct(),
            scale_out_tiers: d_scale_out_tiers(),
            scale_out_fraction: d_scale_out_fraction(),
            time_stop_bars: d_time_stop_bars(),
            time_stop_min_gain_pct: d_time_stop_min_gain_pct(),
            max_risk_pct: d_max_risk_pct(),
            min_rr: d_min_rr(),
            max_entry_risk_level: d_max_entry_risk_level(),
            max_single_pct: d_max_single_pct(),
            max_total_pct: d_max_total_pct(),
            max_holdings: d_max_holdings(),
            cooldown_bars: d_cooldown_bars(),
            breaker_losses: d_breaker_losses(),
            breaker_bars: d_breaker_bars(),
            stale_days: d_stale_days(),
        }
    }
}

impl DisciplineRules {
    /// 从 `discipline_account.rules_json` 反序列化。空串或解析失败一律回落到默认值——
    /// 纪律配置坏掉时用出厂值继续跑，比让整个页面报错更安全。
    pub fn from_json(raw: &str) -> Self {
        if raw.trim().is_empty() {
            return Self::default();
        }
        serde_json::from_str(raw).unwrap_or_default()
    }

    /// 把明显不合理的输入夹回可用范围，避免用户填 0 导致除零或规则失效。
    ///
    /// 注意夹取方向：全部朝「更严格」夹。纪律配置的错误容忍必须偏保守。
    pub fn sanitized(mut self) -> Self {
        self.fixed_stop_pct = self.fixed_stop_pct.clamp(1.0, 20.0);
        self.atr_mult = self.atr_mult.clamp(0.5, 5.0);
        self.atr_period = self.atr_period.clamp(5, 60);
        self.support_buffer_pct = self.support_buffer_pct.clamp(0.0, 5.0);
        self.max_stop_pct = self.max_stop_pct.clamp(2.0, 20.0);
        self.trail_arm_pct = self.trail_arm_pct.clamp(1.0, 100.0);
        self.trail_pct = self.trail_pct.clamp(1.0, 50.0);
        self.scale_out_tiers[0] = self.scale_out_tiers[0].clamp(1.0, 200.0);
        self.scale_out_tiers[1] = self.scale_out_tiers[1].clamp(self.scale_out_tiers[0], 500.0);
        self.scale_out_fraction = self.scale_out_fraction.clamp(0.1, 0.9);
        self.time_stop_bars = self.time_stop_bars.clamp(1, 250);
        self.time_stop_min_gain_pct = self.time_stop_min_gain_pct.clamp(0.0, 50.0);
        self.max_risk_pct = self.max_risk_pct.clamp(0.1, 10.0);
        self.min_rr = self.min_rr.clamp(1.0, 10.0);
        self.max_single_pct = self.max_single_pct.clamp(1.0, 100.0);
        self.max_total_pct = self.max_total_pct.clamp(1.0, 100.0);
        self.max_holdings = self.max_holdings.clamp(1, 50);
        self.cooldown_bars = self.cooldown_bars.clamp(0, 250);
        self.breaker_losses = self.breaker_losses.clamp(1, 20);
        self.breaker_bars = self.breaker_bars.clamp(0, 250);
        self.stale_days = self.stale_days.clamp(1, 60);
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rules_encode_the_standard_preset() {
        let rules = DisciplineRules::default();
        assert_eq!(rules.fixed_stop_pct, 8.0, "标准档硬止损应为 8%");
        assert_eq!(rules.atr_mult, 3.5, "ATR 倍数 3.5 让交叉点落在 ATR%=2.29%，8% 固定线才是主力");
        assert!(
            rules.atr_mult * 2.5 > rules.fixed_stop_pct,
            "ATR%=2.5%（A 股常见水平）时 ATR 线必须让位给固定线，否则止损会被压到 8% 以内"
        );
        assert_eq!(rules.max_risk_pct, 2.0, "标准档单笔风险应为总资产 2%");
        assert_eq!(rules.min_rr, 2.0, "标准档盈亏比门槛应为 2:1");
        assert_eq!(rules.max_single_pct, 25.0, "标准档单票上限应为 25%");
        assert_eq!(rules.max_holdings, 5, "标准档持股上限应为 5 只");
        assert_eq!(rules.scale_out_tiers, [15.0, 30.0], "标准档分批止盈应为 +15%/+30%");
        assert_eq!(rules.scale_out_fraction, 0.5, "标准档分批止盈应为减半");
    }

    #[test]
    fn missing_fields_fall_back_to_defaults() {
        // 只给一个字段，模拟"新增规则后读到的旧 rules_json"
        let rules = DisciplineRules::from_json(r#"{"fixed_stop_pct": 5.0}"#);
        assert_eq!(rules.fixed_stop_pct, 5.0, "已存字段应保留");
        assert_eq!(rules.max_risk_pct, 2.0, "缺失字段应取默认值，不需要数据迁移");
    }

    #[test]
    fn broken_json_falls_back_to_defaults_instead_of_failing() {
        assert_eq!(DisciplineRules::from_json("not json"), DisciplineRules::default());
        assert_eq!(DisciplineRules::from_json(""), DisciplineRules::default());
    }

    #[test]
    fn sanitize_rejects_zero_risk_budget() {
        let broken = DisciplineRules { max_risk_pct: 0.0, ..Default::default() };
        assert!(broken.sanitized().max_risk_pct >= 0.1, "单笔风险为 0 会让所有买入都算出 0 股");
    }

    #[test]
    fn sanitize_keeps_scale_out_tiers_ordered() {
        let inverted = DisciplineRules { scale_out_tiers: [40.0, 10.0], ..Default::default() };
        let fixed = inverted.sanitized();
        assert!(fixed.scale_out_tiers[1] >= fixed.scale_out_tiers[0], "第二档不应低于第一档");
    }

    #[test]
    fn json_round_trip_is_stable() {
        let rules = DisciplineRules::default();
        assert_eq!(DisciplineRules::from_json(&rules.to_json()), rules);
    }
}
