//! 交易纪律引擎 —— 执行约束层，不是预测层。
//!
//! # 边界（与 `commands/watchlist.rs` 的「诚实预测原则」一致）
//!
//! 本模块**不新增任何方向规则**。全部输入是客观事实：成本价、收盘价、持仓期最高价、
//! ATR、支撑位、持有交易日数、账户资金。它回答的是「在什么条件下不该继续持有」和
//! 「这笔买入是否符合资金管理」，不回答「会涨还是会跌」。
//!
//! 三条可被 grep 验证的硬约束：
//!
//! 1. `prediction/` 永远不依赖 `discipline/`（依赖单向）。
//! 2. 本模块不出现 `direction` / `expected_change` / `signal_strength` / `confidence`。
//!    唯一允许透传的预测层产物是 [`RiskLevel`](crate::prediction::types::RiskLevel)，
//!    且**只用于否决，不用于放行**。
//! 3. [`entry::screen_entry`] 返回 `allowed = true` 的语义是「没有违反你自己设的规则」，
//!    **不是**「建议买入」。前端文案必须是"未发现纪律冲突"。
//!
//! # 阈值的诚实性
//!
//! [`rules::DisciplineRules`] 的默认值是行业约定俗成的经验值，**不是从本项目数据优化出来的**。
//! 本模块不提供参数寻优 —— 在几十笔样本上寻优等于过拟合。这与 `risk_warning.rs`
//! 的「严重度代表规则触发级别，不是发生概率」是同一个姿态。
//!
//! # 子模块
//!
//! - [`types`]：输入 / 输出 / 裁决单元
//! - [`rules`]：阈值配置
//! - [`stop`]：止损线（三者取最严 + 棘轮）
//! - [`sizing`]：头寸规模 + A 股手数取整
//! - [`facts`]：从日线序列提取 [`types::MarketFacts`]
//! - [`exit`]：卖出纪律
//! - [`entry`]：买入准入
//! - [`replay`]：严格守纪回放（复盘）
//!
//! 沿用 `analysis/mod.rs` 对 `risk_warning` 的做法：只 `pub mod`，不 glob re-export，
//! 调用方写全路径，避免 glob 污染。

pub mod entry;
pub mod exit;
pub mod facts;
pub mod replay;
pub mod rules;
pub mod sizing;
pub mod stop;
pub mod types;
