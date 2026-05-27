//! 滚动截面多因子排序（市场中性相对强弱）
//!
//! 流程：每个交易日对全市场做因子截面 z-score → 滚动窗口估计各因子 IC 作为权重
//! → 合成打分 → 排序选相对强弱。这是经验证有正样本外 IC 的方法。

use crate::db::models::HistoricalData;
use crate::prediction::factor::{compute_factor_row, factor_dim, FACTOR_LOOKBACK};
use chrono::NaiveDate;
use std::collections::BTreeMap;

/// 单个截面样本（因子已截面标准化）
pub struct PanelRow {
    pub symbol: String,
    pub factors: Vec<f64>,
    pub fwd_return: f64,
}

/// 排名结果
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RankedStock {
    pub symbol: String,
    pub score: f64,
    pub rank: usize,
}

/// 走步评估报告
#[derive(Debug, Clone)]
pub struct WalkForwardReport {
    pub oos_days: usize,
    pub rank_ic: f64,
    pub direction_accuracy: f64,
    pub long_short_spread: f64,
}

/// 皮尔逊相关
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        cov += (x - mx) * (y - my);
        vx += (x - mx).powi(2);
        vy += (y - my).powi(2);
    }
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 最小截面股票数（少于此不计入）
const MIN_CROSS: usize = 5;

/// 构建按日截面面板：逐日做因子截面 z-score。
///
/// `horizon`>0 时计算未来收益（评估用）；=0 时 fwd_return 记为 0（仅用于打分）。
/// 返回按日期升序的截面序列。
pub fn build_panel(
    stocks: &[(String, Vec<HistoricalData>)],
    horizon: usize,
) -> Vec<Vec<PanelRow>> {
    let dim = factor_dim();
    let mut by_date: BTreeMap<NaiveDate, Vec<(String, Vec<f64>, f64)>> = BTreeMap::new();

    for (sym, hist) in stocks {
        let len = hist.len();
        let end = if horizon > 0 {
            len.saturating_sub(horizon)
        } else {
            len
        };
        for i in FACTOR_LOOKBACK..end {
            if let Some(f) = compute_factor_row(hist, i) {
                let base = hist[i].close;
                let fwd = if horizon > 0 && base > 0.0 {
                    (hist[i + horizon].close - base) / base
                } else {
                    0.0
                };
                by_date
                    .entry(hist[i].date)
                    .or_default()
                    .push((sym.clone(), f, fwd));
            }
        }
    }

    let mut panel = Vec::new();
    for (_date, items) in by_date {
        if items.len() < MIN_CROSS {
            continue;
        }
        panel.push(standardize(items, dim));
    }
    panel
}

/// 对一个截面做因子 z-score
fn standardize(items: Vec<(String, Vec<f64>, f64)>, dim: usize) -> Vec<PanelRow> {
    let n = items.len() as f64;
    let mut mean = vec![0.0; dim];
    let mut std = vec![0.0; dim];
    for (_, f, _) in &items {
        for d in 0..dim {
            mean[d] += f[d];
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    for (_, f, _) in &items {
        for d in 0..dim {
            std[d] += (f[d] - mean[d]).powi(2);
        }
    }
    for s in std.iter_mut() {
        *s = (*s / n).sqrt();
    }
    items
        .into_iter()
        .map(|(symbol, f, fwd)| {
            let factors = (0..dim)
                .map(|d| if std[d] > 1e-9 { (f[d] - mean[d]) / std[d] } else { 0.0 })
                .collect();
            PanelRow {
                symbol,
                factors,
                fwd_return: fwd,
            }
        })
        .collect()
}

/// 用一段截面日估计每个因子的平均 IC（作为权重）
fn estimate_weights(days: &[Vec<PanelRow>], dim: usize) -> Vec<f64> {
    let mut w = vec![0.0; dim];
    let mut cnt = 0usize;
    for day in days {
        if day.len() < MIN_CROSS {
            continue;
        }
        let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();
        for d in 0..dim {
            let xs: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();
            w[d] += pearson(&xs, &ys);
        }
        cnt += 1;
    }
    if cnt > 0 {
        for x in w.iter_mut() {
            *x /= cnt as f64;
        }
    }
    w
}

/// 合成打分
fn composite(row: &PanelRow, w: &[f64]) -> f64 {
    row.factors.iter().zip(w).map(|(f, wi)| f * wi).sum()
}

/// 前向滚动评估：每期用过去 `window` 个截面日估权重，对当日样本外评估。
pub fn walk_forward(panel: &[Vec<PanelRow>], window: usize) -> WalkForwardReport {
    let dim = factor_dim();
    let (mut ic_sum, mut ic_n) = (0.0, 0usize);
    let (mut correct, mut total) = (0usize, 0usize);
    let (mut ls_sum, mut ls_n) = (0.0, 0usize);

    for t in window..panel.len() {
        let w = estimate_weights(&panel[t - window..t], dim);
        let day = &panel[t];
        if day.len() < MIN_CROSS {
            continue;
        }
        let comp: Vec<f64> = day.iter().map(|r| composite(r, &w)).collect();
        let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();

        for (c, y) in comp.iter().zip(&ys) {
            if (*c > 0.0 && *y > 0.0) || (*c < 0.0 && *y < 0.0) {
                correct += 1;
            }
            total += 1;
        }
        ic_sum += pearson(&comp, &ys);
        ic_n += 1;

        let mut order: Vec<usize> = (0..day.len()).collect();
        order.sort_by(|&a, &b| comp[b].partial_cmp(&comp[a]).unwrap());
        let k = (day.len() as f64 * 0.2).ceil() as usize;
        if k >= 1 {
            let top = order[..k].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            let bot = order[day.len() - k..].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            ls_sum += top - bot;
            ls_n += 1;
        }
    }

    WalkForwardReport {
        oos_days: ic_n,
        rank_ic: ic_sum / ic_n.max(1) as f64,
        direction_accuracy: correct as f64 / total.max(1) as f64,
        long_short_spread: ls_sum / ls_n.max(1) as f64,
    }
}

/// 生产打分：用最近 `window` 个截面日（带未来收益）估权重，对**最新交易日**截面打分排序。
///
/// `stocks` 为各股票按日期升序的历史数据；`horizon` 为权重估计所用的未来收益周期。
pub fn rank_latest(
    stocks: &[(String, Vec<HistoricalData>)],
    horizon: usize,
    window: usize,
) -> Vec<RankedStock> {
    let dim = factor_dim();

    // 1. 历史面板（带 fwd）估权重
    let hist_panel = build_panel(stocks, horizon.max(1));
    if hist_panel.len() < window.max(20) {
        return Vec::new();
    }
    let w = estimate_weights(&hist_panel[hist_panel.len() - window..], dim);

    // 2. 最新一日截面（无 fwd）：取每只股票最后一根的因子
    let mut latest: Vec<(String, Vec<f64>, f64)> = Vec::new();
    for (sym, hist) in stocks {
        if hist.len() <= FACTOR_LOOKBACK {
            continue;
        }
        if let Some(f) = compute_factor_row(hist, hist.len() - 1) {
            latest.push((sym.clone(), f, 0.0));
        }
    }
    if latest.len() < MIN_CROSS {
        return Vec::new();
    }
    let today = standardize(latest, dim);

    // 3. 打分排序（降序）
    let mut scored: Vec<(String, f64)> = today
        .iter()
        .map(|r| (r.symbol.clone(), composite(r, &w)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scored
        .into_iter()
        .enumerate()
        .map(|(idx, (symbol, score))| RankedStock {
            symbol,
            score,
            rank: idx + 1,
        })
        .collect()
}
