//! 滚动截面多因子排序（市场中性相对强弱）
//!
//! 流程：每个交易日对全市场做因子截面 z-score → 滚动窗口估计各因子 IC 作为权重
//! → 合成打分 → 排序选相对强弱。这是经验证有正样本外 IC 的方法。

use crate::db::models::HistoricalData;
use crate::prediction::factor::{compute_factor_row, factor_dim, FACTOR_LOOKBACK};
use chrono::NaiveDate;
use std::collections::BTreeMap;

/// 单个截面样本（因子已截面标准化）
#[derive(Debug, Clone)]
pub struct PanelRow {
    pub date: NaiveDate,
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

/// 样本外截面排名信号
#[derive(Debug, Clone)]
pub struct CrossSectionRankSignal {
    pub date: NaiveDate,
    pub symbol: String,
    pub score: f64,
    pub rank: usize,
    pub total: usize,
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
        panel.push(standardize(_date, items, dim));
    }
    panel
}

/// 对一个截面做因子 z-score
fn standardize(date: NaiveDate, items: Vec<(String, Vec<f64>, f64)>, dim: usize) -> Vec<PanelRow> {
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
                date,
                symbol,
                factors,
                fwd_return: fwd,
            }
        })
        .collect()
}

/// 因子正交化（逐日 Gram-Schmidt 残差化，去除因子间冗余/共线）。
///
/// 按给定 `order`（通常按 |IC| 降序）依次保留因子，将后续因子对已保留因子做
/// 截面回归取残差，再各列重新 z-score。消除多重共线，使 IC 加权不重复计数。
pub fn orthogonalize_panel(mut panel: Vec<Vec<PanelRow>>, order: &[usize]) -> Vec<Vec<PanelRow>> {
    let dim = factor_dim();
    for day in panel.iter_mut() {
        let n = day.len();
        if n < MIN_CROSS {
            continue;
        }
        // 提取各列
        let mut cols: Vec<Vec<f64>> = (0..dim)
            .map(|d| day.iter().map(|r| r.factors[d]).collect())
            .collect();

        let mut basis: Vec<Vec<f64>> = Vec::new();
        for &d in order {
            // 对已入基的正交列做残差化
            let mut col = cols[d].clone();
            for b in &basis {
                let bb: f64 = b.iter().map(|x| x * x).sum();
                if bb <= 1e-12 {
                    continue;
                }
                let cb: f64 = col.iter().zip(b).map(|(c, x)| c * x).sum();
                let coef = cb / bb;
                for (c, x) in col.iter_mut().zip(b) {
                    *c -= coef * x;
                }
            }
            basis.push(col.clone());
            cols[d] = col;
        }

        // 各列重新 z-score 后写回
        for (d, col) in cols.iter().enumerate().take(dim) {
            let mean = col.iter().sum::<f64>() / n as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let sd = var.sqrt();
            for (r, &v) in day.iter_mut().zip(col.iter()) {
                r.factors[d] = if sd > 1e-9 { (v - mean) / sd } else { 0.0 };
            }
        }
    }
    panel
}

/// 计算各因子全样本平均 IC（按 |IC| 降序返回索引，用于正交化顺序）
pub fn factor_ic_order(panel: &[Vec<PanelRow>]) -> Vec<usize> {
    let dim = factor_dim();
    let mut ic = vec![0.0; dim];
    let mut cnt = 0usize;
    for day in panel {
        if day.len() < MIN_CROSS {
            continue;
        }
        let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();
        for (d, factor_ic) in ic.iter_mut().enumerate().take(dim) {
            let xs: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();
            *factor_ic += pearson(&xs, &ys);
        }
        cnt += 1;
    }
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        (ic[b] / cnt.max(1) as f64).abs().total_cmp(&(ic[a] / cnt.max(1) as f64).abs())
    });
    order
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
        for (d, weight) in w.iter_mut().enumerate().take(dim) {
            let xs: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();
            *weight += pearson(&xs, &ys);
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

/// 前向滚动评估：每期只用预测日前已完成的历史收益估权重。
pub fn walk_forward(panel: &[Vec<PanelRow>], window: usize, horizon: usize) -> WalkForwardReport {
    let dim = factor_dim();
    let horizon = horizon.max(1);
    let (mut ic_sum, mut ic_n) = (0.0, 0usize);
    let (mut correct, mut total) = (0usize, 0usize);
    let (mut ls_sum, mut ls_n) = (0.0, 0usize);

    for t in window + horizon - 1..panel.len() {
        let train_end = t - horizon + 1;
        let w = estimate_weights(&panel[train_end - window..train_end], dim);
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
        order.sort_by(|&a, &b| comp[b].total_cmp(&comp[a]));
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

/// 前向滚动正交化评估：每期只用预测日前已完成的历史收益决定因子顺序和权重。
pub fn walk_forward_orthogonalized(
    panel: &[Vec<PanelRow>],
    window: usize,
    horizon: usize,
) -> WalkForwardReport {
    let dim = factor_dim();
    let horizon = horizon.max(1);
    let (mut ic_sum, mut ic_n) = (0.0, 0usize);
    let (mut correct, mut total) = (0usize, 0usize);
    let (mut ls_sum, mut ls_n) = (0.0, 0usize);

    for t in window + horizon - 1..panel.len() {
        let train_end = t - horizon + 1;
        let train_panel = panel[train_end - window..train_end].to_vec();
        let order = factor_ic_order(&train_panel);
        let train_panel = orthogonalize_panel(train_panel, &order);
        let w = estimate_weights(&train_panel, dim);
        let day = orthogonalize_panel(vec![panel[t].clone()], &order)
            .into_iter()
            .next()
            .unwrap_or_default();
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
        order.sort_by(|&a, &b| comp[b].total_cmp(&comp[a]));
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

/// 前向滚动输出每个样本外交易日的截面排名信号。
pub fn walk_forward_rank_signals(
    stocks: &[(String, Vec<HistoricalData>)],
    horizon: usize,
    window: usize,
) -> Vec<CrossSectionRankSignal> {
    let hist_panel = build_panel(stocks, horizon.max(1));
    if hist_panel.len() < window.max(20) {
        return Vec::new();
    }

    let dim = factor_dim();
    let mut signals = Vec::new();

    let horizon = horizon.max(1);
    for t in window + horizon - 1..hist_panel.len() {
        let train_end = t - horizon + 1;
        let w = estimate_weights(&hist_panel[train_end - window..train_end], dim);
        let day = &hist_panel[t];
        if day.len() < MIN_CROSS {
            continue;
        }

        let mut scored: Vec<(usize, f64)> = day
            .iter()
            .enumerate()
            .map(|(idx, row)| (idx, composite(row, &w)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total = scored.len();
        for (rank_idx, (row_idx, score)) in scored.into_iter().enumerate() {
            let row = &day[row_idx];
            signals.push(CrossSectionRankSignal {
                date: row.date,
                symbol: row.symbol.clone(),
                score,
                rank: rank_idx + 1,
                total,
            });
        }
    }

    signals
}

/// 生产和回测共用的截面排名偏置：返回单日涨跌幅偏置（百分点）。
pub fn daily_bias_from_rank(rank: usize, total: usize) -> f64 {
    if total <= 1 || rank == 0 {
        return 0.0;
    }

    let percentile = (rank - 1) as f64 / (total - 1) as f64;
    if percentile <= 0.20 {
        0.005
    } else if percentile <= 0.40 {
        0.0
    } else if percentile >= 0.80 {
        -0.005
    } else {
        0.0
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
    let horizon = horizon.max(1);
    let min_visible_len = min_visible_len(horizon, window);

    let Some(ranking_date) = latest_broad_cross_section_date(stocks, horizon, window) else {
        return Vec::new();
    };
    let training_stocks: Vec<(String, Vec<HistoricalData>)> = stocks
        .iter()
        .filter_map(|(sym, hist)| {
            let visible: Vec<HistoricalData> = hist
                .iter()
                .take_while(|bar| bar.date <= ranking_date)
                .cloned()
                .collect();
            if visible.len() >= min_visible_len {
                Some((sym.clone(), visible))
            } else {
                None
            }
        })
        .collect();

    // 1. 历史面板（带 fwd）→ 估权重
    let hist_panel = build_panel(&training_stocks, horizon);
    if hist_panel.len() < window.max(20) {
        return Vec::new();
    }
    let w = estimate_weights(&hist_panel[hist_panel.len() - window..], dim);

    // 2. 最新一日截面（无 fwd）：使用覆盖足够股票数的最近共同交易日。
    let mut latest: Vec<(String, Vec<f64>, f64)> = Vec::new();
    for (sym, hist) in &training_stocks {
        if hist.len() <= FACTOR_LOOKBACK {
            continue;
        }
        if hist.last().is_none_or(|bar| bar.date != ranking_date) {
            continue;
        }
        if let Some(f) = compute_factor_row(hist, hist.len() - 1) {
            latest.push((sym.clone(), f, 0.0));
        }
    }
    if latest.len() < MIN_CROSS {
        return Vec::new();
    }
    // 截面标准化（与历史一致）
    let today = standardize(NaiveDate::MIN, latest, dim);

    // 3. 打分排序（降序）
    let mut scored: Vec<(String, f64)> = today
        .iter()
        .map(|r| (r.symbol.clone(), composite(r, &w)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

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

fn latest_broad_cross_section_date(
    stocks: &[(String, Vec<HistoricalData>)],
    horizon: usize,
    window: usize,
) -> Option<NaiveDate> {
    let min_visible_len = min_visible_len(horizon, window);
    // 有效股票数：历史足够长、能进入截面的票。
    let n_eligible = stocks
        .iter()
        .filter(|(_, hist)| hist.len() >= min_visible_len)
        .count();
    if n_eligible < MIN_CROSS {
        return None;
    }
    // 要求覆盖到「大多数」有效股票，否则少数数据更新更早/更晚的票（如分批刷新导致
    // 末日期错位）会把截面日拽到只有零星几只的稀疏日期，漏掉主体票池。
    let required = MIN_CROSS.max((n_eligible as f64 * 0.7).ceil() as usize);
    let mut counts: BTreeMap<NaiveDate, usize> = BTreeMap::new();
    for (_, hist) in stocks {
        for bar in hist.iter().skip(min_visible_len - 1) {
            *counts.entry(bar.date).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .rev()
        .find_map(|(date, count)| (count >= required).then_some(date))
}

fn min_visible_len(horizon: usize, window: usize) -> usize {
    FACTOR_LOOKBACK + horizon.max(1) + window.max(20)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn synthetic_panel(days: usize, cross: usize) -> Vec<Vec<PanelRow>> {
        let dim = factor_dim();
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..days)
            .map(|d| {
                (0..cross)
                    .map(|s| {
                        let signal = s as f64 - (cross as f64 - 1.0) / 2.0;
                        let mut factors = vec![0.0; dim];
                        factors[0] = signal;
                        factors[1] = -signal * 0.5 + d as f64 * 0.001;
                        PanelRow {
                            date: start + Duration::days(d as i64),
                            symbol: format!("s{s}"),
                            factors,
                            fwd_return: signal * 0.01,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn make_history(symbol: &str, days: usize, price_offset: f64) -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..days)
            .map(|d| {
                let close = 10.0 + price_offset + d as f64 * 0.05;
                HistoricalData {
                    symbol: symbol.to_string(),
                    date: start + Duration::days(d as i64),
                    open: close - 0.02,
                    close,
                    high: close + 0.2,
                    low: close - 0.2,
                    volume: 100_000 + d as i64 * 100,
                    amount: close * 100_000.0,
                    amplitude: 2.0,
                    turnover_rate: 2.0 + price_offset * 0.1,
                    volume_ratio: 1.0,
                    change_percent: 0.5,
                    change: 0.05,
                }
            })
            .collect()
    }

    #[test]
    fn test_walk_forward_orthogonalized_uses_oos_windows() {
        let panel = synthetic_panel(16, 6);
        let report = walk_forward_orthogonalized(&panel, 6, 2);

        assert_eq!(report.oos_days, 9);
        assert!(report.rank_ic.is_finite());
        assert!(report.direction_accuracy > 0.5);
        assert!(report.long_short_spread > 0.0);
    }

    #[test]
    fn test_daily_bias_from_rank_boundaries() {
        assert_eq!(daily_bias_from_rank(0, 10), 0.0);
        assert_eq!(daily_bias_from_rank(1, 10), 0.005);
        assert_eq!(daily_bias_from_rank(4, 10), 0.0);
        assert_eq!(daily_bias_from_rank(6, 10), 0.0);
        assert_eq!(daily_bias_from_rank(8, 10), 0.0);
        assert_eq!(daily_bias_from_rank(10, 10), -0.005);
    }

    #[test]
    fn test_rank_latest_filters_stale_symbols() {
        let mut stocks = (0..5)
            .map(|idx| {
                let symbol = format!("fresh{idx}");
                (symbol, make_history(&format!("fresh{idx}"), 60, idx as f64))
            })
            .collect::<Vec<_>>();
        stocks.push(("stale".to_string(), make_history("stale", 59, 10.0)));

        let ranking = rank_latest(&stocks, 1, 20);

        assert!(!ranking.is_empty());
        assert_eq!(ranking.len(), 5);
        assert!(ranking.iter().all(|ranked| ranked.symbol != "stale"));
    }

    #[test]
    fn test_rank_latest_ignores_sparse_fresher_date_from_batch_refresh() {
        // 主体 12 只数据停在 D；少数 6 只更新到 D+2（模拟分批刷新末日期错位）。
        // 新批次 6 只 ≥ MIN_CROSS(5) 但远不足七成，应回退到 D 排主体票池，而非只排这 6 只。
        let mut stocks = (0..12)
            .map(|idx| {
                let symbol = format!("seasoned{idx}");
                (symbol.clone(), make_history(&symbol, 60, idx as f64))
            })
            .collect::<Vec<_>>();
        stocks.extend((0..6).map(|idx| {
            let symbol = format!("fresh{idx}");
            let mut hist = make_history(&symbol, 60, idx as f64 + 20.0);
            for bar in hist.iter_mut() {
                bar.date += Duration::days(2);
            }
            (symbol, hist)
        }));

        let ranking = rank_latest(&stocks, 1, 20);

        // 回退到主体共同日 D：18 只全部参与（而非塌缩到 6 只新票）。
        assert_eq!(ranking.len(), 18);
        assert!(ranking.iter().any(|ranked| ranked.symbol == "seasoned0"));
    }

    #[test]
    fn test_rank_latest_falls_back_to_recent_broad_date() {
        let mut stocks = (0..4)
            .map(|idx| {
                let symbol = format!("latest{idx}");
                (symbol, make_history(&format!("latest{idx}"), 60, idx as f64))
            })
            .collect::<Vec<_>>();
        stocks.extend((0..2).map(|idx| {
            let symbol = format!("fallback{idx}");
            (symbol, make_history(&format!("fallback{idx}"), 59, idx as f64 + 4.0))
        }));

        let ranking = rank_latest(&stocks, 1, 20);

        assert_eq!(ranking.len(), 6);
        assert!(ranking.iter().any(|ranked| ranked.symbol == "fallback0"));
        assert!(ranking.iter().any(|ranked| ranked.symbol == "latest0"));
    }

    #[test]
    fn test_rank_latest_ignores_short_history_when_selecting_broad_date() {
        let mut stocks = (0..5)
            .map(|idx| {
                let symbol = format!("seasoned{idx}");
                (symbol, make_history(&format!("seasoned{idx}"), 60, idx as f64))
            })
            .collect::<Vec<_>>();

        stocks.extend((0..5).map(|idx| {
            let symbol = format!("short{idx}");
            let mut hist = make_history(&symbol, 10, idx as f64 + 10.0);
            for bar in hist.iter_mut() {
                bar.date += Duration::days(80);
            }
            (symbol, hist)
        }));

        let ranking = rank_latest(&stocks, 1, 20);

        assert_eq!(ranking.len(), 5);
        assert!(ranking.iter().all(|ranked| ranked.symbol.starts_with("seasoned")));
    }

    #[test]
    fn test_rank_latest_ignores_symbols_without_full_training_window() {
        let mut stocks = (0..5)
            .map(|idx| {
                let symbol = format!("seasoned{idx}");
                (symbol, make_history(&format!("seasoned{idx}"), 60, idx as f64))
            })
            .collect::<Vec<_>>();

        stocks.extend((0..5).map(|idx| {
            let symbol = format!("partial{idx}");
            let mut hist = make_history(&symbol, FACTOR_LOOKBACK + 5, idx as f64 + 20.0);
            for bar in hist.iter_mut() {
                bar.date += Duration::days(80);
            }
            (symbol, hist)
        }));

        let ranking = rank_latest(&stocks, 1, 20);

        assert_eq!(ranking.len(), 5);
        assert!(ranking.iter().all(|ranked| ranked.symbol.starts_with("seasoned")));
    }

    #[test]
    fn test_rank_latest_ignores_same_date_partial_training_window() {
        let mut stocks = (0..5)
            .map(|idx| {
                let symbol = format!("seasoned{idx}");
                (symbol, make_history(&format!("seasoned{idx}"), 60, idx as f64))
            })
            .collect::<Vec<_>>();

        let seasoned_last_date = stocks[0].1.last().unwrap().date;
        stocks.extend((0..2).map(|idx| {
            let symbol = format!("partial{idx}");
            let mut hist = make_history(&symbol, FACTOR_LOOKBACK + 5, idx as f64 + 20.0);
            let partial_last_date = hist.last().unwrap().date;
            let shift = seasoned_last_date - partial_last_date;
            for bar in hist.iter_mut() {
                bar.date += shift;
            }
            (symbol, hist)
        }));

        let ranking = rank_latest(&stocks, 1, 20);

        assert_eq!(ranking.len(), 5);
        assert!(ranking.iter().all(|ranked| ranked.symbol.starts_with("seasoned")));
    }
}
