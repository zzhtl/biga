//! 持有期 × IC估计窗口 扫描（无前视走步）：每次扩广度/改因子后，用它重找净收益最优配置。
//! 裁判：`cross_section::walk_forward`（已去前视）。报告 Rank IC、扣费净多空、年化、
//! 以及**非重叠样本** t 统计量（避免重叠收益高估自由度）。
//! 用法：cargo run --release --example factor_sweep
//!
//! 经验：5日调仓被成本拖成净负；10–20日持有 × 长IC窗口才进稳健净正区。详见 .claude/CLAUDE.md。

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data;
use biga_lib::prediction::cross_section::{build_panel, walk_forward, PanelRow};
use sqlx::Row;

const COST: f64 = 0.003; // 每次调仓多空双边成本假设 0.3%
const TRADING_DAYS: f64 = 250.0;
const HORIZONS: &[usize] = &[5, 10, 15, 20, 30];
const WINDOWS: &[usize] = &[120, 250];

/// 由每个样本外日的多空价差序列算非重叠 t 统计量（每隔 horizon 天取一点）。
fn nonoverlap_t(ls_series: &[f64], horizon: usize) -> (f64, usize) {
    let indep: Vec<f64> = ls_series.iter().step_by(horizon.max(1)).map(|x| x - COST).collect();
    let m = indep.len();
    if m < 2 {
        return (0.0, m);
    }
    let mean = indep.iter().sum::<f64>() / m as f64;
    let var = indep.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m as f64 - 1.0);
    let sd = var.sqrt();
    if sd > 1e-12 {
        (mean / (sd / (m as f64).sqrt()), m)
    } else {
        (0.0, m)
    }
}

/// 重算每日多空价差序列（与 walk_forward 同口径，用于 t 统计）。复用 walk_forward 的报告取均值，
/// 这里只需序列，故内联一份等价计算。
fn long_short_series(panel: &[Vec<PanelRow>], window: usize, horizon: usize) -> Vec<f64> {
    use biga_lib::prediction::cross_section::pearson;
    let horizon = horizon.max(1);
    let dim = panel.first().map(|d| d.first().map(|r| r.factors.len()).unwrap_or(0)).unwrap_or(0);
    let mut series = Vec::new();
    for t in window + horizon - 1..panel.len() {
        let train_end = t - horizon + 1;
        // 估权重：训练窗口内逐日 Pearson IC 的均值（与 estimate_weights 同口径）
        let mut w = vec![0.0; dim];
        let mut cnt = 0usize;
        for day in &panel[train_end - window..train_end] {
            if day.len() < 5 {
                continue;
            }
            let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();
            for (d, wi) in w.iter_mut().enumerate() {
                let xs: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();
                *wi += pearson(&xs, &ys);
            }
            cnt += 1;
        }
        if cnt == 0 {
            continue;
        }
        for wi in w.iter_mut() {
            *wi /= cnt as f64;
        }
        let day = &panel[t];
        if day.len() < 5 {
            continue;
        }
        let comp: Vec<f64> = day
            .iter()
            .map(|r| r.factors.iter().zip(&w).map(|(f, x)| f * x).sum())
            .collect();
        let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();
        let mut order: Vec<usize> = (0..day.len()).collect();
        order.sort_by(|&a, &b| comp[b].partial_cmp(&comp[a]).unwrap());
        let k = (day.len() as f64 * 0.2).ceil() as usize;
        if k >= 1 {
            let top = order[..k].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            let bot = order[day.len() - k..].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            series.push(top - bot);
        }
    }
    series
}

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");
    let rows = sqlx::query(
        "SELECT symbol, COUNT(*) c FROM historical_data GROUP BY symbol HAVING c >= 300 ORDER BY c DESC",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");
    println!("参与截面的股票数（≥300根）：{}", rows.len());

    // 可选：EXCLUDE_FILE 指向一份代码清单（每行纯6位），按去后缀数字匹配剔除——
    // 用于"只跑原始票池"等对照诊断。
    let exclude: std::collections::HashSet<String> = std::env::var("EXCLUDE_FILE")
        .ok()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| {
            s.lines()
                .map(|l| l.chars().filter(|c| c.is_ascii_digit()).take(6).collect::<String>())
                .filter(|d| d.len() == 6)
                .collect()
        })
        .unwrap_or_default();
    if !exclude.is_empty() {
        println!("剔除清单代码数：{}", exclude.len());
    }
    let digits = |s: &str| s.chars().filter(|c| c.is_ascii_digit()).take(6).collect::<String>();

    // 可选：MIN_CAP（单位亿元）按流通市值设流动域下限。
    let min_cap_yi = std::env::var("MIN_CAP").ok().and_then(|s| s.parse::<f64>().ok());
    let cap_map: std::collections::HashMap<String, f64> = if min_cap_yi.is_some() {
        sqlx::query("SELECT symbol, circulating_market_cap FROM stock_capital")
            .fetch_all(&pool)
            .await
            .unwrap_or_default()
            .into_iter()
            .map(|r| (r.get::<String, _>("symbol"), r.get::<f64, _>("circulating_market_cap")))
            .collect()
    } else {
        Default::default()
    };
    if let Some(c) = min_cap_yi {
        println!("流动域市值下限：{c}亿");
    }

    let mut stocks: Vec<(String, Vec<HistoricalData>)> = Vec::new();
    for r in &rows {
        let symbol: String = r.get("symbol");
        if exclude.contains(&digits(&symbol)) {
            continue;
        }
        if let Some(floor) = min_cap_yi {
            let cap = cap_map.get(&symbol).copied().unwrap_or(0.0) / 1e8;
            if cap < floor {
                continue;
            }
        }
        let hist = get_recent_historical_data(&symbol, 800, &pool)
            .await
            .unwrap_or_default();
        if hist.len() >= 300 {
            stocks.push((symbol, hist));
        }
    }
    println!("满足 ≥300 根的股票数：{}\n", stocks.len());

    for &window in WINDOWS {
        println!("----- IC估计窗口 {window} 日 -----");
        for &horizon in HORIZONS {
            let panel = build_panel(&stocks, horizon);
            if panel.len() < window + horizon + 30 {
                println!("  持有{horizon}日: 面板不足，跳过");
                continue;
            }
            let rep = walk_forward(&panel, window, horizon);
            let series = long_short_series(&panel, window, horizon);
            let (t, n) = nonoverlap_t(&series, horizon);
            let net = rep.long_short_spread - COST;
            println!(
                "  持有{horizon:>2}日  RankIC {:+.4} | 多空 {:+.3}%/期 | 净 {:+.3}%/期 | 年化净 {:+6.1}% | t={:+.2}(n={}) | OOS{}",
                rep.rank_ic,
                rep.long_short_spread * 100.0,
                net * 100.0,
                net * (TRADING_DAYS / horizon as f64) * 100.0,
                t,
                n,
                rep.oos_days
            );
        }
        println!();
    }
}
