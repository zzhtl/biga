//! 风格中性化截面实验（CLAUDE.md 背书的"下一步"：风格中性化是截面前提）。
//!
//! 病因（见 .claude/CLAUDE.md 第2/3条）：原始技术因子在大盘/小盘上符号相反，
//! 不中性化直接混排 = 会翻车的隐性 size 押注。此前因"行业分类缺数据"未做中性化，
//! 现 DB 已含 industry（5389 只）+ 流通市值，可补上这步。
//!
//! 本实验对症对比：原始因子 vs 市值(log cap)中性化 vs 市值+粗行业中性化，
//! 用去前视 walk_forward，跨票池交叉验证，并用**非重叠样本**算 t（每 horizon 抽一点）。
//!
//! 用法：cargo run --release --example neutral_cross_section

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data_for_symbols;
use biga_lib::prediction::cross_section::{build_panel, pearson, PanelRow};
use biga_lib::prediction::factor::factor_dim;
use sqlx::Row;
use std::collections::HashMap;

const MIN_CROSS: usize = 5;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // 市值（流通市值，元）。单一快照（当前），逐日截面内跨股票仍有差异，可做 size 中性化。
    let cap: HashMap<String, f64> = sqlx::query(
        "SELECT symbol, circulating_market_cap c FROM stock_capital WHERE circulating_market_cap > 0",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.get::<String, _>("symbol"), r.get::<f64, _>("c")))
    .collect();

    // 行业 → 粗 sector
    let industry: HashMap<String, String> = sqlx::query(
        "SELECT symbol, industry i FROM stock WHERE industry IS NOT NULL AND TRIM(industry) <> ''",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.get::<String, _>("symbol"), r.get::<String, _>("i")))
    .collect();
    let sector: HashMap<String, String> = industry
        .iter()
        .map(|(s, ind)| (s.clone(), macro_sector(ind).to_string()))
        .collect();

    // 历史 ≥300 根 且 有市值 的票
    let symbols: Vec<String> = sqlx::query(
        "SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 300 ORDER BY symbol",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| r.get::<String, _>("symbol"))
    .filter(|s| cap.get(s).copied().unwrap_or(0.0) > 0.0)
    .collect();

    let stocks: Vec<(String, Vec<HistoricalData>)> =
        get_recent_historical_data_for_symbols(&symbols, 900, &pool)
            .await
            .unwrap()
            .into_iter()
            .filter(|(_, h)| h.len() >= 300)
            .collect();
    println!("可用票数（历史≥300 且有市值）：{}", stocks.len());

    // 票池：full / 市值≥200亿大中盘 / 市值<200亿（含小盘，做对照）
    let pools: Vec<(&str, Box<dyn Fn(&str) -> bool>)> = vec![
        ("full", Box::new(|_: &str| true)),
        (
            "cap>=200亿",
            {
                let cap = cap.clone();
                Box::new(move |s: &str| cap.get(s).copied().unwrap_or(0.0) >= 200.0e8)
            },
        ),
        (
            "cap<200亿",
            {
                let cap = cap.clone();
                Box::new(move |s: &str| cap.get(s).copied().unwrap_or(0.0) < 200.0e8)
            },
        ),
    ];

    println!(
        "\n{:<10} {:<10} {:>4} {:>4} {:>10} {:>11} {:>9} {:>8} {:>6}",
        "pool", "neutralize", "h", "win", "rank_ic", "LS/期(%)", "t(非重叠)", "方向acc", "n_oos"
    );
    println!("{}", "-".repeat(82));

    for (pool_name, in_pool) in &pools {
        let pool_stocks: Vec<(String, Vec<HistoricalData>)> = stocks
            .iter()
            .filter(|(s, _)| in_pool(s))
            .cloned()
            .collect();
        if pool_stocks.len() < 20 {
            println!("{pool_name}: 票数不足({}) 跳过", pool_stocks.len());
            continue;
        }

        for &(horizon, window) in &[(5usize, 250usize), (15usize, 250usize)] {
            let raw_panel = build_panel(&pool_stocks, horizon);
            if raw_panel.len() < window + horizon {
                continue;
            }
            for neut in ["none", "size", "size+sector"] {
                let panel = match neut {
                    "none" => raw_panel.clone(),
                    "size" => neutralize(&raw_panel, &cap, &sector, false),
                    _ => neutralize(&raw_panel, &cap, &sector, true),
                };
                let r = walk_forward_series(&panel, window, horizon);
                println!(
                    "{:<10} {:<10} {:>4} {:>4} {:>+10.4} {:>+11.3} {:>+9.2} {:>7.1}% {:>6}",
                    pool_name,
                    neut,
                    horizon,
                    window,
                    r.rank_ic,
                    r.ls_per_period * 100.0,
                    r.t_nonoverlap,
                    r.direction_accuracy * 100.0,
                    r.oos_days,
                );
            }
        }
        println!();
    }
    println!(
        "说明：LS/期=20%多空组每持有期净收益(%); t用非重叠样本(每{}/{}日抽一点)算; rank_ic为截面Pearson IC均值。",
        5, 15
    );
}

struct WfSeries {
    oos_days: usize,
    rank_ic: f64,
    direction_accuracy: f64,
    ls_per_period: f64,
    t_nonoverlap: f64,
}

/// 去前视走步：每期只用 t-horizon+1 前已兑现标签的窗口估权重（Pearson 均值 IC）。
/// 额外收集每个 OOS 日的多空收益序列，用非重叠抽样算 t。
fn walk_forward_series(panel: &[Vec<PanelRow>], window: usize, horizon: usize) -> WfSeries {
    let dim = factor_dim();
    let horizon = horizon.max(1);
    let (mut ic_sum, mut ic_n) = (0.0, 0usize);
    let (mut correct, mut total) = (0usize, 0usize);
    let mut ls_series: Vec<f64> = Vec::new();

    for t in window + horizon - 1..panel.len() {
        let train_end = t - horizon + 1;
        let w = estimate_weights(&panel[train_end - window..train_end], dim);
        let day = &panel[t];
        if day.len() < MIN_CROSS {
            ls_series.push(f64::NAN);
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
        let ls = if k >= 1 {
            let top = order[..k].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            let bot = order[day.len() - k..].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            top - bot
        } else {
            f64::NAN
        };
        ls_series.push(ls);
    }

    // 非重叠抽样：每 horizon 个 OOS 日取一个，规避重叠收益的假大样本。
    let sampled: Vec<f64> = ls_series
        .iter()
        .step_by(horizon)
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let t_stat = t_statistic(&sampled);
    let ls_mean = {
        let v: Vec<f64> = ls_series.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };

    WfSeries {
        oos_days: ic_n,
        rank_ic: ic_sum / ic_n.max(1) as f64,
        direction_accuracy: correct as f64 / total.max(1) as f64,
        ls_per_period: ls_mean,
        t_nonoverlap: t_stat,
    }
}

fn t_statistic(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    if var <= 0.0 {
        return 0.0;
    }
    mean / (var.sqrt() / (n as f64).sqrt())
}

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

fn composite(row: &PanelRow, w: &[f64]) -> f64 {
    row.factors.iter().zip(w).map(|(f, wi)| f * wi).sum()
}

/// 逐日把每个因子对 log(流通市值)（+可选粗行业组均值）中性化，残差再 z-score。
fn neutralize(
    panel: &[Vec<PanelRow>],
    cap: &HashMap<String, f64>,
    sector: &HashMap<String, String>,
    do_sector: bool,
) -> Vec<Vec<PanelRow>> {
    let dim = factor_dim();
    let mut out = Vec::with_capacity(panel.len());
    for day in panel {
        let n = day.len();
        if n < MIN_CROSS {
            out.push(day.clone());
            continue;
        }
        // 逐日 z-score 的 log cap
        let logcap: Vec<f64> = day
            .iter()
            .map(|r| cap.get(&r.symbol).copied().unwrap_or(0.0).max(1.0).ln())
            .collect();
        let z_logcap = zscore(&logcap);

        let mut new_day = day.clone();
        for d in 0..dim {
            let mut col: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();

            // 1) 粗行业组去均值（仅对 ≥4 只的 sector，避免小组把信号抹平）
            if do_sector {
                group_demean(&mut col, day, sector, 4);
            }

            // 2) 对 z_logcap 做单回归取残差（size 中性化）
            let beta = {
                let mc = mean(&col);
                let mut cov = 0.0;
                let mut vz = 0.0;
                for (c, z) in col.iter().zip(&z_logcap) {
                    cov += (c - mc) * z; // z 已零均值
                    vz += z * z;
                }
                if vz > 1e-9 { cov / vz } else { 0.0 }
            };
            for (c, z) in col.iter_mut().zip(&z_logcap) {
                *c -= beta * z;
            }

            // 3) 残差重新 z-score 写回
            let zc = zscore(&col);
            for (r, v) in new_day.iter_mut().zip(zc) {
                r.factors[d] = v;
            }
        }
        out.push(new_day);
    }
    out
}

fn group_demean(
    col: &mut [f64],
    day: &[PanelRow],
    sector: &HashMap<String, String>,
    min_group: usize,
) {
    let mut sums: HashMap<&str, (f64, usize)> = HashMap::new();
    let secs: Vec<&str> = day
        .iter()
        .map(|r| sector.get(&r.symbol).map(|s| s.as_str()).unwrap_or("其他"))
        .collect();
    for (v, s) in col.iter().zip(&secs) {
        let e = sums.entry(s).or_insert((0.0, 0));
        e.0 += *v;
        e.1 += 1;
    }
    for (v, s) in col.iter_mut().zip(&secs) {
        if let Some(&(sum, cnt)) = sums.get(s) {
            if cnt >= min_group {
                *v -= sum / cnt as f64;
            }
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn zscore(xs: &[f64]) -> Vec<f64> {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len().max(1) as f64;
    let sd = var.sqrt();
    if sd > 1e-9 {
        xs.iter().map(|x| (x - m) / sd).collect()
    } else {
        vec![0.0; xs.len()]
    }
}

/// 110 个细分行业 → ~11 个粗 sector（关键词匹配，fallback=其他）。
fn macro_sector(ind: &str) -> &'static str {
    const RULES: &[(&str, &[&str])] = &[
        ("金融", &["银行", "证券", "保险", "金融", "信托"]),
        ("地产", &["地产", "房", "园区"]),
        (
            "医药",
            &["制药", "生物", "医药", "医疗", "原料药", "中药", "中成药"],
        ),
        (
            "科技",
            &[
                "元器件", "半导体", "软件", "通信", "电子", "互联网", "IT", "计算机", "光学",
                "消费电子",
            ],
        ),
        (
            "消费",
            &[
                "白酒", "食品", "家用电器", "百货", "酒", "饮料", "乳", "服饰", "旅游", "纺织",
                "零售", "酒店", "家居",
            ],
        ),
        ("汽车", &["汽车"]),
        (
            "工业",
            &[
                "机械", "工程", "电气设备", "仪器", "仪表", "输配电", "装修", "装饰", "建筑",
                "重型", "专用",
            ],
        ),
        (
            "材料",
            &[
                "化工", "化纤", "塑料", "钢铁", "有色", "水泥", "玻璃", "化肥", "农药", "建材",
                "造纸",
            ],
        ),
        ("能源", &["煤炭", "发电", "石油", "电力", "燃气", "新能源"]),
        (
            "交运",
            &["港口", "机场", "航空", "铁路", "公路", "航运", "物流", "运输"],
        ),
        ("农业", &["农", "牧", "渔", "种业", "养殖"]),
    ];
    for (sec, kws) in RULES {
        if kws.iter().any(|k| ind.contains(k)) {
            return sec;
        }
    }
    "其他"
}
