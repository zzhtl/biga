//! 基本面因子的「板块中性化」复测——回答一个诚实问题：
//! EP/ROE 在大盘龙头池上 t≈2-3 的显著性，是真价值/质量 alpha，还是
//! 「便宜的能源/矿业 vs 贵的科技」这种**板块押注**的伪装？
//!
//! 方法：对每个交易日的截面，**按板块(category)对因子值和未来收益各自去均值**，
//! 再用残差算 Pearson IC。若信号来自板块间差异，中性化后 IC/t 会塌向 0；
//! 若板块内排序仍预测收益，则中性化后依旧显著——那才是可泛化的 alpha。
//!
//! 同时打印 原始 vs 板块中性 两列对照，跨 h=5/15。
//! 用法：cargo run --release --example fundamental_neutral

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::{HistoricalData, StockFundamental};
use biga_lib::db::repository::{
    get_recent_historical_data_for_symbols, get_stock_fundamentals, get_symbols_with_min_bars,
};
use chrono::{Duration, NaiveDate};
use sqlx::Row;
use std::collections::{BTreeMap, HashMap};

const MIN_CROSS: usize = 8;
const REPORT_LAG_DAYS: i64 = 60;

type FactorFn = &'static (dyn Fn(&StockFundamental, f64) -> Option<f64> + Sync);

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    let symbols = get_symbols_with_min_bars(300, &pool).await.expect("获取股票列表失败");
    let stocks: Vec<(String, Vec<HistoricalData>)> =
        get_recent_historical_data_for_symbols(&symbols, 400, &pool)
            .await
            .expect("获取历史失败")
            .into_iter()
            .filter(|(_, h)| h.len() >= 200)
            .collect();

    // 板块分类（中性化分组依据）
    let sectors: HashMap<String, String> = sqlx::query(
        "SELECT symbol, category FROM stock WHERE category IS NOT NULL AND category != ''",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.get::<String, _>("symbol"), r.get::<String, _>("category")))
    .collect();
    let n_sectors = sectors.values().collect::<std::collections::HashSet<_>>().len();

    // 基本面（点对点序列）
    let mut funds: HashMap<String, Vec<(NaiveDate, StockFundamental)>> = HashMap::new();
    for (sym, _) in &stocks {
        let rows = get_stock_fundamentals(sym, &pool).await.unwrap_or_default();
        let mut parsed: Vec<(NaiveDate, StockFundamental)> = rows
            .into_iter()
            .filter_map(|f| {
                NaiveDate::parse_from_str(&f.report_date, "%Y-%m-%d").ok().map(|d| (d, f))
            })
            .collect();
        parsed.sort_by_key(|(d, _)| *d);
        if !parsed.is_empty() {
            funds.insert(sym.clone(), parsed);
        }
    }
    println!(
        "可用票 {}（有基本面 {}），覆盖 {} 个板块。点对点滞后 {} 天。",
        stocks.len(),
        funds.len(),
        n_sectors,
        REPORT_LAG_DAYS
    );

    let factors: &[(&str, FactorFn)] = &[
        ("BP(净资产/价)", &|f, px| f.bps.map(|b| b / px)),
        ("EP(年化EPS/价)", &|f, px| annualized_eps(f).map(|e| e / px)),
        ("ROE", &|f, _| f.roe),
        ("营收增长", &|f, _| f.revenue_growth),
    ];

    for &horizon in &[5usize, 15] {
        println!(
            "\n===== horizon={horizon} 日 =====\n{:<14} {:<10} {:>9} {:>10} {:>11} {:>7}",
            "因子", "口径", "IC均值", "t(非重叠)", "多空/期(%)", "覆盖天"
        );
        println!("{}", "-".repeat(66));
        for (fname, ffn) in factors {
            for (label, neutral) in [("原始", false), ("板块中性", true)] {
                let r = eval_factor(&stocks, &funds, &sectors, *ffn, horizon, neutral);
                if r.days < 10 {
                    continue;
                }
                println!(
                    "{:<14} {:<10} {:>+9.4} {:>+10.2} {:>+11.3} {:>7}",
                    fname, label, r.ic_mean, r.t_nonoverlap, r.ls_per_period * 100.0, r.days
                );
            }
        }
    }
    println!(
        "\n判读：原始显著、板块中性后 t 塌向 0 → 信号是板块押注（不可泛化）；\n\
         两列都 |t|≥2 且同号 → 板块内仍有 alpha，才值得接生产。"
    );
}

struct FactorReport {
    ic_mean: f64,
    t_nonoverlap: f64,
    ls_per_period: f64,
    days: usize,
}

fn eval_factor(
    stocks: &[(String, Vec<HistoricalData>)],
    funds: &HashMap<String, Vec<(NaiveDate, StockFundamental)>>,
    sectors: &HashMap<String, String>,
    ffn: FactorFn,
    horizon: usize,
    neutralize: bool,
) -> FactorReport {
    // 逐日截面：(因子值, 未来收益, 板块)
    let mut by_date: BTreeMap<NaiveDate, Vec<(f64, f64, String)>> = BTreeMap::new();
    for (sym, hist) in stocks {
        let Some(freports) = funds.get(sym) else { continue };
        let sector = sectors.get(sym).cloned().unwrap_or_default();
        let n = hist.len();
        if n <= horizon {
            continue;
        }
        for i in 0..n - horizon {
            let date = hist[i].date;
            let px = hist[i].close;
            if px <= 0.0 {
                continue;
            }
            let Some(f) = point_in_time(freports, date) else { continue };
            let Some(val) = ffn(f, px) else { continue };
            if !val.is_finite() {
                continue;
            }
            let fwd = (hist[i + horizon].close - px) / px;
            if !fwd.is_finite() {
                continue;
            }
            by_date.entry(date).or_default().push((val, fwd, sector.clone()));
        }
    }

    let mut ic_series: Vec<f64> = Vec::new();
    let mut ls_series: Vec<f64> = Vec::new();
    for (_d, rows) in &by_date {
        if rows.len() < MIN_CROSS {
            continue;
        }
        // 取因子值与未来收益；中性化则按板块各自去均值（剔除当日单成分板块）。
        let (xs, ys): (Vec<f64>, Vec<f64>) = if neutralize {
            sector_demean(rows)
        } else {
            (rows.iter().map(|r| r.0).collect(), rows.iter().map(|r| r.1).collect())
        };
        if xs.len() < MIN_CROSS {
            continue;
        }
        ic_series.push(pearson(&xs, &ys));

        let mut idx: Vec<usize> = (0..xs.len()).collect();
        idx.sort_by(|&a, &b| xs[b].total_cmp(&xs[a]));
        let k = (xs.len() as f64 * 0.2).ceil() as usize;
        if k >= 1 {
            let top = idx[..k].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            let bot = idx[xs.len() - k..].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            ls_series.push(top - bot);
        }
    }

    let sampled: Vec<f64> = ic_series.iter().step_by(horizon).copied().collect();
    FactorReport {
        ic_mean: mean(&ic_series),
        t_nonoverlap: t_stat(&sampled),
        ls_per_period: mean(&ls_series),
        days: ic_series.len(),
    }
}

/// 按板块分别对 (因子值, 未来收益) 去均值，返回残差对。剔除当日仅 1 个成分股的板块。
fn sector_demean(rows: &[(f64, f64, String)]) -> (Vec<f64>, Vec<f64>) {
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, (_, _, s)) in rows.iter().enumerate() {
        groups.entry(s.as_str()).or_default().push(i);
    }
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (_s, idx) in &groups {
        if idx.len() < 2 {
            continue;
        }
        let mx = idx.iter().map(|&i| rows[i].0).sum::<f64>() / idx.len() as f64;
        let my = idx.iter().map(|&i| rows[i].1).sum::<f64>() / idx.len() as f64;
        for &i in idx {
            xs.push(rows[i].0 - mx);
            ys.push(rows[i].1 - my);
        }
    }
    (xs, ys)
}

fn point_in_time<'a>(
    reports: &'a [(NaiveDate, StockFundamental)],
    date: NaiveDate,
) -> Option<&'a StockFundamental> {
    let cutoff = date - Duration::days(REPORT_LAG_DAYS);
    reports.iter().rev().find(|(rd, _)| *rd <= cutoff).map(|(_, f)| f)
}

fn annualized_eps(f: &StockFundamental) -> Option<f64> {
    let eps = f.eps?;
    let month = f.report_date.get(5..7)?.parse::<u32>().ok()?;
    let factor = match month {
        3 => 4.0,
        6 => 2.0,
        9 => 4.0 / 3.0,
        12 => 1.0,
        _ => return None,
    };
    Some(eps * factor)
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = mean(xs);
    let my = mean(ys);
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

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn t_stat(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    if var <= 0.0 {
        return 0.0;
    }
    m / (var.sqrt() / (n as f64).sqrt())
}
