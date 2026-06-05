//! 基本面因子的截面 IC 验证（非技术 alpha 的首次严格检验）。
//!
//! 因子（点对点，报告期+60天滞后避免前视）：
//!   BP=每股净资产/价（价值）、EP=年化EPS/价（价值）、ROE（质量）、利润增长、营收增长。
//! 方法：每个交易日做因子截面 z-score → IC=corr(因子, 未来horizon收益)；
//! 跨 full/≥200亿/<200亿 三池，报告 IC 均值 + 非重叠样本 t + 20%多空。
//!
//! ⚠️ 数据局限：cwzb 每股仅约 4 个最近季度 → 点对点窗口仅约 10 个月，独立样本少、统计功效弱。
//! 这是"起步+随时间积累"的首测，不是定论。用法：cargo run --release --example fundamental_factors
//!
//! 需先 `cargo run --release --example fetch_fundamentals` 把基本面入库。

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::{HistoricalData, StockFundamental};
use biga_lib::db::repository::{
    get_recent_historical_data_for_symbols, get_stock_fundamentals, get_symbols_with_min_bars,
};
use chrono::{Duration, NaiveDate};
use sqlx::Row;
use std::collections::{BTreeMap, HashMap};

const MIN_CROSS: usize = 8;
const REPORT_LAG_DAYS: i64 = 60; // 报告期后约2个月才公开可用，保守滞后避免前视

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

    // 市值（分池）
    let cap: HashMap<String, f64> = sqlx::query(
        "SELECT symbol, circulating_market_cap c FROM stock_capital WHERE circulating_market_cap > 0",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| (r.get::<String, _>("symbol"), r.get::<f64, _>("c")))
    .collect();

    // 基本面（点对点序列）
    let mut funds: HashMap<String, Vec<(NaiveDate, StockFundamental)>> = HashMap::new();
    let mut n_with_fund = 0;
    for (sym, _) in &stocks {
        let rows = get_stock_fundamentals(sym, &pool).await.unwrap_or_default();
        if rows.is_empty() {
            continue;
        }
        let mut parsed: Vec<(NaiveDate, StockFundamental)> = rows
            .into_iter()
            .filter_map(|f| {
                NaiveDate::parse_from_str(&f.report_date, "%Y-%m-%d")
                    .ok()
                    .map(|d| (d, f))
            })
            .collect();
        parsed.sort_by_key(|(d, _)| *d);
        if !parsed.is_empty() {
            n_with_fund += 1;
            funds.insert(sym.clone(), parsed);
        }
    }
    println!(
        "可用票 {}，其中有基本面 {} 只。点对点滞后 {} 天。",
        stocks.len(),
        n_with_fund,
        REPORT_LAG_DAYS
    );

    let factors: &[(&str, FactorFn)] = &[
        ("BP(净资产/价)", &|f, px| f.bps.map(|b| b / px)),
        ("EP(年化EPS/价)", &|f, px| annualized_eps(f).map(|e| e / px)),
        ("ROE", &|f, _| f.roe),
        ("利润增长", &|f, _| f.profit_growth),
        ("营收增长", &|f, _| f.revenue_growth),
    ];

    let pools: Vec<(&str, Box<dyn Fn(&str) -> bool>)> = vec![
        ("full", Box::new(|_: &str| true)),
        ("cap>=200亿", {
            let cap = cap.clone();
            Box::new(move |s: &str| cap.get(s).copied().unwrap_or(0.0) >= 200.0e8)
        }),
        ("cap<200亿", {
            let cap = cap.clone();
            Box::new(move |s: &str| cap.get(s).copied().unwrap_or(0.0) < 200.0e8)
        }),
    ];

    for &horizon in &[5usize, 15] {
        println!(
            "\n===== horizon={horizon} 日 =====\n{:<14} {:<10} {:>9} {:>9} {:>11} {:>7} {:>6}",
            "因子", "pool", "IC均值", "t(非重叠)", "多空/期(%)", "覆盖天", "样本"
        );
        println!("{}", "-".repeat(74));
        for (fname, ffn) in factors {
            for (pname, in_pool) in &pools {
                let r = eval_factor(&stocks, &funds, in_pool, *ffn, horizon);
                if r.days < 10 {
                    continue;
                }
                println!(
                    "{:<14} {:<10} {:>+9.4} {:>+9.2} {:>+11.3} {:>7} {:>6}",
                    fname, pname, r.ic_mean, r.t_nonoverlap, r.ls_per_period * 100.0, r.days, r.obs
                );
            }
        }
    }
    println!("\n判读：|t|≥2 才算显著；多池同号且 t 稳健才可信。样本期短→谨慎，重在随时间积累后复测。");
}

type FactorFn = &'static (dyn Fn(&StockFundamental, f64) -> Option<f64> + Sync);

/// YTD EPS 年化（按报告期季度数）
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

struct FactorReport {
    ic_mean: f64,
    t_nonoverlap: f64,
    ls_per_period: f64,
    days: usize,
    obs: usize,
}

fn eval_factor(
    stocks: &[(String, Vec<HistoricalData>)],
    funds: &HashMap<String, Vec<(NaiveDate, StockFundamental)>>,
    in_pool: &dyn Fn(&str) -> bool,
    ffn: FactorFn,
    horizon: usize,
) -> FactorReport {
    // 按交易日聚合截面：(因子原值, 未来收益)
    let mut by_date: BTreeMap<NaiveDate, Vec<(f64, f64)>> = BTreeMap::new();
    let mut obs = 0usize;

    for (sym, hist) in stocks {
        if !in_pool(sym) {
            continue;
        }
        let Some(freports) = funds.get(sym) else {
            continue;
        };
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
            // 点对点：最近一个 report_date+滞后 ≤ date 的报告
            let Some(f) = point_in_time(freports, date) else {
                continue;
            };
            let Some(val) = ffn(f, px) else {
                continue;
            };
            if !val.is_finite() {
                continue;
            }
            let fwd = (hist[i + horizon].close - px) / px;
            if !fwd.is_finite() {
                continue;
            }
            by_date.entry(date).or_default().push((val, fwd));
            obs += 1;
        }
    }

    // 逐日 IC 与多空
    let mut ic_series: Vec<f64> = Vec::new();
    let mut ls_series: Vec<f64> = Vec::new();
    for (_d, rows) in &by_date {
        if rows.len() < MIN_CROSS {
            continue;
        }
        let xs: Vec<f64> = rows.iter().map(|(v, _)| *v).collect();
        let ys: Vec<f64> = rows.iter().map(|(_, y)| *y).collect();
        ic_series.push(pearson(&xs, &ys));

        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| xs[b].total_cmp(&xs[a])); // 因子降序
        let k = (rows.len() as f64 * 0.2).ceil() as usize;
        if k >= 1 {
            let top = idx[..k].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            let bot = idx[rows.len() - k..].iter().map(|&j| ys[j]).sum::<f64>() / k as f64;
            ls_series.push(top - bot);
        }
    }

    let ic_mean = mean(&ic_series);
    // 非重叠抽样算 t（每 horizon 取一点）
    let sampled: Vec<f64> = ic_series.iter().step_by(horizon).copied().collect();
    let t = t_stat(&sampled);
    let ls = mean(&ls_series);

    FactorReport {
        ic_mean,
        t_nonoverlap: t,
        ls_per_period: ls,
        days: ic_series.len(),
        obs,
    }
}

fn point_in_time<'a>(
    reports: &'a [(NaiveDate, StockFundamental)],
    date: NaiveDate,
) -> Option<&'a StockFundamental> {
    let cutoff = date - Duration::days(REPORT_LAG_DAYS);
    reports
        .iter()
        .rev()
        .find(|(rd, _)| *rd <= cutoff)
        .map(|(_, f)| f)
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
