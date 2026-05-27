//! 专业量化评测：滚动截面多因子（含 量比×换手率 组合因子）。
//! 输出单因子 IC、前向滚动 Rank IC / 多空价差，并演示最新排名。
//! 用法：cargo run --release --example cross_sectional

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data;
use biga_lib::prediction::cross_section::{build_panel, pearson, rank_latest, walk_forward};
use biga_lib::prediction::factor::{factor_dim, factor_names};
use sqlx::Row;

const HORIZON: usize = 5;
const WINDOW: usize = 120;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");
    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_capital (symbol TEXT PRIMARY KEY, circulating_shares REAL NOT NULL DEFAULT 0, total_shares REAL NOT NULL DEFAULT 0, circulating_market_cap REAL NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "ALTER TABLE historical_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
        "ALTER TABLE realtime_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await;
    }

    let rows = sqlx::query(
        "SELECT symbol, COUNT(*) c FROM historical_data GROUP BY symbol HAVING c >= 200 ORDER BY c DESC",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");
    println!("参与截面的股票数：{}", rows.len());

    let mut stocks: Vec<(String, Vec<HistoricalData>)> = Vec::new();
    for r in &rows {
        let symbol: String = r.get("symbol");
        let hist = get_recent_historical_data(&symbol, 800, &pool)
            .await
            .unwrap_or_default();
        if hist.len() >= 120 {
            stocks.push((symbol, hist));
        }
    }

    let panel = build_panel(&stocks, HORIZON);
    println!("截面交易日数：{}", panel.len());

    // 单因子 IC（全样本日均）
    let dim = factor_dim();
    let names = factor_names();
    let mut fic = vec![0.0; dim];
    let mut cnt = 0usize;
    for day in &panel {
        let ys: Vec<f64> = day.iter().map(|r| r.fwd_return).collect();
        for d in 0..dim {
            let xs: Vec<f64> = day.iter().map(|r| r.factors[d]).collect();
            fic[d] += pearson(&xs, &ys);
        }
        cnt += 1;
    }
    let mut ranked: Vec<(String, f64)> = (0..dim)
        .map(|d| (names[d].clone(), fic[d] / cnt.max(1) as f64))
        .collect();
    ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    println!("\n----- 单因子 IC（全样本日均，{HORIZON}日相对收益）-----");
    for (name, ic) in &ranked {
        println!("  {name:<18} IC = {ic:+.4}");
    }

    // 前向滚动多因子
    let rep = walk_forward(&panel, WINDOW);
    println!("\n========= 前向滚动多因子（窗口{WINDOW}日，{}个样本外日）=========", rep.oos_days);
    println!(">>> 前向 Rank IC = {:+.4}", rep.rank_ic);
    println!(">>> 相对方向准确率 = {:.2}%（基准50%）", rep.direction_accuracy * 100.0);
    println!(">>> 多空价差（top20%-bot20%）= {:+.2}%", rep.long_short_spread * 100.0);

    // 最新排名演示
    let ranking = rank_latest(&stocks, HORIZON, WINDOW);
    if !ranking.is_empty() {
        println!("\n----- 最新交易日相对强弱排名（前5/后5）-----");
        for s in ranking.iter().take(5) {
            println!("  #{:<3} {:<10} score={:+.3}", s.rank, s.symbol, s.score);
        }
        println!("  ...");
        for s in ranking.iter().rev().take(5).rev() {
            println!("  #{:<3} {:<10} score={:+.3}", s.rank, s.symbol, s.score);
        }
    }
}
