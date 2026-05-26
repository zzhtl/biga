//! 调优测量脚本：在本地数据库上对若干股票做真实走步回测，输出方向准确率等指标。
//! 用法：cargo run --example tune_backtest

use biga_lib::api::stock::fetch_stock_capital;
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::StockCapital;
use biga_lib::db::repository::{
    backfill_volume_metrics, get_latest_close_price, get_recent_historical_data,
    upsert_stock_capital,
};
use biga_lib::prediction::backtest::run_backtest;
use biga_lib::prediction::model::features::build_dataset;
use biga_lib::prediction::model::network::train_and_save;
use sqlx::Row;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // 应用迁移 03（建表 + 量比列），与 lib.rs 一致
    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_capital (symbol TEXT PRIMARY KEY, circulating_shares REAL NOT NULL DEFAULT 0, total_shares REAL NOT NULL DEFAULT 0, circulating_market_cap REAL NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "ALTER TABLE historical_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
        "ALTER TABLE realtime_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await; // 忽略 duplicate column
    }

    // 取数据最多的若干股票
    let rows = sqlx::query(
        "SELECT symbol, COUNT(*) c FROM historical_data GROUP BY symbol HAVING c >= 150 ORDER BY c DESC LIMIT 5",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");

    if rows.is_empty() {
        println!("数据库中没有足够历史数据的股票");
        return;
    }

    let step = 3usize;

    // 回填股本与换手率（拉流通市值 → 推导流通股本 → 回填 turnover_rate），让换手率特征真正生效
    println!("=== 回填股本/换手率 ===");
    for r in &rows {
        let symbol: String = r.get("symbol");
        if let Ok(q) = fetch_stock_capital(&symbol).await {
            if let Ok(Some(close)) = get_latest_close_price(&symbol, &pool).await {
                if close > 0.0 && q.circulating_market_cap > 0.0 {
                    let cap = StockCapital {
                        symbol: symbol.clone(),
                        circulating_shares: q.circulating_market_cap / close,
                        total_shares: q.total_market_cap / close,
                        circulating_market_cap: q.circulating_market_cap,
                    };
                    let _ = upsert_stock_capital(&pool, &cap).await;
                }
            }
        }
        let _ = backfill_volume_metrics(&symbol, &pool).await;
    }

    // 预取数据（此时已带真实换手率）
    let mut data = Vec::new();
    for r in &rows {
        let symbol: String = r.get("symbol");
        let hist = get_recent_historical_data(&symbol, 500, &pool)
            .await
            .unwrap_or_default();
        let avg_turnover = if hist.is_empty() {
            0.0
        } else {
            hist.iter().map(|h| h.turnover_rate).sum::<f64>() / hist.len() as f64
        };
        println!("{symbol:<10} 平均换手率={avg_turnover:.2}%");
        data.push((symbol, hist));
    }

    for horizon in [1usize, 3, 5] {
        let mut acc_sum = 0.0;
        let mut n_stocks = 0;
        println!("\n=== 走步回测（horizon={horizon}, step={step}）===");
        for (symbol, hist) in &data {
            match run_backtest(symbol, hist, 60, horizon, step) {
                Ok(rep) => {
                    let m = &rep.metrics;
                    println!(
                        "{symbol:<10} 引擎准确率={:>5.1}% 朴素基准={:>5.1}% 超额={:>+5.1}% | 预测涨占比={:>4.0}% 实际涨={:>4.0}%",
                        m.direction_accuracy * 100.0,
                        m.baseline_accuracy * 100.0,
                        m.edge() * 100.0,
                        m.predicted_up_ratio * 100.0,
                        m.actual_up_ratio * 100.0
                    );
                    acc_sum += m.direction_accuracy;
                    n_stocks += 1;
                }
                Err(e) => println!("{symbol}: 回测失败 {e}"),
            }
        }
        if n_stocks > 0 {
            println!(
                ">>> horizon={horizon} 平均方向准确率 = {:.2}%",
                acc_sum / n_stocks as f64 * 100.0
            );
        }
    }

    // ML 模型：次日方向准确率（测试集），特征含量比/换手率
    println!("\n=== ML 模型（candle MLP）next-day 方向准确率（测试集）===");
    let mut ml_acc = 0.0;
    let mut nm = 0;
    for (symbol, hist) in &data {
        let (f, l, n) = build_dataset(hist);
        if n < 40 {
            continue;
        }
        let path = std::env::temp_dir().join(format!("tune_{symbol}.safetensors"));
        // 训练有随机初始化，取 3 次平均以控制方差
        let runs = 3;
        let mut acc = 0.0;
        let mut mae = 0.0;
        let mut ok = 0;
        for _ in 0..runs {
            if let Ok(o) = train_and_save(&f, &l, n, 400, 0.01, 0.8, &path) {
                acc += o.direction_accuracy;
                mae += o.mae;
                ok += 1;
            }
        }
        if ok > 0 {
            let acc = acc / ok as f64;
            println!(
                "{symbol:<10} 方向准确率(3次均)={:>5.1}% MAE={:>5.2}",
                acc * 100.0,
                mae / ok as f64
            );
            ml_acc += acc;
            nm += 1;
        }
        std::fs::remove_file(&path).ok();
    }
    if nm > 0 {
        println!(">>> ML 平均方向准确率 = {:.2}%", ml_acc / nm as f64 * 100.0);
    }
}
