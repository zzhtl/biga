//! 调优测量：对多只股票做走步回测，输出聚合方向准确率 / 朴素基准 / 超额(edge)。
//! 用法：cargo run --example tune_backtest

use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::get_recent_historical_data;
use biga_lib::prediction::backtest::run_backtest;
use sqlx::Row;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // 取历史足够长的股票
    let rows = sqlx::query(
        "SELECT symbol, COUNT(*) c FROM historical_data GROUP BY symbol HAVING c >= 200 ORDER BY c DESC LIMIT 25",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");

    println!("参与回测股票数：{}", rows.len());

    let mut data = Vec::new();
    for r in &rows {
        let symbol: String = r.get("symbol");
        let hist = get_recent_historical_data(&symbol, 300, &pool)
            .await
            .unwrap_or_default();
        if hist.len() >= 120 {
            data.push((symbol, hist));
        }
    }

    for horizon in [1usize, 5] {
        let mut acc = 0.0;
        let mut base = 0.0;
        let mut edge = 0.0;
        let mut pred_up = 0.0;
        let mut n = 0;
        for (symbol, hist) in &data {
            if let Ok(rep) = run_backtest(symbol, hist, 60, horizon, 3) {
                let m = &rep.metrics;
                acc += m.direction_accuracy;
                base += m.baseline_accuracy;
                edge += m.edge();
                pred_up += m.predicted_up_ratio;
                n += 1;
            }
        }
        if n > 0 {
            let n = n as f64;
            println!(
                "horizon={horizon}: 引擎={:.1}% 基准={:.1}% 超额={:+.2}% 预测涨占比={:.0}%（{} 只）",
                acc / n * 100.0,
                base / n * 100.0,
                edge / n * 100.0,
                pred_up / n * 100.0,
                n as usize
            );
        }
    }
}
