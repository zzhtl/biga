//! 调优测量：对多只股票做走步回测，输出聚合方向准确率 / 朴素基准 / 超额(edge)。
//! 用法：cargo run --example tune_backtest

use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::{get_recent_historical_data_for_symbols, get_symbols_with_min_bars};
use biga_lib::prediction::backtest::run_backtest;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    let symbols = get_symbols_with_min_bars(200, &pool)
        .await
        .expect("查询股票失败");

    let data = get_recent_historical_data_for_symbols(&symbols, 3000, &pool)
        .await
        .expect("查询历史数据失败")
        .into_iter()
        .filter(|(_, hist)| hist.len() >= 120)
        .collect::<Vec<_>>();
    let eval_symbols = data
        .iter()
        .take(25)
        .map(|(symbol, _)| symbol.as_str())
        .collect::<Vec<_>>();

    println!("参与截面股票数：{}，参与端到端回测股票数：{}", data.len(), eval_symbols.len());

    for horizon in [1usize, 5] {
        let mut acc = 0.0;
        let mut base = 0.0;
        let mut edge = 0.0;
        let mut pred_up = 0.0;
        let mut n = 0;
        for (symbol, hist) in data
            .iter()
            .filter(|(symbol, _)| eval_symbols.iter().any(|eval| *eval == symbol))
        {
            if let Ok(rep) = run_backtest(symbol, hist, 60, horizon, 10) {
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
