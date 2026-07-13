//! 区间预测校准实验：方向不可测但波动可测——验证用 GARCH / 已实现波动率构造的
//! H 日涨跌区间带是否"校准"（名义 80% 带是否真覆盖 ~80% 的实际结果），
//! 并求出校准用的 z 倍数与平均带宽。确认能校准后再把区间接进生产预测输出。
//!
//! 用法：cargo run --release --example interval_calibration

use biga_lib::db::connection::create_pool;
use biga_lib::db::models::HistoricalData;
use biga_lib::db::repository::get_recent_historical_data_for_symbols;
use biga_lib::prediction::analysis::prediction_interval;
use biga_lib::prediction::analysis::volatility_forecast::{
    calculate_realized_volatility, GarchForecaster,
};
use sqlx::Row;

const MIN_LOOKBACK: usize = 80; // GARCH 估参需要足够历史

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    let symbols: Vec<String> = sqlx::query(
        "SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 400 ORDER BY symbol LIMIT 40",
    )
    .fetch_all(&pool)
    .await
    .unwrap()
    .into_iter()
    .map(|r| r.get::<String, _>("symbol"))
    .collect();

    let stocks: Vec<(String, Vec<HistoricalData>)> =
        get_recent_historical_data_for_symbols(&symbols, 900, &pool)
            .await
            .unwrap()
            .into_iter()
            .filter(|(_, h)| h.len() >= 400)
            .collect();
    println!("参与校准的股票数：{}", stocks.len());

    // 正态分位（参照：若残差服从正态，名义 z 应给出对应覆盖率）
    let nominal = [(1.0_f64, 68.3), (1.2816, 80.0), (1.6449, 90.0), (1.9600, 95.0)];
    let production_80_z =
        prediction_interval::calibrated_z(prediction_interval::DEFAULT_COVERAGE);
    let production_95_z = prediction_interval::calibrated_z(0.95);
    let mut production_passed = true;
    let mut production_checks = 0usize;

    for &horizon in &[1usize, 5, 10] {
        for method in ["garch", "realized20"] {
            // 收集标准化残差 |actual / sigma_H| 与带宽
            let mut resid: Vec<f64> = Vec::new();
            let mut halfwidth_pct: Vec<f64> = Vec::new(); // 名义80%带半宽(%)，= 1.2816*sigma_H*100

            for (_, hist) in &stocks {
                let closes: Vec<f64> = hist.iter().map(|h| h.close).collect();
                let n = closes.len();
                // 非重叠抽样：每 horizon 取一个发起日
                let mut base_idx = MIN_LOOKBACK;
                while base_idx + horizon < n {
                    let base = closes[base_idx];
                    let future = closes[base_idx + horizon];
                    if base > 0.0 && future > 0.0 {
                        let actual = (future - base) / base; // 收益（小数）
                        let sigma_h = match method {
                            "garch" => {
                                let f = GarchForecaster::from_prices(&closes[..=base_idx]);
                                let daily = f.forecast(horizon).forecast; // 逐日 sigma
                                (daily.iter().map(|s| s * s).sum::<f64>()).sqrt()
                            }
                            _ => {
                                // 近20日已实现日波动 × sqrt(H)
                                let start = (base_idx + 1).saturating_sub(20);
                                let sd = calculate_realized_volatility(&closes[start..=base_idx]);
                                sd * (horizon as f64).sqrt()
                            }
                        };
                        if sigma_h > 1e-6 && actual.is_finite() {
                            resid.push((actual / sigma_h).abs());
                            halfwidth_pct.push(1.2816 * sigma_h * 100.0);
                        }
                    }
                    base_idx += horizon.max(1);
                }
            }

            if resid.len() < 50 {
                continue;
            }
            let mut sorted = resid.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let n = sorted.len() as f64;
            let cov_at = |z: f64| sorted.iter().filter(|&&r| r <= z).count() as f64 / n * 100.0;
            let quantile = |p: f64| {
                let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };
            let avg_hw = halfwidth_pct.iter().sum::<f64>() / halfwidth_pct.len() as f64;

            println!(
                "\n[h={horizon} 方法={method}] 样本={} 名义80%带平均半宽=±{:.2}%",
                sorted.len(),
                avg_hw
            );
            println!("  名义z覆盖率(残差服从正态时应=右列):");
            for (z, want) in nominal {
                println!(
                    "    z={:.3} → 经验覆盖 {:>5.1}%  (正态名义 {:>4.1}%)",
                    z,
                    cov_at(z),
                    want
                );
            }
            println!("  校准用经验分位 z（用这个倍数构带才真覆盖目标）:");
            println!(
                "    80%→z={:.3}  90%→z={:.3}  95%→z={:.3}",
                quantile(80.0),
                quantile(90.0),
                quantile(95.0)
            );
            if method == "realized20" {
                production_checks += 1;
                let production_80 = cov_at(production_80_z);
                let production_95 = cov_at(production_95_z);
                let pass_80 = (75.0..=85.0).contains(&production_80);
                let pass_95 = (90.0..=98.0).contains(&production_95);
                production_passed &= pass_80 && pass_95;
                println!(
                    "  生产口径：80% z={production_80_z:.2} → {:.1}% [{}]；95% z={production_95_z:.2} → {:.1}% [{}]",
                    production_80,
                    if pass_80 { "PASS" } else { "FAIL" },
                    production_95,
                    if pass_95 { "PASS" } else { "FAIL" },
                );
            } else {
                println!("  研究对照：此方法不用于生产区间输出");
            }
        }
    }
    assert_eq!(production_checks, 3, "生产区间校准样本不足");
    assert!(production_passed, "生产区间经验覆盖率未达到校准目标");
    println!(
        "\n判读：若'名义z覆盖'低于右列正态值→真实分布更厚尾，需用更大 z（见'经验分位z'）。\n带宽=区间是否有用的尺度（如5日±X%）。"
    );
}
