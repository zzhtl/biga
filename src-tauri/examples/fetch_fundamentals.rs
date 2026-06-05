//! 批量拉取基本面财务指标（zhitu hs/gs/cwzb）入库 stock_fundamentals。
//! 非技术数据——给截面因子提供估值(EP/BP)、质量(ROE)、成长(增长率)维度。
//! 连续多次 API 失败提前停止（疑似当日额度耗尽）。
//!
//! 用法：
//!   cargo run --release --example fetch_fundamentals                 # 拉历史≥300根的全部票
//!   FUND_MAX=10 cargo run --release --example fetch_fundamentals     # 先拉 10 只验证
//!   FUND_MIN_BARS=150 cargo run --release --example fetch_fundamentals

use biga_lib::api::stock::fetch_financial_indicators;
use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::{get_symbols_with_min_bars, upsert_stock_fundamental};

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // examples 不走 Tauri 启动迁移，自行确保表存在（与迁移 04 一致）。
    let _ = sqlx::query(
        "CREATE TABLE IF NOT EXISTS stock_fundamentals (\
            symbol TEXT NOT NULL, report_date TEXT NOT NULL, eps REAL, bps REAL, roe REAL, \
            profit_growth REAL, revenue_growth REAL, debt_ratio REAL, \
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, \
            PRIMARY KEY (symbol, report_date))",
    )
    .execute(&pool)
    .await;

    let min_bars = std::env::var("FUND_MIN_BARS")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(300);
    let symbols = get_symbols_with_min_bars(min_bars, &pool)
        .await
        .expect("获取股票列表失败");
    let max = std::env::var("FUND_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(symbols.len())
        .min(symbols.len());
    println!(
        "待拉取基本面 {} 只（历史≥{min_bars}根共 {} 只，上限 {max}）",
        max,
        symbols.len()
    );

    let (mut ok, mut empty, mut failed, mut rows_total, mut consec_fail) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for (idx, symbol) in symbols.iter().take(max).enumerate() {
        match fetch_financial_indicators(symbol).await {
            Ok(reports) if !reports.is_empty() => {
                consec_fail = 0;
                let mut stored = 0u32;
                for f in &reports {
                    if upsert_stock_fundamental(&pool, f).await.is_ok() {
                        stored += 1;
                    }
                }
                rows_total += stored;
                ok += 1;
                if ok <= 3 || (idx + 1) % 20 == 0 {
                    // 接口按报告期倒序返回，first() 为最新一期
                    let latest = reports.first().unwrap();
                    println!(
                        "[{}/{max}] {symbol}: {stored} 个报告期 ✓（最新 {} ROE={:?} EPS={:?} BPS={:?}）",
                        idx + 1,
                        latest.report_date,
                        latest.roe,
                        latest.eps,
                        latest.bps
                    );
                }
            }
            Ok(_) => {
                consec_fail = 0; // 接口正常，只是没数据
                empty += 1;
                println!("[{}/{max}] {symbol}: 无财务数据，跳过", idx + 1);
            }
            Err(_) => {
                failed += 1;
                consec_fail += 1;
                println!("[{}/{max}] {symbol}: 拉取失败（连续 {consec_fail}）", idx + 1);
                if consec_fail >= 15 {
                    println!("⚠️ 连续 {consec_fail} 次失败，疑似当日额度耗尽，提前停止。");
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    println!(
        "\n完成：成功 {ok} 只 / 无数据 {empty} / 失败 {failed}，共写入 {rows_total} 个报告期行。"
    );
}
