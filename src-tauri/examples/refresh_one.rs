//! 一键刷新单只股票的全部所需数据（历史 + 股本/估值PE/PB + 基本面 + 量比换手率回填），
//! 与前端"刷新"按钮走同一条 `services::historical::refresh_stock_full`。
//! 也用于在 Tauri 外验证该流程、并把估值列(迁移05)幂等补到现有 DB。
//!
//! 用法：REFRESH_SYMBOL=600519 cargo run --release --example refresh_one

use biga_lib::db::connection::create_pool;
use biga_lib::db::repository::{get_stock_capital, get_stock_fundamentals};
use biga_lib::services::historical::refresh_stock_full;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // examples 不走 Tauri 启动迁移：幂等确保 stock_fundamentals 表与 stock_capital 估值列存在。
    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_fundamentals (symbol TEXT NOT NULL, report_date TEXT NOT NULL, eps REAL, bps REAL, roe REAL, profit_growth REAL, revenue_growth REAL, debt_ratio REAL, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (symbol, report_date))",
        "ALTER TABLE stock_capital ADD COLUMN pe REAL NOT NULL DEFAULT 0",
        "ALTER TABLE stock_capital ADD COLUMN pb REAL NOT NULL DEFAULT 0",
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await;
    }

    let symbol = std::env::var("REFRESH_SYMBOL").unwrap_or_else(|_| "000001".to_string());
    println!("刷新 {symbol} 的全部数据……");

    match refresh_stock_full(&symbol, &pool).await {
        Ok(s) => {
            println!(
                "✓ 完成：历史 {} 条 / 股本估值更新 {} / 基本面 {} 个报告期",
                s.bars, s.capital_updated, s.fundamental_reports
            );
            // 回读验证
            if let Ok(Some(cap)) = get_stock_capital(&symbol, &pool).await {
                println!(
                    "  估值：PE={:.2} PB={:.2} 流通市值={:.2}亿",
                    cap.pe,
                    cap.pb,
                    cap.circulating_market_cap / 1.0e8
                );
            }
            if let Ok(funds) = get_stock_fundamentals(&symbol, &pool).await {
                if let Some(latest) = funds.last() {
                    println!(
                        "  基本面(最新报告期 {})：ROE={:?} EPS={:?} BPS={:?}",
                        latest.report_date, latest.roe, latest.eps, latest.bps
                    );
                }
            }
        }
        Err(e) => eprintln!("✗ 刷新失败：{e}"),
    }
}
