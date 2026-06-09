//! 批量回填全库**全部非历史维度**：股本/估值(PE/PB) + 基本面(ROE/EPS/BPS/增长率) + 量比/换手率。
//! 不重拉历史（使用现有K线计算），一条命令补齐全库数据，避免分散多次操作。
//! 连续多次 API 失败提前停止（疑似当日额度耗尽）。
//! 用法：cargo run --release --example batch_refresh

use biga_lib::api::stock::{fetch_financial_indicators, fetch_stock_capital};
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::StockCapital;
use biga_lib::db::repository::{
    backfill_volume_metrics, get_latest_close_price, upsert_stock_capital, upsert_stock_fundamental,
};
use sqlx::Row;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // 应用迁移 03（建 stock_capital 表 + 量比列），与 lib.rs 启动迁移一致
    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_capital (symbol TEXT PRIMARY KEY, circulating_shares REAL NOT NULL DEFAULT 0, total_shares REAL NOT NULL DEFAULT 0, circulating_market_cap REAL NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "ALTER TABLE stock_capital ADD COLUMN pe REAL NOT NULL DEFAULT 0",
        "ALTER TABLE stock_capital ADD COLUMN pb REAL NOT NULL DEFAULT 0",
        "ALTER TABLE historical_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
        "ALTER TABLE realtime_data ADD COLUMN volume_ratio REAL NOT NULL DEFAULT 0",
        // 迁移 04：基本面表（与 fetch_fundamentals 一致）
        "CREATE TABLE IF NOT EXISTS stock_fundamentals (symbol TEXT NOT NULL, report_date TEXT NOT NULL, eps REAL, bps REAL, roe REAL, profit_growth REAL, revenue_growth REAL, debt_ratio REAL, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (symbol, report_date))",
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await;
    }

    // 未填 PE 的（pe=0 或无股本行）排前面，使受额度限制时每次重跑都优先补未完成的票，
    // 避免重复消耗额度在已填的票上。
    let rows = sqlx::query(
        "SELECT h.symbol FROM (SELECT DISTINCT symbol FROM historical_data) h \
         LEFT JOIN stock_capital c ON c.symbol = h.symbol \
         ORDER BY CASE WHEN COALESCE(c.pe, 0) = 0 THEN 0 ELSE 1 END, h.symbol",
    )
    .fetch_all(&pool)
    .await
    .expect("查询股票失败");

    let total = rows.len();
    println!("待处理股票数：{total}");

    let mut ok_capital = 0usize;
    let mut ok_fund = 0usize;
    let mut fund_rows = 0u32;
    let mut filled = 0usize;
    let mut consec_fail = 0u32; // 连续 API 失败计数（探测当日额度耗尽）
    for (i, r) in rows.iter().enumerate() {
        let symbol: String = r.get("symbol");

        // 拉股本/估值（失败不阻断，量比仍可回填）
        match fetch_stock_capital(&symbol).await {
            Ok(q) if q.circulating_market_cap > 0.0 => {
                consec_fail = 0;
                match get_latest_close_price(&symbol, &pool).await {
                    Ok(Some(close)) if close > 0.0 => {
                        let cap = StockCapital {
                            symbol: symbol.clone(),
                            circulating_shares: q.circulating_market_cap / close,
                            total_shares: q.total_market_cap / close,
                            circulating_market_cap: q.circulating_market_cap,
                            pe: q.pe,
                            pb: q.pb,
                        };
                        match upsert_stock_capital(&pool, &cap).await {
                            Ok(_) => ok_capital += 1,
                            Err(e) => println!("  [{symbol}] upsert 失败: {e}"),
                        }
                    }
                    Ok(_) => println!("  [{symbol}] 无最新收盘价"),
                    Err(e) => println!("  [{symbol}] 取收盘价失败: {e}"),
                }
            }
            Ok(_) => {
                consec_fail = 0; // 接口正常，只是数据为 0
                println!("  [{symbol}] 股本数据为 0");
            }
            Err(e) => {
                consec_fail += 1;
                println!("  [{symbol}] 取股本失败（连续 {consec_fail}）: {e}");
            }
        }

        // 基本面财务指标（cwzb：ROE/EPS/BPS/增长率，非技术预测维度）
        match fetch_financial_indicators(&symbol).await {
            Ok(reports) if !reports.is_empty() => {
                consec_fail = 0;
                let mut stored = 0u32;
                for f in &reports {
                    if upsert_stock_fundamental(&pool, f).await.is_ok() {
                        stored += 1;
                    }
                }
                fund_rows += stored;
                ok_fund += 1;
            }
            Ok(_) => consec_fail = 0, // 接口正常，无数据
            Err(e) => {
                consec_fail += 1;
                println!("  [{symbol}] 取基本面失败（连续 {consec_fail}）: {e}");
            }
        }

        if consec_fail >= 15 {
            println!("⚠️ 连续 {consec_fail} 次 API 失败，疑似当日额度耗尽，提前停止（已处理 {} 只）。", i + 1);
            break;
        }

        match backfill_volume_metrics(&symbol, &pool).await {
            Ok(_) => filled += 1,
            Err(e) => println!("  [{symbol}] 回填失败: {e}"),
        }

        if (i + 1) % 50 == 0 || i + 1 == total {
            println!(
                "进度 {}/{total}（股本 {ok_capital}，基本面 {ok_fund}/{fund_rows}行，回填 {filled}）",
                i + 1
            );
        }

        // 轻微限速，避免触发接口频控
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    }

    println!(
        "\n完成：共 {total} 只，股本成功 {ok_capital}，基本面 {ok_fund} 只/{fund_rows} 个报告期行，量比/换手率回填 {filled}"
    );
}
