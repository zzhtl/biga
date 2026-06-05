//! 批量回填全库股本与量比/换手率（不重拉历史，使用现有K线计算），使数据完整。
//! 用法：cargo run --example batch_refresh

use biga_lib::api::stock::fetch_stock_capital;
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::StockCapital;
use biga_lib::db::repository::{backfill_volume_metrics, get_latest_close_price, upsert_stock_capital};
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
    ] {
        let _ = sqlx::query(stmt).execute(&pool).await;
    }

    let rows = sqlx::query("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        .fetch_all(&pool)
        .await
        .expect("查询股票失败");

    let total = rows.len();
    println!("待处理股票数：{total}");

    let mut ok_capital = 0usize;
    let mut filled = 0usize;
    for (i, r) in rows.iter().enumerate() {
        let symbol: String = r.get("symbol");

        // 拉股本（失败不阻断，量比仍可回填）
        match fetch_stock_capital(&symbol).await {
            Ok(q) if q.circulating_market_cap > 0.0 => {
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
            Ok(_) => println!("  [{symbol}] 股本数据为 0"),
            Err(e) => println!("  [{symbol}] 取股本失败: {e}"),
        }

        match backfill_volume_metrics(&symbol, &pool).await {
            Ok(_) => filled += 1,
            Err(e) => println!("  [{symbol}] 回填失败: {e}"),
        }

        if (i + 1) % 50 == 0 || i + 1 == total {
            println!("进度 {}/{total}（股本成功 {ok_capital}，回填 {filled}）", i + 1);
        }

        // 轻微限速，避免触发接口频控
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    }

    println!("\n完成：共 {total} 只，股本成功 {ok_capital}，量比/换手率回填 {filled}");
}
