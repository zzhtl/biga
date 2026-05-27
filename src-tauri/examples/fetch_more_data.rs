//! 拉取更多大盘股历史+股本，扩充回测样本。控制在每日调用额度内。
//! 用法：cargo run --example fetch_more_data

use biga_lib::api::stock::{fetch_historical_data, fetch_stock_capital};
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::{StockCapital, StockInfo};
use biga_lib::db::repository::{
    backfill_volume_metrics, batch_insert_historical_data, batch_insert_stock_info,
    get_latest_close_price, upsert_stock_capital,
};

/// 约 40 只流动性较好、历史较长的大盘股（纯 6 位代码）
const CODES: &[&str] = &[
    "600519", "600036", "601318", "600276", "600030", "601166", "600887", "600009", "601888",
    "600031", "600585", "601012", "600438", "603259", "600104", "601668", "601398", "601288",
    "601988", "600028", "000001", "000002", "000333", "000651", "000858", "002415", "002594",
    "300750", "300059", "000725", "002304", "000568", "002230", "300760", "002352", "000063",
    "600196", "601658", "600050", "601601",
];

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

    let mut calls = 0u32;
    let mut ok = 0u32;
    for code in CODES {
        // 历史数据（先试纯代码，失败再试带交易所后缀）
        let exchange = if code.starts_with('6') { "sh" } else { "sz" };
        let suffixed = format!("{code}.{}", exchange.to_uppercase());

        calls += 1;
        let (used_symbol, data) = match fetch_historical_data(code).await {
            Ok(v) if v.len() >= 120 => (code.to_string(), v),
            _ => {
                calls += 1;
                match fetch_historical_data(&suffixed).await {
                    Ok(v) if v.len() >= 120 => (suffixed.clone(), v),
                    Ok(v) => {
                        println!("{code}: 历史数据不足({})，跳过", v.len());
                        continue;
                    }
                    Err(e) => {
                        println!("{code}: 拉取历史失败 {e}");
                        continue;
                    }
                }
            }
        };

        // 确保 stock_info 有该代码（batch_insert_historical_data 需要 name）
        let info = StockInfo {
            symbol: used_symbol.clone(),
            name: used_symbol.clone(),
            exchange: exchange.to_string(),
        };
        let _ = batch_insert_stock_info(&pool, vec![info]).await;

        let bars = data.len();
        if let Err(e) = batch_insert_historical_data(&used_symbol, &pool, data).await {
            println!("{used_symbol}: 入库失败 {e}");
            continue;
        }

        // 股本 + 回填
        calls += 1;
        if let Ok(q) = fetch_stock_capital(&used_symbol).await {
            if q.circulating_market_cap > 0.0 {
                if let Ok(Some(close)) = get_latest_close_price(&used_symbol, &pool).await {
                    if close > 0.0 {
                        let cap = StockCapital {
                            symbol: used_symbol.clone(),
                            circulating_shares: q.circulating_market_cap / close,
                            total_shares: q.total_market_cap / close,
                            circulating_market_cap: q.circulating_market_cap,
                        };
                        let _ = upsert_stock_capital(&pool, &cap).await;
                    }
                }
            }
        }
        let _ = backfill_volume_metrics(&used_symbol, &pool).await;

        ok += 1;
        println!("{used_symbol}: 入库 {bars} 条 ✓（累计调用 {calls}）");

        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    println!("\n完成：成功 {ok}/{} 只，共调用接口约 {calls} 次", CODES.len());
}
