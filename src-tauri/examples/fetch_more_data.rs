//! 拉取更多大盘股历史+股本，扩充回测样本。控制在每日调用额度内。
//! 用法：cargo run --example fetch_more_data

use biga_lib::api::stock::{fetch_historical_data, fetch_stock_capital};
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::{StockCapital, StockInfo};
use biga_lib::db::repository::{
    backfill_volume_metrics, batch_insert_historical_data, batch_insert_stock_info,
    get_latest_close_price, upsert_stock_capital,
};

/// 第三批 ~35 只（纯 6 位代码，与前两批不重复）
const CODES: &[&str] = &[
    "600015", "600018", "600183", "600309", "600352", "600362", "600487", "600498", "600515",
    "600518", "600570", "600637", "600690", "600741", "600795", "600809", "600848", "600886",
    "601009", "601021", "601155", "601169", "601225", "601229", "601328", "601360", "601555",
    "601607", "601688", "601818", "601877", "601898", "601919", "601939", "601998",
];

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    for stmt in [
        "CREATE TABLE IF NOT EXISTS stock_capital (symbol TEXT PRIMARY KEY, circulating_shares REAL NOT NULL DEFAULT 0, total_shares REAL NOT NULL DEFAULT 0, circulating_market_cap REAL NOT NULL DEFAULT 0, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        "ALTER TABLE stock_capital ADD COLUMN pe REAL NOT NULL DEFAULT 0",
        "ALTER TABLE stock_capital ADD COLUMN pb REAL NOT NULL DEFAULT 0",
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
                            pe: q.pe,
                            pb: q.pb,
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
