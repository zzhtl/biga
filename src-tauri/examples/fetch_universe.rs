//! 按代码清单批量扩充全市场历史+股本（扩广度，提升截面 alpha 的独立样本）。
//! 代码清单从文件读取（每行一个纯 6 位代码），统一以纯 6 位入库（避免 .SZ/.SH 混格）。
//! 连续多次 API 失败会提前停止（疑似当日额度耗尽）。
//!
//! 用法：
//!   CODES_FILE=/tmp/new_codes.txt FETCH_MAX=400 cargo run --release --example fetch_universe

use biga_lib::api::stock::{fetch_historical_data, fetch_stock_capital};
use biga_lib::db::connection::create_pool;
use biga_lib::db::models::{HistoricalData, StockCapital, StockInfo};
use biga_lib::db::repository::{
    backfill_volume_metrics, batch_insert_historical_data, batch_insert_stock_info,
    get_latest_close_price, upsert_stock_capital,
};

/// 单只股票历史拉取结果
enum Fetched {
    Data(Vec<HistoricalData>),
    Insufficient(usize),
    ApiError,
}

/// 先试纯 6 位代码，失败再试带交易所后缀；都失败返回 ApiError。
async fn fetch_history(code: &str, suffixed: &str) -> Fetched {
    match fetch_historical_data(code).await {
        Ok(v) if v.len() >= 120 => return Fetched::Data(v),
        Ok(_) | Err(_) => {}
    }
    match fetch_historical_data(suffixed).await {
        Ok(v) if v.len() >= 120 => Fetched::Data(v),
        Ok(v) => Fetched::Insufficient(v.len()),
        Err(_) => Fetched::ApiError,
    }
}

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

    let path = std::env::var("CODES_FILE").unwrap_or_else(|_| "/tmp/new_codes.txt".to_string());
    let codes: Vec<String> = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("读取代码清单失败 {path}: {e}"))
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| l.len() == 6 && l.chars().all(|c| c.is_ascii_digit()))
        .collect();
    let max = std::env::var("FETCH_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(codes.len())
        .min(codes.len());
    println!("待拉取 {} 只（清单 {} 条，上限 {max}）", max, codes.len());

    let (mut calls, mut ok, mut insufficient, mut failed, mut consec_fail) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for (idx, code) in codes.iter().take(max).enumerate() {
        let exchange = if code.starts_with('6') { "sh" } else { "sz" };
        let suffixed = format!("{code}.{}", exchange.to_uppercase());

        calls += 2; // 至多两次历史尝试
        let mut data = match fetch_history(code, &suffixed).await {
            Fetched::Data(v) => {
                consec_fail = 0;
                v
            }
            Fetched::Insufficient(n) => {
                consec_fail = 0; // 接口正常，只是该股历史不够
                insufficient += 1;
                println!("[{}/{max}] {code}: 历史不足({n})，跳过", idx + 1);
                continue;
            }
            Fetched::ApiError => {
                failed += 1;
                consec_fail += 1;
                println!("[{}/{max}] {code}: 拉取失败（连续 {consec_fail}）", idx + 1);
                if consec_fail >= 15 {
                    println!("⚠️ 连续 {consec_fail} 次失败，疑似当日额度耗尽，提前停止。");
                    break;
                }
                continue;
            }
        };

        // 统一以纯 6 位代码入库（无论历史是经后缀拉到的）
        for d in data.iter_mut() {
            d.symbol = code.clone();
        }
        let info = StockInfo {
            symbol: code.clone(),
            name: code.clone(),
            exchange: exchange.to_string(),
        };
        let _ = batch_insert_stock_info(&pool, vec![info]).await;

        let bars = data.len();
        if let Err(e) = batch_insert_historical_data(code, &pool, data).await {
            println!("[{}/{max}] {code}: 入库失败 {e}", idx + 1);
            failed += 1;
            continue;
        }

        // 股本 + 量比/换手率回填
        calls += 1;
        if let Ok(q) = fetch_stock_capital(code).await {
            if q.circulating_market_cap > 0.0 {
                if let Ok(Some(close)) = get_latest_close_price(code, &pool).await {
                    if close > 0.0 {
                        let cap = StockCapital {
                            symbol: code.clone(),
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
        let _ = backfill_volume_metrics(code, &pool).await;

        ok += 1;
        if ok <= 3 || (idx + 1) % 20 == 0 {
            println!("[{}/{max}] {code}: 入库 {bars} 条 ✓（成功 {ok}，调用约 {calls}）", idx + 1);
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }

    println!(
        "\n完成：成功 {ok} 只 / 历史不足 {insufficient} / 失败 {failed}，共调用接口约 {calls} 次。"
    );
}
