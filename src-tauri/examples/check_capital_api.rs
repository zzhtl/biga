//! 闭环检测：直接调用 zhitu hs/real/ssjy 接口，确认能否返回 lt（流通市值）。
//! 用法：cargo run --example check_capital_api

use biga_lib::api::stock::fetch_stock_capital;

#[tokio::main]
async fn main() {
    let token = std::env::var("STOCK_API_TOKEN")
        .unwrap_or_else(|_| "C5BFE522-34E7-4931-8216-CAD281648165".to_string());
    let client = reqwest::Client::new();

    // 试多种股票代码格式，找出接口接受的形式
    let candidates = [
        "000001",
        "sz000001",
        "000001.SZ",
        "600519",
        "sh600519",
        "600519.SH",
        "000002.SZ",
    ];

    println!("=== 原始响应 ===");
    for sym in candidates {
        let url = format!("https://api.zhituapi.com/hs/real/ssjy/{sym}?token={token}");
        match client
            .get(&url)
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await
        {
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                let preview: String = body.chars().take(300).collect();
                println!("[{sym}] status={status} body={preview}");
            }
            Err(e) => println!("[{sym}] 请求错误: {e}"),
        }
    }

    println!("\n=== fetch_stock_capital 解析结果 ===");
    for sym in ["000001", "sz000001", "600519", "000002.SZ"] {
        match fetch_stock_capital(sym).await {
            Ok(q) => println!(
                "{sym} => lt(流通市值)={} sz(总市值)={} hs(换手)={} lb(量比)={}",
                q.circulating_market_cap, q.total_market_cap, q.turnover_rate, q.volume_ratio
            ),
            Err(e) => println!("{sym} 解析失败: {e}"),
        }
    }
}
