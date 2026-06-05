//! 一次性数据覆盖率探查：看非技术数据（行业分类 / 市值）在库里是否可用，
//! 以判断"行业/市值中性化截面"是否具备落地前提。用完即可删。

use biga_lib::db::connection::create_pool;
use sqlx::Row;

#[tokio::main]
async fn main() {
    let pool = create_pool().await.expect("创建连接池失败");

    // 历史数据可用股票数（≥300 根）
    let bars = sqlx::query(
        "SELECT COUNT(*) c FROM (SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 300)",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    println!("历史≥300根的股票数: {}", bars.get::<i64, _>("c"));

    // stock 表行业覆盖
    let total_stock = sqlx::query("SELECT COUNT(*) c FROM stock")
        .fetch_one(&pool)
        .await
        .map(|r| r.get::<i64, _>("c"))
        .unwrap_or(-1);
    let with_industry = sqlx::query(
        "SELECT COUNT(*) c FROM stock WHERE industry IS NOT NULL AND TRIM(industry) <> ''",
    )
    .fetch_one(&pool)
    .await
    .map(|r| r.get::<i64, _>("c"))
    .unwrap_or(-1);
    let distinct_industry = sqlx::query(
        "SELECT COUNT(DISTINCT industry) c FROM stock WHERE industry IS NOT NULL AND TRIM(industry) <> ''",
    )
    .fetch_one(&pool)
    .await
    .map(|r| r.get::<i64, _>("c"))
    .unwrap_or(-1);
    println!(
        "stock 表: 总数={} 有行业={} 不同行业数={}",
        total_stock, with_industry, distinct_industry
    );

    // 市值覆盖（stock_capital）
    let cap_total = sqlx::query("SELECT COUNT(*) c FROM stock_capital")
        .fetch_one(&pool)
        .await
        .map(|r| r.get::<i64, _>("c"))
        .unwrap_or(-1);
    let cap_pos = sqlx::query(
        "SELECT COUNT(*) c FROM stock_capital WHERE circulating_market_cap > 0",
    )
    .fetch_one(&pool)
    .await
    .map(|r| r.get::<i64, _>("c"))
    .unwrap_or(-1);
    println!("stock_capital: 总数={} 流通市值>0={}", cap_total, cap_pos);

    // 同时满足：历史≥300 且 有行业 且 市值>0 —— 中性化截面的可用样本
    let usable = sqlx::query(
        "SELECT COUNT(*) c FROM (
           SELECT h.symbol
           FROM historical_data h
           JOIN stock s ON s.symbol = h.symbol AND s.industry IS NOT NULL AND TRIM(s.industry) <> ''
           JOIN stock_capital c ON c.symbol = h.symbol AND c.circulating_market_cap > 0
           GROUP BY h.symbol HAVING COUNT(*) >= 300
         )",
    )
    .fetch_one(&pool)
    .await
    .map(|r| r.get::<i64, _>("c"))
    .unwrap_or(-1);
    println!("可用于中性化截面(历史≥300 且 有行业 且 市值>0)的股票数: {}", usable);

    // 行业分布 Top（看每行业有几只，能否做组内中性化）
    let rows = sqlx::query(
        "SELECT s.industry ind, COUNT(*) c
         FROM stock s
         JOIN stock_capital cap ON cap.symbol = s.symbol AND cap.circulating_market_cap > 0
         WHERE s.industry IS NOT NULL AND TRIM(s.industry) <> ''
           AND s.symbol IN (SELECT symbol FROM historical_data GROUP BY symbol HAVING COUNT(*) >= 300)
         GROUP BY s.industry ORDER BY c DESC LIMIT 25",
    )
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    println!("行业分布(可用域, Top25 / 共上面那么多只):");
    for r in &rows {
        println!("  {:<12} {}", r.get::<String, _>("ind"), r.get::<i64, _>("c"));
    }
}
