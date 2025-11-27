/// 测试获取历史数据
///
/// 运行: cargo run --example test_historical_data

use biga_lib::api::stock;

#[tokio::main]
async fn main() {
    // 测试获取历史数据
    println!("🚀 测试获取股票历史数据...\n");
    let symbol = "603005.SH"; // 晶方科技
    
    match stock::fetch_historical_data(symbol).await {
        Ok(data) => {
            println!("✅ 成功获取 {} 条历史数据\n", data.len());
            
            // 打印前5条数据
            println!("📊 前5条数据示例:");
            println!("{:-<100}", "");
            for (i, item) in data.iter().take(5).enumerate() {
                println!(
                    "{}: {} | 开: {:.2} | 收: {:.2} | 高: {:.2} | 低: {:.2} | 量: {} | 额: {:.2} | 涨跌幅: {:+.2}%",
                    i+1, item.date, item.open, item.close, item.high, item.low, 
                    item.volume, item.amount, item.change_percent
                );
            }
            println!("{:-<100}", "");
            
            // 打印最后5条数据
            if data.len() > 5 {
                println!("\n📊 最近5条数据:");
                println!("{:-<100}", "");
                let start = data.len().saturating_sub(5);
                for (i, item) in data.iter().skip(start).enumerate() {
                    println!(
                        "{}: {} | 开: {:.2} | 收: {:.2} | 高: {:.2} | 低: {:.2} | 量: {} | 额: {:.2} | 涨跌幅: {:+.2}%",
                        start + i + 1, item.date, item.open, item.close, item.high, item.low,
                        item.volume, item.amount, item.change_percent
                    );
                }
                println!("{:-<100}", "");
            }
        }
        Err(err) => {
            eprintln!("❌ 获取历史数据失败: {err}");
        }
    }
    
    println!("\n✨ 测试完成！");
} 