/// 测试金融级股票预测策略
/// 
/// 使用方法：
/// ```bash
/// cargo run --example test_professional_prediction
/// ```

use biga_lib::stock_prediction::types::PredictionRequest;
use biga_lib::stock_prediction::prediction::predict_with_professional_strategy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 ========== 金融级股票预测策略测试 ==========\n");
    
    // 示例：分析贵州茅台
    let stock_codes = vec![
        "600519.SH",  // 贵州茅台
        // "000001.SZ",  // 平安银行
        // "600036.SH",  // 招商银行
    ];
    
    for stock_code in stock_codes {
        println!("\n{}", "=".repeat(60));
        println!("📊 正在分析股票: {}", stock_code);
        println!("{}\n", "=".repeat(60));
        
        let request = PredictionRequest {
            stock_code: stock_code.to_string(),
            prediction_days: 5,
            model_name: None,
            use_candle: true,
        };
        
        match predict_with_professional_strategy(request).await {
            Ok((prediction_response, professional_analysis)) => {
                println!("\n📈 ========== 预测结果汇总 ==========");
                
                // 显示最近真实数据
                if let Some(last_real) = &prediction_response.last_real_data {
                    println!("最新真实数据:");
                    println!("  日期: {}", last_real.date);
                    println!("  价格: {:.2}元", last_real.price);
                    println!("  涨跌幅: {:+.2}%\n", last_real.change_percent);
                }
                
                // 显示未来预测
                println!("未来{}天预测:", prediction_response.predictions.len());
                for (i, pred) in prediction_response.predictions.iter().enumerate() {
                    println!(
                        "  Day {}: {} - 价格: {:.2}元 ({:+.2}%) | 置信度: {:.0}% | 信号: {}",
                        i + 1,
                        pred.target_date,
                        pred.predicted_price,
                        pred.predicted_change_percent,
                        pred.confidence * 100.0,
                        pred.trading_signal.as_ref().unwrap_or(&"N/A".to_string())
                    );
                }
                
                println!("\n🎯 ========== 专业分析总结 ==========");
                println!("操作建议: {}", professional_analysis.current_advice);
                println!("风险评级: {}", professional_analysis.risk_level);
                
                // 买入点总结
                if !professional_analysis.buy_points.is_empty() {
                    println!("\n💎 最佳买入点:");
                    let best_buy = &professional_analysis.buy_points[0];
                    println!("  类型: {}", best_buy.point_type);
                    println!("  信号强度: {:.0}/100", best_buy.signal_strength);
                    println!("  建议价格: {:.2}元", best_buy.price_level);
                    println!("  止损位: {:.2}元", best_buy.stop_loss);
                    println!("  止盈位: {}", 
                        best_buy.take_profit.iter()
                            .map(|p| format!("{:.2}元", p))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    println!("  风险收益比: 1:{:.2}", best_buy.risk_reward_ratio);
                    println!("  置信度: {:.0}%", best_buy.confidence * 100.0);
                }
                
                // 卖出点总结
                if !professional_analysis.sell_points.is_empty() {
                    println!("\n⚠️  最佳卖出点:");
                    let best_sell = &professional_analysis.sell_points[0];
                    println!("  类型: {}", best_sell.point_type);
                    println!("  信号强度: {:.0}/100", best_sell.signal_strength);
                    println!("  建议价格: {:.2}元", best_sell.price_level);
                    println!("  止损位: {:.2}元", best_sell.stop_loss);
                    println!("  目标位: {}", 
                        best_sell.take_profit.iter()
                            .map(|p| format!("{:.2}元", p))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                    println!("  置信度: {:.0}%", best_sell.confidence * 100.0);
                }
                
                println!("\n✅ {} 分析完成！\n", stock_code);
            }
            Err(e) => {
                eprintln!("❌ 预测失败: {}", e);
            }
        }
    }
    
    println!("\n🎉 所有股票分析完成！");
    println!("\n⚠️  风险提示：");
    println!("  1. 本预测仅供参考，不构成投资建议");
    println!("  2. 股市有风险，投资需谨慎");
    println!("  3. 请根据自身风险承受能力做出决策");
    println!("  4. 建议设置止损位，控制风险\n");
    
    Ok(())
} 