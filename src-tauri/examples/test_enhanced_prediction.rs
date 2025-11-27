/// 测试增强的金融级预测功能
/// 展示K线形态识别和量价分析
///
/// 运行: cargo run --example test_enhanced_prediction

use biga_lib::prediction::types::PredictionRequest;
use biga_lib::prediction::model::inference;

#[tokio::main]
async fn main() {
    println!("\n🚀 ========== 增强版金融级预测测试 ==========\n");
    
    // 测试股票：贵州茅台
    let stock_code = "sh600519".to_string();
    
    let request = PredictionRequest {
        stock_code: stock_code.clone(),
        model_name: None,
        prediction_days: 5,
        use_candle: true,
    };
    
    match inference::predict(request).await {
        Ok(predictions) => {
            println!("\n✅ 预测成功！\n");
            
            // 显示最近真实数据
            if let Some(last_real) = &predictions.last_real_data {
                println!("📊 最新真实数据:");
                println!("   日期: {}", last_real.date);
                println!("   价格: {:.2}元", last_real.price);
                println!("   涨跌幅: {:+.2}%\n", last_real.change_percent);
            }
            
            println!("🎯 ========== 预测价格 ==========\n");
            for pred in predictions.predictions.iter().take(5) {
                let signal = pred.trading_signal.as_ref().map(|s| s.as_str()).unwrap_or("N/A");
                let strength = pred.signal_strength.unwrap_or(0.0);
                
                println!("   {} : {:.2}元 ({:+.2}%)", 
                         pred.target_date, 
                         pred.predicted_price,
                         pred.predicted_change_percent);
                println!("      信号: {} | 强度: {:.0}% | 置信度: {:.0}%",
                         signal, strength * 100.0, pred.confidence * 100.0);
                
                if let Some(ref reason) = pred.prediction_reason {
                    println!("      原因: {}", reason);
                }
                
                if let Some(ref factors) = pred.key_factors {
                    if !factors.is_empty() {
                        println!("      关键因素: {}", factors.join(", "));
                    }
                }
                println!();
            }
            
            // 技术指标展示
            if let Some(pred) = predictions.predictions.first() {
                if let Some(ref indicators) = pred.technical_indicators {
                    println!("📈 技术指标分析:");
                    println!("   RSI: {:.2}", indicators.rsi);
                    println!("   MACD柱: {:.4}", indicators.macd_histogram);
                    println!("   KDJ-J: {:.2}", indicators.kdj_j);
                    println!("   CCI: {:.2}", indicators.cci);
                    
                    if indicators.macd_golden_cross {
                        println!("   🟢 MACD金叉信号");
                    }
                    if indicators.macd_death_cross {
                        println!("   🔴 MACD死叉信号");
                    }
                    if indicators.kdj_oversold {
                        println!("   💡 KDJ超卖区域");
                    }
                    if indicators.kdj_overbought {
                        println!("   ⚠️ KDJ超买区域");
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("\n❌ 预测失败: {}", e);
        }
    }
    
    println!("\n✨ 测试完成！\n");
}
