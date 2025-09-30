/// 测试增强的金融级预测功能
/// 展示K线形态识别和量价分析

use biga_lib::stock_prediction::types::PredictionRequest;
use biga_lib::stock_prediction::prediction::predict_with_professional_strategy;

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
    
    match predict_with_professional_strategy(request).await {
        Ok((predictions, analysis)) => {
            println!("\n✅ 预测成功！\n");
            
            println!("📊 ========== 新增功能展示 ==========\n");
            
            // 展示K线形态
            println!("🕯️  K线形态识别:");
            if analysis.candle_patterns.is_empty() {
                println!("   未检测到明显形态");
            } else {
                for pattern in &analysis.candle_patterns {
                    println!("   ✨ {}", pattern.description);
                    println!("      强度: {:.1}% | 可靠性: {:.1}%", 
                             pattern.strength * 100.0,
                             pattern.reliability * 100.0);
                }
            }
            
            println!("\n📈 量价关系分析:");
            println!("   量能趋势: {}", analysis.volume_analysis.volume_trend);
            println!("   量价配合: {}", if analysis.volume_analysis.volume_price_sync { "✅ 良好" } else { "⚠️ 背离" });
            println!("   OBV趋势: {}", analysis.volume_analysis.obv_trend);
            println!("   吸筹信号: {:.0}分", analysis.volume_analysis.accumulation_signal);
            
            if analysis.volume_analysis.accumulation_signal > 70.0 {
                println!("   💎 强烈的主力吸筹信号！");
            } else if analysis.volume_analysis.accumulation_signal > 50.0 {
                println!("   💡 检测到可能的主力吸筹");
            }
            
            println!("\n💎 ========== 综合分析 ==========\n");
            
            // 买入点
            if !analysis.buy_points.is_empty() {
                println!("🟢 买入信号 ({}个):", analysis.buy_points.len());
                for bp in &analysis.buy_points {
                    println!("   📍 {} (强度: {:.0}分, 置信度: {:.0}%)", 
                             bp.point_type, 
                             bp.signal_strength,
                             bp.confidence * 100.0);
                }
            }
            
            // 卖出点
            if !analysis.sell_points.is_empty() {
                println!("\n🔴 卖出信号 ({}个):", analysis.sell_points.len());
                for sp in &analysis.sell_points {
                    println!("   📍 {} (强度: {:.0}分, 置信度: {:.0}%)", 
                             sp.point_type,
                             sp.signal_strength,
                             sp.confidence * 100.0);
                }
            }
            
            println!("\n📋 操作建议: {}", analysis.current_advice);
            println!("⚠️  风险等级: {}", analysis.risk_level);
            
            println!("\n🎯 ========== 预测价格 ==========\n");
            for pred in predictions.predictions.iter().take(5) {
                println!("   {} : {:.2}元 ({:+.2}%)", 
                         pred.target_date, 
                         pred.predicted_price,
                         pred.predicted_change_percent);
            }
        }
        Err(e) => {
            eprintln!("\n❌ 预测失败: {}", e);
        }
    }
    
    println!("\n✨ 测试完成！\n");
} 