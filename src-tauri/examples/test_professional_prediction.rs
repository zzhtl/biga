/// 测试金融级股票预测策略
/// 
/// 使用方法：
/// ```bash
/// cargo run --example test_professional_prediction
/// ```

use biga_lib::prediction::types::PredictionRequest;
use biga_lib::prediction::model::inference;
use biga_lib::prediction::analysis::{trend, volume, pattern, support_resistance};
use biga_lib::prediction::indicators;
use biga_lib::prediction::strategy::multi_factor;
use biga_lib::db::{connection::create_temp_pool, repository::get_recent_historical_data};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 ========== 金融级股票预测策略测试 ==========\n");
    
    // 示例：分析贵州茅台
    let stock_codes = vec![
        "sh600519",  // 贵州茅台
    ];
    
    for stock_code in stock_codes {
        println!("\n{}", "=".repeat(60));
        println!("📊 正在分析股票: {}", stock_code);
        println!("{}\n", "=".repeat(60));
        
        // 获取历史数据进行专业分析
        let pool = match create_temp_pool().await {
            Ok(p) => p,
            Err(e) => {
                eprintln!("❌ 数据库连接失败: {}", e);
                continue;
            }
        };
        
        let historical = match get_recent_historical_data(stock_code, 200, &pool).await {
            Ok(h) => h,
            Err(e) => {
                eprintln!("❌ 获取历史数据失败: {}", e);
                continue;
            }
        };
        
        if historical.is_empty() {
            eprintln!("⚠️ 未找到该股票的历史数据");
            continue;
        }
        
        // 提取数据
        let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
        let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
        let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
        let volumes: Vec<i64> = historical.iter().map(|h| h.volume).collect();
        let opens: Vec<f64> = historical.iter().map(|h| h.open).collect();
        
        let current_price = *prices.last().unwrap();
        let last_data = historical.last().unwrap();
        
        println!("📈 最新数据:");
        println!("   日期: {}", last_data.date);
        println!("   价格: {:.2}元", current_price);
        println!("   涨跌幅: {:+.2}%\n", last_data.change_percent);
        
        // 技术分析
        let trend_analysis = trend::analyze_trend(&prices, &highs, &lows);
        let volume_signal = volume::analyze_volume_price(&prices, &highs, &lows, &volumes);
        let patterns = pattern::recognize_patterns(&opens, &prices, &highs, &lows);
        let sr = support_resistance::calculate_support_resistance(&prices, &highs, &lows, current_price);
        let tech_indicators = indicators::calculate_all_indicators(&prices, &highs, &lows, &volumes);
        
        // 波动率计算
        let volatility = trend::calculate_historical_volatility(&prices, 20);
        
        // 多因子评分
        let multi_factor_score = multi_factor::calculate_multi_factor_score(
            &trend_analysis.overall_trend,
            &volume_signal,
            &tech_indicators,
            &patterns,
            &sr,
            volatility,
        );
        
        println!("📊 技术指标分析:");
        println!("   RSI: {:.2}", tech_indicators.rsi);
        println!("   MACD DIF: {:.4}", tech_indicators.macd_dif);
        println!("   MACD DEA: {:.4}", tech_indicators.macd_dea);
        println!("   MACD柱: {:.4}", tech_indicators.macd_histogram);
        println!("   KDJ-K: {:.2}", tech_indicators.kdj_k);
        println!("   KDJ-D: {:.2}", tech_indicators.kdj_d);
        println!("   KDJ-J: {:.2}", tech_indicators.kdj_j);
        println!("   CCI: {:.2}", tech_indicators.cci);
        
        println!("\n📈 趋势分析:");
        println!("   描述: {}", trend_analysis.description);
        println!("   趋势强度: {:.2}", trend_analysis.trend_strength);
        println!("   置信度: {:.0}%", trend_analysis.trend_confidence * 100.0);
        
        println!("\n📊 量价分析:");
        println!("   方向: {}", volume_signal.direction);
        println!("   信号: {}", volume_signal.signal);
        println!("   价格趋势: {}", volume_signal.price_trend);
        println!("   成交量趋势: {}", volume_signal.volume_trend);
        println!("   关键因素: {}", volume_signal.key_factors.join(", "));
        
        println!("\n🕯️ K线形态:");
        if patterns.is_empty() {
            println!("   未检测到明显形态");
        } else {
            for p in &patterns {
                let signal = if p.is_bullish { "看涨" } else { "看跌" };
                println!("   {} ({}) - 可靠性: {:.0}%", p.pattern_type, signal, p.reliability * 100.0);
                println!("      {}", p.description);
            }
        }
        
        println!("\n💪 支撑/阻力位:");
        println!("   当前位置: {}", sr.current_position);
        if !sr.support_levels.is_empty() {
            println!("   支撑位: {:?}", sr.support_levels.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
        }
        if !sr.resistance_levels.is_empty() {
            println!("   阻力位: {:?}", sr.resistance_levels.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>());
        }
        
        println!("\n🎯 多因子综合评分:");
        println!("   总评分: {:.1}/100", multi_factor_score.total_score);
        println!("   趋势因子: {:.1}", multi_factor_score.trend_score);
        println!("   量价因子: {:.1}", multi_factor_score.volume_price_score);
        println!("   动量因子: {:.1}", multi_factor_score.momentum_score);
        println!("   形态因子: {:.1}", multi_factor_score.pattern_score);
        println!("   支撑阻力因子: {:.1}", multi_factor_score.support_resistance_score);
        println!("   情绪因子: {:.1}", multi_factor_score.sentiment_score);
        println!("   波动率因子: {:.1}", multi_factor_score.volatility_score);
        println!("   信号: {} (强度: {:.0}%)", multi_factor_score.signal, multi_factor_score.signal_strength * 100.0);
        
        // 进行预测
        let request = PredictionRequest {
            stock_code: stock_code.to_string(),
            prediction_days: 5,
            model_name: None,
            use_candle: true,
        };
        
        match inference::predict(request).await {
            Ok(prediction_response) => {
                println!("\n🔮 未来{}天预测:", prediction_response.predictions.len());
                for (i, pred) in prediction_response.predictions.iter().enumerate() {
                    println!(
                        "   Day {}: {} - 价格: {:.2}元 ({:+.2}%) | 置信度: {:.0}% | 信号: {}",
                        i + 1,
                        pred.target_date,
                        pred.predicted_price,
                        pred.predicted_change_percent,
                        pred.confidence * 100.0,
                        pred.trading_signal.as_ref().unwrap_or(&"N/A".to_string())
                    );
                }
            }
            Err(e) => {
                eprintln!("❌ 预测失败: {}", e);
            }
        }
        
        println!("\n✅ {} 分析完成！", stock_code);
    }
    
    println!("\n🎉 所有股票分析完成！");
    println!("\n⚠️ 风险提示：");
    println!("   1. 本预测仅供参考，不构成投资建议");
    println!("   2. 股市有风险，投资需谨慎");
    println!("   3. 请根据自身风险承受能力做出决策");
    println!("   4. 建议设置止损位，控制风险\n");
    
    Ok(())
}
