use candle_core::{Device, Tensor};
use candle_nn::{Module, VarMap};
use rand;
use chrono;
use crate::stock_prediction::types::{
    ModelConfig, Prediction, TechnicalIndicatorValues, PredictionRequest, 
    PredictionResponse, LastRealData, TradingSignal
};
use crate::stock_prediction::model_management::{list_models, get_model_file_path};
use crate::stock_prediction::database::get_recent_market_data;
use crate::stock_prediction::utils::{
    get_next_trading_day, clamp_daily_change, calculate_historical_volatility,
    calculate_recent_trend, calculate_support_resistance, analyze_historical_volatility_pattern,
    print_last_real_vs_prediction, analyze_stock_trend, TrendState, predict_with_volume_price, 
    calculate_volume_price_change, VolumePricePredictionStrategy
};
use crate::stock_prediction::technical_analysis::analyze_technical_signals;

// 简化的模型创建函数（与training.rs中的相同，用于加载模型）
fn create_model(config: &ModelConfig, device: &Device) -> Result<(VarMap, Box<dyn Module + Send + Sync>), candle_core::Error> {
    // 创建一个简单的线性回归模型
    let varmap = VarMap::new();
    
    struct LinearRegression {
        linear: candle_nn::Linear,
    }
    
    impl LinearRegression {
        fn new(in_size: usize, out_size: usize, vb: candle_nn::VarBuilder) -> Result<Self, candle_core::Error> {
            let linear = candle_nn::linear(in_size, out_size, vb)?;
            Ok(Self { linear })
        }
    }
    
    unsafe impl Send for LinearRegression {}
    unsafe impl Sync for LinearRegression {}
    
    impl Module for LinearRegression {
        fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
            self.linear.forward(xs)
        }
    }
    
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    let model = LinearRegression::new(config.input_size, config.output_size, vb)?;
    
    let model: Box<dyn Module + Send + Sync> = Box::new(model);
    
    Ok((varmap, model))
}

// 股票预测函数 - 基于趋势分析的改进版本
pub async fn predict_with_candle(request: PredictionRequest) -> std::result::Result<PredictionResponse, String> {
    let model_list = list_models(&request.stock_code);
    
    if model_list.is_empty() {
        return Err("没有找到可用的模型".to_string());
    }
    
    // 获取模型元数据
    let metadata = if let Some(model_name) = &request.model_name {
        model_list.iter()
            .find(|m| m.name == *model_name)
            .ok_or_else(|| format!("找不到名为 {} 的模型", model_name))?
            .clone()
    } else {
        // 如果没有指定模型名称，使用最新的模型
        model_list[0].clone()
    };
    
    // 加载模型
    let device = Device::Cpu;
    
    let config = ModelConfig {
        model_type: metadata.model_type.clone(),
        input_size: metadata.features.len(),
        hidden_size: 64,
        output_size: 1,
        dropout: 0.0,
        learning_rate: 0.001,
        n_layers: 2,
        n_heads: 4,
        max_seq_len: 60,
    };
    
    let mut varmap = VarMap::new();
    
    let (_, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {}", e))?;
    
    let model_path = get_model_file_path(&metadata.id);
    varmap.load(&model_path).map_err(|e| format!("模型加载失败: {}", e))?;
    
    // 获取最近的真实市场数据
    let (current_price, current_change_percent, dates, mut prices, mut volumes, mut highs, mut lows) = get_recent_market_data(&request.stock_code, 60).await
        .map_err(|e| format!("获取市场数据失败: {}", e))?;
    
    if prices.len() < 20 {
        return Err("历史数据不足，无法进行预测，需要至少20天数据".to_string());
    }
    
    // === 新增：趋势分析 ===
    let trend_analysis = analyze_stock_trend(&prices, &highs, &lows, &volumes);
    
    println!("🔮 基于趋势分析进行预测:");
    println!("   📈 当前趋势: {:?}", trend_analysis.overall_trend);
    println!("   🎯 趋势强度: {:.2}", trend_analysis.trend_strength);
    println!("   🔒 趋势置信度: {:.0}%", trend_analysis.trend_confidence * 100.0);
    println!("   ⚖️  预测偏向倍数: {:.2}", trend_analysis.bias_multiplier);
    
    // 计算特征向量
    let mut features = Vec::new();
    let last_idx = prices.len() - 1;
    
    // 为每个特征计算值
    for feature_name in &metadata.features {
        match feature_name.as_str() {
            "close" => {
                let price_min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let price_max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let price_range = price_max - price_min;
                let normalized = if price_range > 0.0 {
                    (current_price - price_min) / price_range
                } else {
                    0.5
                };
                features.push(normalized);
            },
            "volume" => {
                let latest_volume = volumes[last_idx];
                let vol_min = volumes.iter().fold(i64::MAX, |a, &b| a.min(b));
                let vol_max = volumes.iter().fold(i64::MIN, |a, &b| a.max(b));
                let vol_range = (vol_max - vol_min) as f64;
                let normalized = if vol_range > 0.0 {
                    (latest_volume - vol_min) as f64 / vol_range
                } else {
                    0.5
                };
                features.push(normalized);
            },
            _ => {
                features.push(0.0);
            }
        }
    }
    
    // 创建输入张量
    let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
    let input_tensor = Tensor::from_slice(&features_f32, &[1, features.len()], &device)
        .map_err(|e| format!("创建输入张量失败: {}", e))?;
    
    // 进行预测
    let output = model.forward(&input_tensor)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    let raw_change_rate = match output.dims() {
        [_] => {
            output.to_vec1::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0] as f64
        },
        [_, n] => {
            if *n == 1 {
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0][0] as f64
            } else {
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0][0] as f64
            }
        },
        _ => {
            return Err(format!("预测输出维度不支持: {:?}", output.dims()));
        }
    };
    
    // 计算历史数据特征
    let historical_volatility = calculate_historical_volatility(&prices);
    let _recent_trend = calculate_recent_trend(&prices);
    let _support_resistance = calculate_support_resistance(&prices, current_price);
    let _volatility_features = analyze_historical_volatility_pattern(&prices, 30);
    let mut technical_signals = analyze_technical_signals(&prices, &highs, &lows, &volumes);
    
    // 生成预测
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(&dates.last().unwrap_or(&"2023-01-01".to_string()), "%Y-%m-%d")
        .unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    for day in 1..=request.prediction_days {
        // 确保预测日期为A股交易日
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // === 改进的预测算法：基于趋势分析 ===
        
        // 1. 基础模型预测（权重降低）
        let base_model_prediction = raw_change_rate * 0.02; // 降低基础模型权重
        
        // 2. 趋势主导因子（大幅提高权重）
        let trend_bias = trend_analysis.trend_strength * 0.015; // 趋势强度转换为日变化率
        let trend_factor = trend_bias * trend_analysis.bias_multiplier * 0.6; // 60%权重给趋势
        
        // 3. 技术指标确认（与趋势配合）
        let tech_decay = 0.92_f64.powi(day as i32);
        let technical_impact = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::Bullish => {
                // 涨势中，优先看日线金叉信号，加大权重
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    // 日线金叉时大幅加强预测上涨
                    technical_signals.signal_strength * 0.035 * tech_decay
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    // 涨势中出现死叉，降低影响（可能是短期回调）
                    technical_signals.signal_strength * 0.005 * tech_decay
                } else {
                    // 一般技术信号
                    technical_signals.signal_strength * 0.015 * tech_decay
                }
            },
            TrendState::StrongBearish | TrendState::Bearish => {
                // 跌势中，优先看日线死叉信号，加大权重
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    // 日线死叉时大幅加强预测下跌
                    technical_signals.signal_strength * 0.035 * tech_decay
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    // 跌势中出现金叉，降低影响（可能是短期反弹）
                    technical_signals.signal_strength * 0.005 * tech_decay
                } else {
                    // 一般技术信号
                    technical_signals.signal_strength * 0.015 * tech_decay
                }
            },
            TrendState::Neutral => {
                // 中性趋势，技术指标权重正常，但更重视金叉死叉
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * 0.025 * tech_decay
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * 0.025 * tech_decay
                } else {
                    technical_signals.signal_strength * 0.012 * tech_decay
                }
            }
        };
        
        // 4. 波动率调整（根据趋势一致性调整）
        let volatility_factor = historical_volatility.clamp(0.01, 0.08);
        let trend_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                // 强趋势时衰减更慢，特别是有日线金叉死叉确认时
                if (technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBullish) ||
                   (technical_signals.macd_death_cross || technical_signals.kdj_death_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBearish) {
                    0.99_f64.powi(day as i32) // 技术信号确认时衰减最慢
                } else {
                    0.97_f64.powi(day as i32)
                }
            },
            TrendState::Bullish | TrendState::Bearish => 0.95_f64.powi(day as i32),
            TrendState::Neutral => 0.90_f64.powi(day as i32),
        };
        
        // 5. 随机扰动（根据趋势强度和技术信号一致性调整）
        let noise_amplitude = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                // 强趋势且有技术信号确认时，降低随机性
                if (technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross || 
                    technical_signals.macd_death_cross || technical_signals.kdj_death_cross) {
                    volatility_factor * 0.6 // 技术信号确认时随机性最低
                } else {
                    volatility_factor * 0.8
                }
            },
            TrendState::Bullish | TrendState::Bearish => volatility_factor * 1.0,
            TrendState::Neutral => volatility_factor * 1.3,
        };
        let market_noise = (rand::random::<f64>() * 2.0 - 1.0) * noise_amplitude;
        
        // 6. 综合预测变化率（调整权重分配）
        let mut predicted_change_rate = base_model_prediction * 0.08  // 基础模型 8%（进一步降低）
            + trend_factor * trend_decay * 0.52                       // 趋势因子 52%
            + technical_impact * 0.30                                 // 技术指标 30%（提高，特别是金叉死叉）
            + market_noise * 0.10;                                   // 随机扰动 10%
        
        // 7. 趋势一致性增强（特别重视日线金叉死叉）
        match trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                // 强涨势：如果有日线金叉，大幅增强上涨预测
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 {
                        predicted_change_rate *= 0.2; // 大幅减少下跌预测
                    }
                    predicted_change_rate += 0.012; // 大幅增加上涨基础
                } else {
                    if predicted_change_rate < 0.0 {
                        predicted_change_rate *= 0.3;
                    }
                    predicted_change_rate += 0.008;
                }
            },
            TrendState::Bullish => {
                // 涨势：日线金叉时增强预测
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 {
                        predicted_change_rate *= 0.4;
                    }
                    predicted_change_rate += 0.007;
                } else {
                    if predicted_change_rate < 0.0 {
                        predicted_change_rate *= 0.6;
                    }
                    predicted_change_rate += 0.004;
                }
            },
            TrendState::StrongBearish => {
                // 强跌势：如果有日线死叉，大幅增强下跌预测
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 {
                        predicted_change_rate *= 0.2; // 大幅减少上涨预测
                    }
                    predicted_change_rate -= 0.012; // 大幅增加下跌基础
                } else {
                    if predicted_change_rate > 0.0 {
                        predicted_change_rate *= 0.3;
                    }
                    predicted_change_rate -= 0.008;
                }
            },
            TrendState::Bearish => {
                // 跌势：日线死叉时增强预测
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 {
                        predicted_change_rate *= 0.4;
                    }
                    predicted_change_rate -= 0.007;
                } else {
                    if predicted_change_rate > 0.0 {
                        predicted_change_rate *= 0.6;
                    }
                    predicted_change_rate -= 0.004;
                }
            },
            TrendState::Neutral => {
                // 中性：重视金叉死叉信号的方向性
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    predicted_change_rate += 0.005; // 轻微偏向上涨
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    predicted_change_rate -= 0.005; // 轻微偏向下跌
                }
            }
        }
        
        // 8. 应用A股涨跌停限制
        let change_percent = clamp_daily_change(predicted_change_rate * 100.0);
        let clamped_change_rate = change_percent / 100.0;
        let predicted_price = last_price * (1.0 + clamped_change_rate);
        
        // 9. 计算置信度（基于趋势一致性）
        let base_confidence = (metadata.accuracy + 0.3).min(0.8);
        let trend_confidence_boost = trend_analysis.trend_confidence * 0.2; // 趋势置信度加成
        let volatility_impact = 1.0 - (volatility_factor * 6.0).min(0.3);
        let prediction_magnitude = 1.0 - (change_percent.abs() / 12.0).min(0.25);
        let time_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => 0.97_f64.powi(day as i32),
            _ => 0.94_f64.powi(day as i32),
        };
        
        let confidence = (base_confidence 
            * volatility_impact 
            * prediction_magnitude 
            * time_decay 
            + trend_confidence_boost)
            .clamp(0.35, 0.92);
        
        // 10. 交易信号（结合趋势状态和日线技术指标）
        let trading_signal_str = match &trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                // 强涨势：日线金叉时给出最强烈信号
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "强烈买入" // 双金叉确认
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "强烈买入" // 单个金叉也给强烈信号
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有" // 强涨势中出现死叉，降级为持有
                } else {
                    "买入" // 强涨势默认买入
                }
            },
            TrendState::Bullish => {
                // 涨势：日线金叉时增强信号
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入" // 金叉确认涨势
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有" // 死叉时谨慎
                } else {
                    "买入" // 涨势默认买入
                }
            },
            TrendState::Neutral => {
                // 中性：完全依据日线技术指标
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "买入" // 双金叉时看涨
                } else if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "卖出" // 双死叉时看跌
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入" // 单金叉偏多
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出" // 单死叉偏空
                } else {
                    // 无明确金叉死叉，看其他技术指标
                    match &technical_signals.signal {
                        TradingSignal::StrongBuy => "买入",
                        TradingSignal::Buy => "买入",
                        TradingSignal::Hold => "持有",
                        TradingSignal::Sell => "卖出", 
                        TradingSignal::StrongSell => "卖出",
                    }
                }
            },
            TrendState::Bearish => {
                // 跌势：日线死叉时增强信号
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出" // 死叉确认跌势
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有" // 金叉时谨慎
                } else {
                    "卖出" // 跌势默认卖出
                }
            },
            TrendState::StrongBearish => {
                // 强跌势：日线死叉时给出最强烈信号
                if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "强烈卖出" // 双死叉确认
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "强烈卖出" // 单个死叉也给强烈信号
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有" // 强跌势中出现金叉，降级为持有
                } else {
                    "卖出" // 强跌势默认卖出
                }
            },
        };
        
        let technical_indicators = TechnicalIndicatorValues {
            rsi: technical_signals.rsi,
            macd_histogram: technical_signals.macd_histogram,
            kdj_j: technical_signals.kdj_j,
            cci: technical_signals.cci,
            obv_trend: if technical_signals.obv > 0.0 { 1.0 } else { -1.0 },
            macd_dif: technical_signals.macd_dif,
            macd_dea: technical_signals.macd_dea,
            kdj_k: technical_signals.kdj_k,
            kdj_d: technical_signals.kdj_d,
            macd_golden_cross: technical_signals.macd_golden_cross,
            macd_death_cross: technical_signals.macd_death_cross,
            kdj_golden_cross: technical_signals.kdj_golden_cross,
            kdj_death_cross: technical_signals.kdj_death_cross,
            kdj_overbought: technical_signals.kdj_overbought,
            kdj_oversold: technical_signals.kdj_oversold,
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(trading_signal_str.to_string()),
            signal_strength: Some(trend_analysis.trend_strength),
            technical_indicators: Some(technical_indicators),
        });
        
        last_price = predicted_price;
        
        // 更新价格序列以便下一天预测
        if day < request.prediction_days {
            prices.push(predicted_price);
            highs.push(predicted_price * 1.005);
            lows.push(predicted_price * 0.995);
            
            if let Some(&last_volume) = volumes.last() {
                let volume_change = 0.8 + rand::random::<f64>() * 0.4;
                volumes.push((last_volume as f64 * volume_change) as i64);
            }
            
            technical_signals = analyze_technical_signals(&prices, &highs, &lows, &volumes);
        }
    }
    
    // 构建最新真实数据
    let last_real_data = if !dates.is_empty() {
        Some(LastRealData {
            date: dates.last().unwrap().clone(),
            price: current_price,
            change_percent: current_change_percent,
        })
    } else {
        None
    };
    
    // 打印对比信息
    if !predictions.is_empty() {
        print_last_real_vs_prediction(&dates, &prices, &technical_signals, &predictions[0], current_change_percent);
    }
    
    println!("\n🎯 预测完成! 基于{}趋势进行了{}天预测", 
             match trend_analysis.overall_trend {
                 TrendState::StrongBullish => "强烈上涨",
                 TrendState::Bullish => "上涨",
                 TrendState::Neutral => "震荡",
                 TrendState::Bearish => "下跌",
                 TrendState::StrongBearish => "强烈下跌",
             },
             request.prediction_days);
    
    Ok(PredictionResponse {
        predictions,
        last_real_data,
    })
}

// 基于量价关系的预测函数 - 专注于核心要素
pub async fn predict_with_simple_strategy(request: PredictionRequest) -> std::result::Result<PredictionResponse, String> {
    // 获取最近的真实市场数据
    let (current_price, current_change_percent, dates, prices, volumes, highs, lows) = get_recent_market_data(&request.stock_code, 60).await
        .map_err(|e| format!("获取市场数据失败: {}", e))?;
    
    if prices.len() < 10 {
        return Err("历史数据不足，无法进行预测，需要至少10天数据".to_string());
    }
    
    println!("🎯 使用量价关系策略进行预测:");
    println!("   📊 历史数据: {}天", prices.len());
    println!("   💰 当前价格: {:.2}元", current_price);
    
    // 使用量价关系预测策略
    let volume_price_strategy = predict_with_volume_price(&prices, &highs, &lows, &volumes, current_price);
    
    // 计算历史波动率
    let historical_volatility = calculate_historical_volatility(&prices);
    
    // 生成预测
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut last_price = current_price;
    
    let last_date = chrono::NaiveDate::parse_from_str(&dates.last().unwrap_or(&"2023-01-01".to_string()), "%Y-%m-%d")
        .unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    // 为每一天生成预测
    for day in 1..=request.prediction_days {
        // 确保预测日期为A股交易日
        let mut target_date = last_date;
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 使用量价关系计算涨跌幅
        let (predicted_change_percent, confidence) = calculate_volume_price_change(&volume_price_strategy, historical_volatility, day as i32);
        let predicted_price = last_price * (1.0 + predicted_change_percent / 100.0);
        
        // 交易信号直接来自量价策略
        let trading_signal = &volume_price_strategy.volume_price_signal;
        
        // 创建技术指标值（基于量价关系）
        let technical_indicators = TechnicalIndicatorValues {
            rsi: 50.0,
            macd_histogram: 0.0,
            kdj_j: 50.0,
            cci: 0.0,
            obv_trend: 0.0,
            macd_dif: 0.0,
            macd_dea: 0.0,
            kdj_k: 50.0,
            kdj_d: 50.0,
            macd_golden_cross: volume_price_strategy.predicted_direction == "上涨",
            macd_death_cross: volume_price_strategy.predicted_direction == "下跌",
            kdj_golden_cross: volume_price_strategy.predicted_direction == "上涨",
            kdj_death_cross: volume_price_strategy.predicted_direction == "下跌",
            kdj_overbought: false,
            kdj_oversold: false,
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent,
            confidence,
            trading_signal: Some(trading_signal.clone()),
            signal_strength: Some(volume_price_strategy.direction_confidence),
            technical_indicators: Some(technical_indicators),
        });
        
        last_price = predicted_price;
    }
    
    // 构建最新真实数据
    let last_real_data = if !dates.is_empty() {
        Some(LastRealData {
            date: dates.last().unwrap().clone(),
            price: current_price,
            change_percent: current_change_percent,
        })
    } else {
        None
    };
    
    println!("\n✅ 量价关系预测完成!");
    println!("   🎯 预测方向: {} (置信度: {:.0}%)", 
             volume_price_strategy.predicted_direction, 
             volume_price_strategy.direction_confidence * 100.0);
    println!("   📊 价格趋势: {}", volume_price_strategy.price_trend);
    println!("   📊 成交量趋势: {}", volume_price_strategy.volume_trend);
    println!("   💡 量价信号: {}", volume_price_strategy.volume_price_signal);
    println!("   📋 关键因素: {}", volume_price_strategy.key_factors.join(", "));
    println!("   📈 预测天数: {}天", request.prediction_days);
    
    Ok(PredictionResponse {
        predictions,
        last_real_data,
    })
} 