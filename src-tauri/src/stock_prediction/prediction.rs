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
use crate::stock_prediction::technical_indicators::{get_feature_required_days, calculate_feature_value};

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

// 本地特征标准化（与训练阶段逻辑一致：按列计算 mean/std，再标准化）
fn normalize_features_local(features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if features.is_empty() { return Vec::new(); }
    let cols = features[0].len();
    let rows = features.len();
    let mut means = vec![0.0; cols];
    let mut stds = vec![0.0; cols];

    for c in 0..cols {
        let mut sum = 0.0;
        for r in 0..rows { sum += features[r][c]; }
        let mean = sum / rows as f64;
        means[c] = mean;
        let mut var_sum = 0.0;
        for r in 0..rows {
            let diff = features[r][c] - mean;
            var_sum += diff * diff;
        }
        let std = (var_sum / rows as f64).sqrt().max(1e-8);
        stds[c] = std;
    }

    let mut normalized = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            normalized[r][c] = (features[r][c] - means[c]) / stds[c];
        }
    }
    normalized
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
    
    // === 新增：与训练一致的特征计算与标准化 ===
    // 计算所需的最小历史窗口
    let required_days = metadata
        .features
        .iter()
        .map(|f| get_feature_required_days(f))
        .max()
        .unwrap_or(20);
    let lookback_window = required_days.max(30).min(prices.len().saturating_sub(1));
    let start_idx = lookback_window;
    let end_idx = prices.len() - 1;

    if start_idx > end_idx {
        return Err("历史数据不足以生成特征".to_string());
    }

    // 基于最近一段窗口，构建特征矩阵（用于标准化）
    let mut features_matrix: Vec<Vec<f64>> = Vec::with_capacity(end_idx - start_idx + 1);
    for i in start_idx..=end_idx {
        let mut feature_vector = Vec::with_capacity(metadata.features.len());
        for feature_name in &metadata.features {
            let value = calculate_feature_value(
                feature_name,
                &prices,
                &volumes,
                i,
                lookback_window,
                Some(&highs),
                Some(&lows),
            ).map_err(|e| format!("计算特征 '{}' 失败: {}", feature_name, e))?;
            feature_vector.push(value);
        }
        features_matrix.push(feature_vector);
    }

    // 标准化，并选取最新一行作为推理输入
    let normalized_matrix = normalize_features_local(&features_matrix);
    let last_normalized_row = normalized_matrix
        .last()
        .cloned()
        .ok_or_else(|| "标准化特征为空".to_string())?;

    // 创建输入张量
    let features_f32: Vec<f32> = last_normalized_row.iter().map(|&x| x as f32).collect();
    let input_tensor = Tensor::from_slice(&features_f32, &[1, metadata.features.len()], &device)
        .map_err(|e| format!("创建输入张量失败: {}", e))?;
    
    // 进行预测（基础模型输出变化率）
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
        
        // === 改进的预测算法：基于趋势分析 + 均线/量能融合 ===
        
        // 1. 基础模型预测（权重降低）
        let base_model_prediction = raw_change_rate * 0.02; // 降低基础模型权重
        
        // 2. 趋势主导因子（大幅提高权重）
        let trend_bias = trend_analysis.trend_strength * 0.015; // 趋势强度转换为日变化率
        let trend_factor = trend_bias * trend_analysis.bias_multiplier * 0.6; // 60%权重给趋势
        
        // 3. 技术指标确认（与趋势配合）
        let tech_decay = 0.92_f64.powi(day as i32);
        let technical_impact = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * 0.035 * tech_decay
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * 0.005 * tech_decay
                } else {
                    technical_signals.signal_strength * 0.015 * tech_decay
                }
            },
            TrendState::StrongBearish | TrendState::Bearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * 0.035 * tech_decay
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * 0.005 * tech_decay
                } else {
                    technical_signals.signal_strength * 0.015 * tech_decay
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    technical_signals.signal_strength * 0.025 * tech_decay
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    technical_signals.signal_strength * 0.025 * tech_decay
                } else {
                    technical_signals.signal_strength * 0.012 * tech_decay
                }
            }
        };
        
        // 4. 均线与量能偏置（新增）：MA5/10/20 与量比
        let mut ma_bias: f64 = 0.0;
        let mut vol_bias: f64 = 0.0;
        if prices.len() >= 21 && volumes.len() >= 21 {
            let n = prices.len();
            let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
            let ma5 = avg(&prices[n-5..n]);
            let ma10 = avg(&prices[n-10..n]);
            let ma20 = avg(&prices[n-20..n]);
            let price = last_price;

            // 均线位置与多空排列
            if price > ma5 { ma_bias += 0.4; } else { ma_bias -= 0.4; }
            if ma5 > ma10 { ma_bias += 0.3; } else { ma_bias -= 0.3; }
            if ma10 > ma20 { ma_bias += 0.3; } else { ma_bias -= 0.3; }

            // 均线斜率
            let prev_ma5 = avg(&prices[n-6..n-1]);
            let prev_ma10 = avg(&prices[n-11..n-1]);
            let prev_ma20 = avg(&prices[n-21..n-1]);
            if ma5 > prev_ma5 { ma_bias += 0.2; } else { ma_bias -= 0.2; }
            if ma10 > prev_ma10 { ma_bias += 0.15; } else { ma_bias -= 0.15; }
            if ma20 > prev_ma20 { ma_bias += 0.1; } else { ma_bias -= 0.1; }
            ma_bias = ma_bias.clamp(-2.0, 2.0) * 0.01; // 映射到约±1%

            // 量能偏置：5日/20日量比
            let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
            let v5 = avgv(&volumes[n-5..n]);
            let v20 = avgv(&volumes[n-20..n]);
            let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
            if vr > 1.5 { vol_bias += 0.008; }
            else if vr > 1.2 { vol_bias += 0.004; }
            if vr < 0.6 { vol_bias -= 0.008; }
            else if vr < 0.8 { vol_bias -= 0.004; }
        }
        let ma_vol_decay = 0.96_f64.powi(day as i32);
        
        // 5. 波动率调整（根据趋势一致性调整）
        let volatility_factor = historical_volatility.clamp(0.01, 0.08);
        let trend_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                if (technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBullish) ||
                   (technical_signals.macd_death_cross || technical_signals.kdj_death_cross) && 
                   matches!(trend_analysis.overall_trend, TrendState::StrongBearish) {
                    0.99_f64.powi(day as i32)
                } else {
                    0.97_f64.powi(day as i32)
                }
            },
            TrendState::Bullish | TrendState::Bearish => 0.95_f64.powi(day as i32),
            TrendState::Neutral => 0.90_f64.powi(day as i32),
        };
        
        // 6. 随机扰动（根据趋势强度和技术信号一致性调整）
        let noise_amplitude = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => {
                if (technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross || 
                    technical_signals.macd_death_cross || technical_signals.kdj_death_cross) {
                    volatility_factor * 0.6
                } else {
                    volatility_factor * 0.8
                }
            },
            TrendState::Bullish | TrendState::Bearish => volatility_factor * 1.0,
            TrendState::Neutral => volatility_factor * 1.3,
        };
        let market_noise = (rand::random::<f64>() * 2.0 - 1.0) * noise_amplitude;
        
        // 7. 综合预测变化率（调整权重分配 + 新增MA/量能项）
        let mut predicted_change_rate = base_model_prediction * 0.08
            + trend_factor * trend_decay * 0.52
            + technical_impact * 0.30
            + (ma_bias + vol_bias) * ma_vol_decay * 0.20
            + market_noise * 0.10;
        
        // 8. 趋势一致性增强（特别重视日线金叉死叉）
        match trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= 0.2; }
                    predicted_change_rate += 0.012;
                } else {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= 0.3; }
                    predicted_change_rate += 0.008;
                }
            },
            TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= 0.4; }
                    predicted_change_rate += 0.007;
                } else {
                    if predicted_change_rate < 0.0 { predicted_change_rate *= 0.6; }
                    predicted_change_rate += 0.004;
                }
            },
            TrendState::StrongBearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= 0.2; }
                    predicted_change_rate -= 0.012;
                } else {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= 0.3; }
                    predicted_change_rate -= 0.008;
                }
            },
            TrendState::Bearish => {
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= 0.4; }
                    predicted_change_rate -= 0.007;
                } else {
                    if predicted_change_rate > 0.0 { predicted_change_rate *= 0.6; }
                    predicted_change_rate -= 0.004;
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    predicted_change_rate += 0.005;
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    predicted_change_rate -= 0.005;
                }
            }
        }

        // === 新增：方向投票（优先保证涨/跌判断的合理性） ===
        let (direction_prob_up, _direction_score) = {
            let mut score: f64 = 0.0;

            // 趋势权重
            score += match trend_analysis.overall_trend {
                TrendState::StrongBullish => 2.0,
                TrendState::Bullish => 1.0,
                TrendState::Neutral => 0.0,
                TrendState::Bearish => -1.0,
                TrendState::StrongBearish => -2.0,
            };

            // MACD 权重
            if technical_signals.macd_golden_cross { score += 1.2; }
            if technical_signals.macd_death_cross { score -= 1.2; }
            if technical_signals.macd_histogram > 0.0 { score += 0.6; } else { score -= 0.6; }
            if technical_signals.macd_zero_cross_up { score += 0.8; }
            if technical_signals.macd_zero_cross_down { score -= 0.8; }

            // KDJ 权重
            if technical_signals.kdj_golden_cross { score += 0.8; }
            if technical_signals.kdj_death_cross { score -= 0.8; }
            if technical_signals.kdj_j > 80.0 { score -= 0.6; }
            if technical_signals.kdj_j < 20.0 { score += 0.6; }

            // RSI 权重
            if technical_signals.rsi > 70.0 { score += 0.8; }
            else if technical_signals.rsi > 55.0 { score += 0.5; }
            else if technical_signals.rsi < 30.0 { score -= 0.8; }
            else if technical_signals.rsi < 45.0 { score -= 0.5; }

            // 均线排列与斜率
            if prices.len() >= 21 {
                let n = prices.len();
                let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
                let ma5 = avg(&prices[n-5..n]);
                let ma10 = avg(&prices[n-10..n]);
                let ma20 = avg(&prices[n-20..n]);
                if ma5 > ma10 && ma10 > ma20 { score += 1.0; }
                if ma5 < ma10 && ma10 < ma20 { score -= 1.0; }
                let prev_ma5 = avg(&prices[n-6..n-1]);
                if ma5 > prev_ma5 { score += 0.3; } else { score -= 0.3; }
            }

            // 量比与 OBV
            if volumes.len() >= 20 {
                let n = volumes.len();
                let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
                let v5 = avgv(&volumes[n-5..n]);
                let v20 = avgv(&volumes[n-20..n]);
                let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
                if vr > 1.2 { score += 0.3; }
                if vr < 0.8 { score -= 0.3; }
            }
            if technical_signals.obv > 0.0 { score += 0.2; } else { score -= 0.2; }

            let k = 0.9_f64; // 温和的放大系数
            let prob_up = 1.0 / (1.0 + (-k * score).exp());
            (prob_up, score)
        };

        // 根据方向概率调整预测方向，并设定保守幅度（更关注方向正确性）
        if direction_prob_up >= 0.55 && predicted_change_rate < 0.0 {
            predicted_change_rate = predicted_change_rate.abs();
        }
        if direction_prob_up <= 0.45 && predicted_change_rate > 0.0 {
            predicted_change_rate = -predicted_change_rate.abs();
        }
        // 使用基于波动率与趋势置信的保守幅度（不追求幅度精确）
        let dir_mag = (volatility_factor * (0.6 + 0.4 * trend_analysis.trend_confidence) * (0.98_f64.powi((day as i32) - 1)))
            .clamp(0.003, 0.06);
        if predicted_change_rate == 0.0 {
            predicted_change_rate = if direction_prob_up >= 0.5 { dir_mag } else { -dir_mag };
        } else {
            predicted_change_rate = predicted_change_rate.signum() * dir_mag;
        }

        // 9. 应用A股涨跌停限制
        let change_percent = clamp_daily_change(predicted_change_rate * 100.0);
        let clamped_change_rate = change_percent / 100.0;
        let predicted_price = last_price * (1.0 + clamped_change_rate);
        
        // 10. 置信度（趋势一致性 + MA/量能增强）
        let base_confidence = (metadata.accuracy + 0.3).min(0.8);
        let trend_confidence_boost = trend_analysis.trend_confidence * 0.2;
        let volatility_impact = 1.0 - (volatility_factor * 6.0).min(0.3);
        let prediction_magnitude = 1.0 - (change_percent.abs() / 12.0).min(0.25);
        let time_decay = match trend_analysis.overall_trend {
            TrendState::StrongBullish | TrendState::StrongBearish => 0.97_f64.powi(day as i32),
            _ => 0.94_f64.powi(day as i32),
        };
        
        // MA排列契合及量比对置信度的贡献
        let mut confidence_extra = 0.0;
        if prices.len() >= 21 && volumes.len() >= 21 {
            let n = prices.len();
            let avg = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
            let ma5 = avg(&prices[n-5..n]);
            let ma10 = avg(&prices[n-10..n]);
            let ma20 = avg(&prices[n-20..n]);
            if ma5 > ma10 && ma10 > ma20 { confidence_extra += 0.03; }
            if ma5 < ma10 && ma10 < ma20 { confidence_extra += 0.03; }
            let avgv = |slice: &[i64]| slice.iter().map(|&v| v as f64).sum::<f64>() / slice.len() as f64;
            let v5 = avgv(&volumes[n-5..n]);
            let v20 = avgv(&volumes[n-20..n]);
            let vr = if v20 > 0.0 { v5 / v20 } else { 1.0 };
            if vr > 1.2 { confidence_extra += 0.02; }
        }
        
        let confidence = (base_confidence 
            * volatility_impact 
            * prediction_magnitude 
            * time_decay 
            + trend_confidence_boost
            + confidence_extra)
            .clamp(0.35, 0.92);
        
        // 11. 交易信号（结合趋势状态和日线技术指标）
        let trading_signal_str = match &trend_analysis.overall_trend {
            TrendState::StrongBullish => {
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "强烈买入"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "强烈买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有"
                } else {
                    "买入"
                }
            },
            TrendState::Bullish => {
                if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "持有"
                } else {
                    "买入"
                }
            },
            TrendState::Neutral => {
                if technical_signals.macd_golden_cross && technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "买入"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出"
                } else {
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
                if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有"
                } else {
                    "卖出"
                }
            },
            TrendState::StrongBearish => {
                if technical_signals.macd_death_cross && technical_signals.kdj_death_cross {
                    "强烈卖出"
                } else if technical_signals.macd_death_cross || technical_signals.kdj_death_cross {
                    "强烈卖出"
                } else if technical_signals.macd_golden_cross || technical_signals.kdj_golden_cross {
                    "持有"
                } else {
                    "卖出"
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