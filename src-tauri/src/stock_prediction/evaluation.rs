use crate::stock_prediction::types::EvaluationResult;
use crate::stock_prediction::database::get_historical_data_from_db;
use crate::stock_prediction::utils::calculate_direction_focused_accuracy;
use crate::stock_prediction::model_management::{list_models, get_model_file_path};
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarMap};
use crate::stock_prediction::types::ModelConfig;

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

// 模型评估函数
pub async fn evaluate_candle_model(stock_code: String, model_name: Option<String>) -> std::result::Result<EvaluationResult, String> {
    // 获取模型列表
    let model_list = list_models(&stock_code);
    
    if model_list.is_empty() {
        return Err("没有找到可用的模型".to_string());
    }
    
    // 获取模型元数据
    let metadata = if let Some(model_name) = &model_name {
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
        dropout: 0.0, // 评估时不使用dropout
        learning_rate: 0.001,
        n_layers: 2,
        n_heads: 4,
        max_seq_len: 60,
    };
    
    let mut varmap = VarMap::new();
    
    let (_, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {}", e))?;
    
    let model_path = get_model_file_path(&metadata.id);
    if !model_path.exists() {
        return Err(format!("模型文件不存在: {:?}", model_path));
    }
    
    varmap.load(&model_path).map_err(|e| format!("模型加载失败: {}", e))?;
    
    // 获取测试数据
    let end_date = chrono::Local::now().naive_local().date();
    let start_date = end_date - chrono::Duration::days(60); // 获取最近60天数据
    
    let start_date_str = start_date.format("%Y-%m-%d").to_string();
    let end_date_str = end_date.format("%Y-%m-%d").to_string();
    
    println!("评估数据范围: {} 到 {}", start_date_str, end_date_str);
    
    // 从数据库获取历史数据
    let historical_data = get_historical_data_from_db(&stock_code, &start_date_str, &end_date_str).await
        .map_err(|e| format!("获取历史数据失败: {}", e))?;
    
    if historical_data.len() < 30 {
        return Err("评估数据不足，需要至少30天数据".to_string());
    }
    
    // 按日期排序（从旧到新）
    let mut sorted_data = historical_data;
    sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    // 准备评估数据
    let prices: Vec<f64> = sorted_data.iter().map(|d| d.close).collect();
    let volumes: Vec<i64> = sorted_data.iter().map(|d| d.volume).collect();
    
    let mut actual_values = Vec::new();
    let mut predicted_values = Vec::new();
    
    // 为每个数据点计算预测值
    let lookback_window = 20; // 使用20天回看窗口
    
    for i in lookback_window..prices.len()-1 {
        // 准备特征向量
        let mut features = Vec::new();
        
        // 为每个特征计算值
        for feature_name in &metadata.features {
            match feature_name.as_str() {
                "close" => {
                    // 归一化收盘价
                    let window_prices = &prices[i-lookback_window+1..=i];
                    let price_min = window_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let price_max = window_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let price_range = price_max - price_min;
                    let normalized = if price_range > 0.0 {
                        (prices[i] - price_min) / price_range
                    } else {
                        0.5
                    };
                    features.push(normalized);
                },
                "volume" => {
                    // 归一化成交量
                    let window_volumes = &volumes[i-lookback_window+1..=i];
                    let vol_min = window_volumes.iter().fold(i64::MAX, |a, &b| a.min(b));
                    let vol_max = window_volumes.iter().fold(i64::MIN, |a, &b| a.max(b));
                    let vol_range = (vol_max - vol_min) as f64;
                    let normalized = if vol_range > 0.0 {
                        (volumes[i] - vol_min) as f64 / vol_range
                    } else {
                        0.5
                    };
                    features.push(normalized);
                },
                "change_percent" => {
                    // 计算日变化率
                    if i > 0 {
                        let change_rate = (prices[i] - prices[i-1]) / prices[i-1];
                        let normalized = (change_rate / 0.1).clamp(-1.0, 1.0); // 假设正常变化率在±10%内
                        features.push(normalized);
                    } else {
                        features.push(0.0);
                    }
                },
                "ma5" => {
                    // 5日移动平均线
                    if i >= 5 {
                        let ma5 = prices[i-4..=i].iter().sum::<f64>() / 5.0;
                        let normalized = (ma5 - prices[i]) / prices[i];
                        features.push(normalized);
                    } else {
                        features.push(0.0);
                    }
                },
                "ma10" => {
                    // 10日移动平均线
                    if i >= 10 {
                        let ma10 = prices[i-9..=i].iter().sum::<f64>() / 10.0;
                        let normalized = (ma10 - prices[i]) / prices[i];
                        features.push(normalized);
                    } else {
                        features.push(0.0);
                    }
                },
                "ma20" => {
                    // 20日移动平均线
                    if i >= 20 {
                        let ma20 = prices[i-19..=i].iter().sum::<f64>() / 20.0;
                        let normalized = (ma20 - prices[i]) / prices[i];
                        features.push(normalized);
                    } else {
                        features.push(0.0);
                    }
                },
                _ => {
                    // 其他特征，简化处理
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
        
        // 获取预测结果
        let predicted_change_rate = match output.dims() {
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
        
        // 计算实际变化率
        let actual_change_rate = (prices[i+1] - prices[i]) / prices[i];
        
        predicted_values.push(predicted_change_rate);
        actual_values.push(actual_change_rate);
    }
    
    if predicted_values.is_empty() {
        return Err("无法生成评估预测，数据不足".to_string());
    }
    
    // 计算评估指标
    let (direction_accuracy, combined_accuracy) = calculate_direction_focused_accuracy(&predicted_values, &actual_values);
    
    // 计算MSE和MAE
    let mse = actual_values.iter()
        .zip(predicted_values.iter())
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>() / actual_values.len() as f64;
    
    let mae = actual_values.iter()
        .zip(predicted_values.iter())
        .map(|(actual, predicted)| (actual - predicted).abs())
        .sum::<f64>() / actual_values.len() as f64;
    
    let rmse = mse.sqrt();
    
    // 计算方向预测准确率
    let correct_directions = actual_values.iter()
        .zip(predicted_values.iter())
        .filter(|(&actual, &predicted)| {
            (actual > 0.0 && predicted > 0.0) || (actual < 0.0 && predicted < 0.0) || (actual == 0.0 && predicted.abs() < 0.001)
        })
        .count();
    
    let direction_accuracy_simple = correct_directions as f64 / actual_values.len() as f64;
    
    println!("评估结果:");
    println!("  评估样本数: {}", actual_values.len());
    println!("  方向预测准确率: {:.2}%", direction_accuracy * 100.0);
    println!("  综合准确率: {:.2}%", combined_accuracy * 100.0);
    println!("  简单方向准确率: {:.2}%", direction_accuracy_simple * 100.0);
    println!("  均方误差 (MSE): {:.6}", mse);
    println!("  平均绝对误差 (MAE): {:.6}", mae);
    println!("  均方根误差 (RMSE): {:.6}", rmse);
    
    Ok(EvaluationResult {
        model_id: metadata.id.clone(),
        model_name: metadata.name.clone(),
        stock_code: stock_code.clone(),
        test_samples: actual_values.len(),
        accuracy: combined_accuracy,
        direction_accuracy,
        mse,
        mae,
        rmse,
        evaluation_date: chrono::Local::now().naive_local().date().format("%Y-%m-%d").to_string(),
    })
} 