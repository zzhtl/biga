use crate::models::{ModelConfig, create_model, save_model};
use candle_core::{Device, Tensor, DType};
use candle_nn::{Module, Optimizer, VarBuilder, VarMap, AdamW};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use rand;
use chrono;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String, 
    pub stock_code: String,
    pub created_at: u64,
    pub model_type: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub target_date: String,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRequest {
    pub stock_code: String,
    pub model_name: String,
    pub start_date: String,
    pub end_date: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: usize,
    pub model_type: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub dropout: f64,
    pub train_test_split: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub prediction_days: usize,
    pub use_candle: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metadata: ModelInfo,
    pub accuracy: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub accuracy: f64,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub prediction_examples: Vec<PredictionExample>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionExample {
    pub actual: f64,
    pub predicted: f64,
    pub features: Vec<f64>,
}

// 定义获取模型保存目录的函数
fn get_models_dir() -> PathBuf {
    let app_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from("./data"));
    let models_dir = app_dir.join("biga/models");
    fs::create_dir_all(&models_dir).unwrap_or_default();
    models_dir
}

// 定义获取特定模型目录的函数
fn get_model_dir(model_id: &str) -> PathBuf {
    get_models_dir().join(model_id)
}

// 保存模型元数据
fn save_model_metadata(metadata: &ModelInfo) -> std::io::Result<()> {
    let model_dir = get_model_dir(&metadata.id);
    fs::create_dir_all(&model_dir)?;
    
    let metadata_path = model_dir.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(metadata)?;
    fs::write(metadata_path, metadata_json)?;
    
    Ok(())
}

// 读取模型元数据
fn load_model_metadata(model_id: &str) -> std::io::Result<ModelInfo> {
    let metadata_path = get_model_dir(model_id).join("metadata.json");
    let metadata_json = fs::read_to_string(metadata_path)?;
    let metadata: ModelInfo = serde_json::from_str(&metadata_json)?;
    Ok(metadata)
}

// 列出特定股票代码的所有模型
pub fn list_models(symbol: &str) -> Vec<ModelInfo> {
    let models_dir = get_models_dir();
    
    let mut models = Vec::new();
    
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            
            if path.is_dir() {
                let metadata_path = path.join("metadata.json");
                
                if metadata_path.exists() {
                    if let Ok(metadata_json) = fs::read_to_string(metadata_path) {
                        if let Ok(metadata) = serde_json::from_str::<ModelInfo>(&metadata_json) {
                            if metadata.stock_code == symbol {
                                models.push(metadata);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 按创建时间降序排序
    models.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    models
}

// 删除模型
pub fn delete_model(model_id: &str) -> std::io::Result<()> {
    let model_dir = get_model_dir(model_id);
    fs::remove_dir_all(model_dir)?;
    Ok(())
}

// 股票预测数据准备函数
fn prepare_stock_data(
    request: &TrainingRequest,
) -> std::result::Result<(Tensor, Tensor, Tensor, Tensor, Vec<String>), candle_core::Error> {
    // 这里应该是从数据库或API获取股票数据
    // 为简化示例，我们将通过简单的模拟数据展示如何进行特征工程

    // 构建特征和标签
    let device = Device::Cpu;
    
    // 模拟获取历史股票数据
    // 实际应用中，应该从数据库查询
    let mut dates = Vec::new();
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    let mut features_matrix = Vec::new();
    
    // 模拟180天的历史数据
    let days = 180;
    let mut price = 100.0;
    let base_date = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    
    for i in 0..days {
        // 生成日期
        let date = base_date + chrono::Duration::days(i);
        dates.push(date.format("%Y-%m-%d").to_string());
        
        // 生成价格(有一定的随机波动和趋势)
        let trend = ((i as f64 / 30.0).sin() * 10.0) + ((i as f64 / 90.0).cos() * 5.0);
        let random = (rand::random::<f64>() - 0.5) * 4.0;
        price = price + trend * 0.01 + random;
        prices.push(price);
        
        // 生成成交量
        let volume = (1000000.0 * (1.0 + (rand::random::<f64>() - 0.5) * 0.3)) as i64;
        volumes.push(volume);
        
        // 为每天准备一个特征向量
        if i >= 20 {  // 需要前20天数据计算某些指标
            let mut feature_vector = Vec::new();
            
            // 收集请求中的特征
            for feature_name in &request.features {
                match feature_name.as_str() {
                    "close" => {
                        // 归一化收盘价
                        let normalized = (price - 95.0) / 20.0; // 假设价格范围在95-115之间
                        feature_vector.push(normalized);
                    },
                    "volume" => {
                        // 归一化成交量
                        let normalized = volume as f64 / 2000000.0;
                        feature_vector.push(normalized);
                    },
                    "change_percent" => {
                        // 计算价格变化百分比
                        let prev_price = prices[prices.len() - 2];
                        let change = (price - prev_price) / prev_price;
                        feature_vector.push(change);
                    },
                    "ma5" => {
                        // 5日移动平均线
                        let ma5 = prices[prices.len()-5..].iter().sum::<f64>() / 5.0;
                        let normalized = (ma5 - price) / price;
                        feature_vector.push(normalized);
                    },
                    "ma10" => {
                        // 10日移动平均线
                        let ma10 = prices[prices.len()-10..].iter().sum::<f64>() / 10.0;
                        let normalized = (ma10 - price) / price;
                        feature_vector.push(normalized);
                    },
                    "ma20" => {
                        // 20日移动平均线
                        let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                        let normalized = (ma20 - price) / price;
                        feature_vector.push(normalized);
                    },
                    "rsi" => {
                        // 简化的RSI计算
                        let gains = prices[prices.len()-15..prices.len()-1]
                            .iter()
                            .zip(prices[prices.len()-14..].iter())
                            .map(|(prev, curr)| {
                                let diff = curr - prev;
                                if diff > 0.0 { diff } else { 0.0 }
                            })
                            .sum::<f64>() / 14.0;
                            
                        let losses = prices[prices.len()-15..prices.len()-1]
                            .iter()
                            .zip(prices[prices.len()-14..].iter())
                            .map(|(prev, curr)| {
                                let diff = prev - curr;
                                if diff > 0.0 { diff } else { 0.0 }
                            })
                            .sum::<f64>() / 14.0;
                            
                        let rsi = if losses == 0.0 { 
                            100.0 
                        } else { 
                            100.0 - (100.0 / (1.0 + (gains / losses))) 
                        };
                        
                        feature_vector.push(rsi / 100.0);
                    },
                    "macd" => {
                        // 简化的MACD计算
                        let ema12 = prices[prices.len()-12..].iter().sum::<f64>() / 12.0;
                        let ema26 = prices[prices.len()-26..].iter().sum::<f64>() / 26.0;
                        let macd = ema12 - ema26;
                        let normalized = macd / price;
                        feature_vector.push(normalized);
                    },
                    "bollinger" => {
                        // 布林带计算
                        let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                        let variance = prices[prices.len()-20..]
                            .iter()
                            .map(|p| (p - ma20).powi(2))
                            .sum::<f64>() / 20.0;
                        let std_dev = variance.sqrt();
                        let upper_band = ma20 + 2.0 * std_dev;
                        let lower_band = ma20 - 2.0 * std_dev;
                        
                        // 计算价格在布林带中的相对位置 (-1到1)
                        let position = if upper_band == lower_band {
                            0.0
                        } else {
                            2.0 * (price - lower_band) / (upper_band - lower_band) - 1.0
                        };
                        
                        feature_vector.push(position);
                    },
                    "stochastic_k" => {
                        // K值计算
                        let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        
                        let k = if highest_high == lowest_low {
                            0.5
                        } else {
                            (price - lowest_low) / (highest_high - lowest_low)
                        };
                        
                        feature_vector.push(k);
                    },
                    "stochastic_d" => {
                        // 简化D值计算(K值的3日移动平均)
                        let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        
                        let k1 = if highest_high == lowest_low {
                            0.5
                        } else {
                            (price - lowest_low) / (highest_high - lowest_low)
                        };
                        
                        let k2 = if highest_high == lowest_low {
                            0.5
                        } else {
                            (prices[prices.len()-2] - lowest_low) / (highest_high - lowest_low)
                        };
                        
                        let k3 = if highest_high == lowest_low {
                            0.5
                        } else {
                            (prices[prices.len()-3] - lowest_low) / (highest_high - lowest_low)
                        };
                        
                        let d = (k1 + k2 + k3) / 3.0;
                        feature_vector.push(d);
                    },
                    "momentum" => {
                        // 动量指标(当前价格与n日前价格的比率)
                        if i >= 30 {
                            let price_n_days_ago = prices[prices.len()-10];
                            let momentum = price / price_n_days_ago - 1.0;
                            feature_vector.push(momentum);
                        } else {
                            feature_vector.push(0.0);
                        }
                    },
                    _ => {
                        // 未知特征，添加0值
                        feature_vector.push(0.0);
                    }
                }
            }
            
            features_matrix.push(feature_vector);
        }
    }
    
    // 移除前20天(因为某些指标需要历史数据计算)
    dates = dates[20..].to_vec();
    prices = prices[20..].to_vec();
    
    // 创建目标变量: 使用未来n天的价格变化率
    let pred_days = request.prediction_days;
    let mut targets = Vec::new();
    
    for i in 0..features_matrix.len() - pred_days {
        let current_price = prices[i];
        let future_price = prices[i + pred_days];
        let change_rate = (future_price - current_price) / current_price;
        targets.push(change_rate);
    }
    
    // 截断特征矩阵，使其与目标变量长度匹配
    features_matrix.truncate(targets.len());
    dates.truncate(targets.len());
    
    // 划分训练集和测试集
    let train_size = (targets.len() as f64 * request.train_test_split) as usize;
    
    // 转换为张量
    let features_len = features_matrix[0].len();
    let x_train_vec: Vec<f64> = features_matrix[0..train_size].iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let y_train_vec: Vec<f64> = targets[0..train_size].to_vec();
    
    let x_test_vec: Vec<f64> = features_matrix[train_size..].iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let y_test_vec: Vec<f64> = targets[train_size..].to_vec();
    
    // 创建张量
    let x_train = Tensor::from_slice(&x_train_vec, &[train_size, features_len], &device)?;
    let y_train = Tensor::from_slice(&y_train_vec, &[train_size, 1], &device)?;
    
    let test_size = targets.len() - train_size;
    let x_test = Tensor::from_slice(&x_test_vec, &[test_size, features_len], &device)?;
    let y_test = Tensor::from_slice(&y_test_vec, &[test_size, 1], &device)?;
    
    Ok((x_train, y_train, x_test, y_test, dates))
}

// 训练模型函数
pub async fn train_candle_model(request: TrainingRequest) -> std::result::Result<TrainingResult, String> {
    let model_id = Uuid::new_v4().to_string();
    let model_type = request.model_type.clone();
    
    // 准备数据
    let (x_train, y_train, x_test, y_test, _) = prepare_stock_data(&request)
        .map_err(|e| format!("数据准备失败: {}", e))?;
    
    // 设置设备
    let device = Device::Cpu;
    
    // 创建模型配置
    let config = ModelConfig {
        model_type: model_type.clone(),
        input_size: request.features.len(),
        hidden_size: 64, // 隐藏层大小
        output_size: 1,  // 输出尺寸 (股价)
        dropout: request.dropout,
        learning_rate: request.learning_rate,
        n_layers: 2,     // 默认值
        n_heads: 4,      // 默认值
        max_seq_len: 60, // 默认值
    };
    
    // 创建模型
    let (varmap, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {}", e))?;
    
    // 创建优化器
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), request.learning_rate)
        .map_err(|e| format!("优化器创建失败: {}", e))?;
    
    // 训练模型
    let batch_size = request.batch_size;
    let num_batches = x_train.dim(0).unwrap() / batch_size;
    
    for epoch in 0..request.epochs {
        let mut epoch_loss = 0.0;
        
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let x_batch = x_train.narrow(0, batch_start, batch_size)
                .map_err(|e| format!("批次数据准备失败: {}", e))?;
            let y_batch = y_train.narrow(0, batch_start, batch_size)
                .map_err(|e| format!("批次数据准备失败: {}", e))?;
            
            // 前向传播
            let output = model.forward(&x_batch)
                .map_err(|e| format!("前向传播失败: {}", e))?;
            
            // 计算损失 (均方误差)
            let loss = output.sub(&y_batch).map_err(|e| format!("计算损失失败: {}", e))?;
            let loss_squared = loss.sqr().map_err(|e| format!("计算平方失败: {}", e))?;
            let loss = loss_squared.mean_all().map_err(|e| format!("计算均值失败: {}", e))?;
            
            // 反向传播
            optimizer.backward_step(&loss)
                .map_err(|e| format!("反向传播失败: {}", e))?;
            
            epoch_loss += loss.to_scalar::<f32>().unwrap() as f64;
        }
        
        // 每10个epoch记录一次损失
        if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == request.epochs - 1 {
            println!("Epoch {}/{}, Loss: {:.4}", epoch + 1, request.epochs, epoch_loss / num_batches as f64);
        }
    }
    
    // 评估模型
    let y_pred = model.forward(&x_test)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    // 计算MSE (均方误差)
    let diff = y_pred.sub(&y_test).map_err(|e| format!("计算MSE失败: {}", e))?;
    let squared_diff = diff.sqr().map_err(|e| format!("计算平方失败: {}", e))?;
    let mse = squared_diff.mean_all().map_err(|e| format!("计算均值失败: {}", e))?;
    let mse = mse.to_scalar::<f32>().unwrap() as f64;
    
    // 计算RMSE (均方根误差)
    let rmse = mse.sqrt();
    
    // 计算精度 (1 - 归一化RMSE)
    let y_mean = y_test.mean_all().map_err(|e| format!("计算平均值失败: {}", e))?.to_scalar::<f32>().unwrap() as f64;
    let accuracy = 1.0 - (rmse / y_mean).min(1.0);
    
    println!("评估结果: MSE = {:.4}, RMSE = {:.4}, Accuracy = {:.4}", mse, rmse, accuracy);
    
    // 保存模型
    let model_dir = get_model_dir(&model_id);
    fs::create_dir_all(&model_dir).map_err(|e| format!("创建模型目录失败: {}", e))?;
    
    let model_path = model_dir.join("model.safetensors");
    save_model(&varmap, &model_path).map_err(|e| format!("模型保存失败: {}", e))?;
    
    // 保存模型元数据
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    
    let metadata = ModelInfo {
        id: model_id,
        name: request.model_name,
        stock_code: request.stock_code,
        created_at: now,
        model_type,
        features: request.features,
        target: request.target,
        prediction_days: request.prediction_days,
        accuracy,
    };
    
    save_model_metadata(&metadata).map_err(|e| format!("元数据保存失败: {}", e))?;
    
    Ok(TrainingResult {
        metadata,
        accuracy,
    })
}

// 股票预测函数
pub async fn predict_with_candle(request: PredictionRequest) -> std::result::Result<Vec<Prediction>, String> {
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
        dropout: 0.0, // 预测时不使用dropout
        learning_rate: 0.001,
        n_layers: 2,   // 默认值
        n_heads: 4,    // 默认值
        max_seq_len: 60, // 默认值
    };
    
    let mut varmap = VarMap::new();
    
    let (_, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {}", e))?;
    
    let model_path = get_model_dir(&metadata.id).join("model.safetensors");
    varmap.load(&model_path).map_err(|e| format!("模型加载失败: {}", e))?;
    
    // 获取最近的市场数据用于预测
    // 在实际应用中，应从数据库获取最近的股票数据
    // 这里使用模拟数据

    // 模拟获取最近股票数据
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    let mut dates = Vec::new();
    let mut current_price = 100.0;
    let today = chrono::Local::now().naive_local().date();
    
    // 获取过去30天的数据用于计算特征
    for i in (1..=30).rev() {
        let date = today - chrono::Duration::days(i);
        dates.push(date.format("%Y-%m-%d").to_string());
        
        // 模拟价格和成交量
        let random = (rand::random::<f64>() - 0.5) * 2.0;
        current_price = current_price * (1.0 + random * 0.01);
        prices.push(current_price);
        
        let volume = (1000000.0 * (1.0 + (rand::random::<f64>() - 0.5) * 0.3)) as i64;
        volumes.push(volume);
    }
    
    // 计算特征向量
    let mut features = Vec::new();
    
    for feature_name in &metadata.features {
        match feature_name.as_str() {
            "close" => {
                // 归一化收盘价
                let normalized = (current_price - 95.0) / 20.0; // 假设价格范围在95-115之间
                features.push(normalized);
            },
            "volume" => {
                // 归一化成交量
                let latest_volume = volumes.last().unwrap();
                let normalized = *latest_volume as f64 / 2000000.0;
                features.push(normalized);
            },
            "change_percent" => {
                // 计算价格变化百分比
                let prev_price = prices[prices.len() - 2];
                let change = (current_price - prev_price) / prev_price;
                features.push(change);
            },
            "ma5" => {
                // 5日移动平均线
                let ma5 = prices[prices.len()-5..].iter().sum::<f64>() / 5.0;
                let normalized = (ma5 - current_price) / current_price;
                features.push(normalized);
            },
            "ma10" => {
                // 10日移动平均线
                let ma10 = prices[prices.len()-10..].iter().sum::<f64>() / 10.0;
                let normalized = (ma10 - current_price) / current_price;
                features.push(normalized);
            },
            "ma20" => {
                // 20日移动平均线
                let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                let normalized = (ma20 - current_price) / current_price;
                features.push(normalized);
            },
            "rsi" => {
                // 简化的RSI计算
                let gains = prices[prices.len()-15..prices.len()-1]
                    .iter()
                    .zip(prices[prices.len()-14..].iter())
                    .map(|(prev, curr)| {
                        let diff = curr - prev;
                        if diff > 0.0 { diff } else { 0.0 }
                    })
                    .sum::<f64>() / 14.0;
                    
                let losses = prices[prices.len()-15..prices.len()-1]
                    .iter()
                    .zip(prices[prices.len()-14..].iter())
                    .map(|(prev, curr)| {
                        let diff = prev - curr;
                        if diff > 0.0 { diff } else { 0.0 }
                    })
                    .sum::<f64>() / 14.0;
                    
                let rsi = if losses == 0.0 { 
                    100.0 
                } else { 
                    100.0 - (100.0 / (1.0 + (gains / losses))) 
                };
                
                features.push(rsi / 100.0);
            },
            "macd" => {
                // 简化的MACD计算
                let ema12 = prices[prices.len()-12..].iter().sum::<f64>() / 12.0;
                let ema26 = prices[prices.len()-26..].iter().sum::<f64>() / 26.0;
                let macd = ema12 - ema26;
                let normalized = macd / current_price;
                features.push(normalized);
            },
            "bollinger" => {
                // 布林带计算
                let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                let variance = prices[prices.len()-20..]
                    .iter()
                    .map(|p| (p - ma20).powi(2))
                    .sum::<f64>() / 20.0;
                let std_dev = variance.sqrt();
                let upper_band = ma20 + 2.0 * std_dev;
                let lower_band = ma20 - 2.0 * std_dev;
                
                // 计算价格在布林带中的相对位置 (-1到1)
                let position = if upper_band == lower_band {
                    0.0
                } else {
                    2.0 * (current_price - lower_band) / (upper_band - lower_band) - 1.0
                };
                
                features.push(position);
            },
            "stochastic_k" => {
                // K值计算
                let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                
                let k = if highest_high == lowest_low {
                    0.5
                } else {
                    (current_price - lowest_low) / (highest_high - lowest_low)
                };
                
                features.push(k);
            },
            "stochastic_d" => {
                // 简化D值计算(K值的3日移动平均)
                let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                
                let k1 = if highest_high == lowest_low {
                    0.5
                } else {
                    (current_price - lowest_low) / (highest_high - lowest_low)
                };
                
                let k2 = if highest_high == lowest_low {
                    0.5
                } else {
                    (prices[prices.len()-2] - lowest_low) / (highest_high - lowest_low)
                };
                
                let k3 = if highest_high == lowest_low {
                    0.5
                } else {
                    (prices[prices.len()-3] - lowest_low) / (highest_high - lowest_low)
                };
                
                let d = (k1 + k2 + k3) / 3.0;
                features.push(d);
            },
            "momentum" => {
                // 动量指标(当前价格与n日前价格的比率)
                let price_n_days_ago = prices[prices.len()-10];
                let momentum = current_price / price_n_days_ago - 1.0;
                features.push(momentum);
            },
            _ => {
                // 未知特征，添加0值
                features.push(0.0);
            }
        }
    }
    
    // 创建输入张量
    let input_tensor = Tensor::from_slice(&features, &[1, features.len()], &device)
        .map_err(|e| format!("创建输入张量失败: {}", e))?;
    
    // 进行预测
    let output = model.forward(&input_tensor)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    // 获取预测结果(价格变化率)
    let change_rate = output.to_scalar::<f32>().unwrap() as f64;
    
    // 生成预测
    let mut predictions = Vec::new();
    let mut last_price = current_price;
    
    // 计算价格波动预测的置信度(基于特征分布)
    // 实际应用中可以使用更复杂的方法计算置信度
    let compute_confidence = |days_ahead: i64| -> f64 {
        // 置信度随预测天数增加而降低
        let base_confidence = 0.95;
        let decay_rate = 0.03;
        (base_confidence * (1.0 - decay_rate * days_ahead as f64)).max(0.5)
    };
    
    for day in 1..=request.prediction_days {
        // 创建目标日期
        let target_date = today + chrono::Duration::days(day as i64);
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 每天的变化率在上一次预测的基础上微调
        let day_change_rate = change_rate * (1.0 + (rand::random::<f64>() - 0.5) * 0.2);
        
        // 计算预测价格
        let predicted_price = last_price * (1.0 + day_change_rate);
        
        // 计算变化百分比
        let change_percent = day_change_rate * 100.0;
        
        // 计算置信度
        let confidence = compute_confidence(day as i64);
        
        // 添加预测结果
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
        });
        
        // 更新上一个预测价格
        last_price = predicted_price;
    }
    
    Ok(predictions)
}

// 重新训练模型
pub async fn retrain_candle_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> std::result::Result<(), String> {
    // 加载模型元数据
    let metadata = load_model_metadata(&model_id)
        .map_err(|e| format!("加载模型元数据失败: {}", e))?;
    
    // 构建训练请求
    let request = TrainingRequest {
        stock_code: metadata.stock_code.clone(),
        model_name: metadata.name.clone(),
        start_date: "2023-01-01".to_string(), // 示例日期
        end_date: "2023-12-31".to_string(),   // 示例日期
        features: metadata.features.clone(),
        target: metadata.target.clone(),
        prediction_days: metadata.prediction_days,
        model_type: metadata.model_type.clone(),
        epochs: epochs as usize,
        batch_size: batch_size as usize,
        learning_rate,
        dropout: 0.2,
        train_test_split: 0.8,
    };
    
    // 删除旧模型
    delete_model(&model_id).map_err(|e| format!("删除旧模型失败: {}", e))?;
    
    // 训练新模型
    let result = train_candle_model(request).await?;
    
    println!("模型重新训练完成，新准确率: {:.4}", result.accuracy);
    
    Ok(())
}

// 评估模型
pub async fn evaluate_candle_model(model_id: String) -> std::result::Result<EvaluationResult, String> {
    // 加载模型元数据
    let metadata = load_model_metadata(&model_id)
        .map_err(|e| format!("加载模型元数据失败: {}", e))?;
    
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
    
    let model_path = get_model_dir(&model_id).join("model.safetensors");
    varmap.load(&model_path).map_err(|e| format!("模型加载失败: {}", e))?;
    
    // 创建测试数据 (在实际应用中，应从数据库获取实际股票数据)
    // 这里进行基本的模拟，以展示评估过程
    
    // 模拟100天的历史数据
    let days = 100;
    let mut price = 100.0;
    let mut prices = Vec::with_capacity(days);
    let mut volumes = Vec::with_capacity(days);
    
    for i in 0..days {
        // 生成价格(有一定的随机波动和趋势)
        let trend = ((i as f64 / 30.0).sin() * 10.0) + ((i as f64 / 90.0).cos() * 5.0);
        let random = (rand::random::<f64>() - 0.5) * 4.0;
        price = price + trend * 0.01 + random;
        prices.push(price);
        
        // 生成成交量
        let volume = (1000000.0 * (1.0 + (rand::random::<f64>() - 0.5) * 0.3)) as i64;
        volumes.push(volume);
    }
    
    // 生成特征矩阵
    let mut features_matrix = Vec::new();
    for i in 19..days {  // 从第20天开始，保证有足够的历史数据计算指标
        let mut feature_vector = Vec::new();
        
        for feature_name in &metadata.features {
            match feature_name.as_str() {
                "close" => {
                    let normalized = (prices[i] - 95.0) / 20.0;
                    feature_vector.push(normalized);
                },
                "volume" => {
                    let normalized = volumes[i] as f64 / 2000000.0;
                    feature_vector.push(normalized);
                },
                "change_percent" => {
                    let change = (prices[i] - prices[i-1]) / prices[i-1];
                    feature_vector.push(change);
                },
                "ma5" => {
                    let ma5 = prices[i-4..=i].iter().sum::<f64>() / 5.0;
                    let normalized = (ma5 - prices[i]) / prices[i];
                    feature_vector.push(normalized);
                },
                "ma10" => {
                    let ma10 = prices[i-9..=i].iter().sum::<f64>() / 10.0;
                    let normalized = (ma10 - prices[i]) / prices[i];
                    feature_vector.push(normalized);
                },
                "ma20" => {
                    let ma20 = prices[i-19..=i].iter().sum::<f64>() / 20.0;
                    let normalized = (ma20 - prices[i]) / prices[i];
                    feature_vector.push(normalized);
                },
                "rsi" => {
                    // 简化的RSI计算
                    let gains = prices[i-14..i].iter()
                        .zip(prices[i-13..=i].iter())
                        .map(|(prev, curr)| {
                            let diff = curr - prev;
                            if diff > 0.0 { diff } else { 0.0 }
                        })
                        .sum::<f64>() / 14.0;
                        
                    let losses = prices[i-14..i].iter()
                        .zip(prices[i-13..=i].iter())
                        .map(|(prev, curr)| {
                            let diff = prev - curr;
                            if diff > 0.0 { diff } else { 0.0 }
                        })
                        .sum::<f64>() / 14.0;
                        
                    let rsi = if losses == 0.0 { 
                        100.0 
                    } else { 
                        100.0 - (100.0 / (1.0 + (gains / losses))) 
                    };
                    
                    feature_vector.push(rsi / 100.0);
                },
                "macd" => {
                    // 简化的MACD计算
                    let ema12 = prices[i-11..=i].iter().sum::<f64>() / 12.0;
                    let ema26 = prices[i-25..=i].iter().sum::<f64>() / 26.0;
                    let macd = ema12 - ema26;
                    let normalized = macd / prices[i];
                    feature_vector.push(normalized);
                },
                "bollinger" => {
                    // 布林带计算
                    let ma20 = prices[i-19..=i].iter().sum::<f64>() / 20.0;
                    let variance = prices[i-19..=i]
                        .iter()
                        .map(|p| (p - ma20).powi(2))
                        .sum::<f64>() / 20.0;
                    let std_dev = variance.sqrt();
                    let upper_band = ma20 + 2.0 * std_dev;
                    let lower_band = ma20 - 2.0 * std_dev;
                    
                    // 计算价格在布林带中的相对位置 (-1到1)
                    let position = if upper_band == lower_band {
                        0.0
                    } else {
                        2.0 * (prices[i] - lower_band) / (upper_band - lower_band) - 1.0
                    };
                    
                    feature_vector.push(position);
                },
                "stochastic_k" => {
                    // K值计算
                    let highest_high = prices[i-13..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lowest_low = prices[i-13..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    
                    let k = if highest_high == lowest_low {
                        0.5
                    } else {
                        (prices[i] - lowest_low) / (highest_high - lowest_low)
                    };
                    
                    feature_vector.push(k);
                },
                "stochastic_d" => {
                    // K值计算
                    let highest_high = prices[i-13..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lowest_low = prices[i-13..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    
                    let k1 = if highest_high == lowest_low {
                        0.5
                    } else {
                        (prices[i] - lowest_low) / (highest_high - lowest_low)
                    };
                    
                    let k2 = if highest_high == lowest_low {
                        0.5
                    } else {
                        (prices[i-1] - lowest_low) / (highest_high - lowest_low)
                    };
                    
                    let k3 = if highest_high == lowest_low {
                        0.5
                    } else {
                        (prices[i-2] - lowest_low) / (highest_high - lowest_low)
                    };
                    
                    let d = (k1 + k2 + k3) / 3.0;
                    feature_vector.push(d);
                },
                "momentum" => {
                    // 动量指标(当前价格与10日前价格的比率)
                    if i >= 29 {
                        let momentum = prices[i] / prices[i-10] - 1.0;
                        feature_vector.push(momentum);
                    } else {
                        feature_vector.push(0.0);
                    }
                },
                _ => {
                    // 未知特征，添加0值
                    feature_vector.push(0.0);
                }
            }
        }
        
        features_matrix.push(feature_vector);
    }
    
    // 生成目标变量: 使用5天后的价格变化率
    let pred_days = metadata.prediction_days;
    let mut targets = Vec::new();
    
    // 价格从第20天开始
    let prices_offset = 20;
    
    for i in 0..features_matrix.len() - pred_days {
        let current_price = prices[i + prices_offset];
        let future_price = prices[i + prices_offset + pred_days];
        
        // 计算未来价格相对当前价格的变化率
        let change_rate = (future_price - current_price) / current_price;
        
        // 将变化率分类为上涨/下跌/持平
        let target = if change_rate > 0.01 {
            1.0  // 明显上涨
        } else if change_rate < -0.01 {
            -1.0 // 明显下跌
        } else {
            0.0  // 基本持平
        };
        
        targets.push(target);
    }
    
    // 截断特征矩阵，使其与目标变量长度匹配
    features_matrix.truncate(targets.len());
    
    // 将特征和目标转换为张量
    let features_len = features_matrix[0].len();
    let test_size = features_matrix.len();
    
    let x_test_vec: Vec<f64> = features_matrix.iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let x_test = Tensor::from_slice(&x_test_vec, &[test_size, features_len], &device)
        .map_err(|e| format!("创建测试特征张量失败: {}", e))?;
    
    // 进行预测
    let y_pred = model.forward(&x_test)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    // 提取预测结果
    let y_pred_vec = y_pred.flatten(0, 1).map_err(|e| format!("展平预测结果失败: {}", e))?
        .to_vec1::<f32>().map_err(|e| format!("转换预测结果失败: {}", e))?;
    
    // 计算评估指标
    let mut correct = 0;
    let mut confusion_matrix = vec![vec![0; 3]; 3];  // 3x3矩阵: [下跌, 持平, 上涨]
    
    for (i, &pred_val) in y_pred_vec.iter().enumerate() {
        let actual_val = targets[i];
        
        // 将预测的变化率转换为类别
        let pred_class = if pred_val > 0.01 {
            1.0  // 上涨
        } else if pred_val < -0.01 {
            -1.0 // 下跌
        } else {
            0.0  // 持平
        };
        
        // 更新正确预测计数
        if (pred_class > 0.0 && actual_val > 0.0) || 
           (pred_class < 0.0 && actual_val < 0.0) || 
           (pred_class == 0.0 && actual_val == 0.0) {
            correct += 1;
        }
        
        // 更新混淆矩阵
        let actual_idx = match actual_val {
            -1.0 => 0, // 下跌
            0.0 => 1,  // 持平
            _ => 2,    // 上涨
        };
        
        let pred_idx = match pred_class {
            -1.0 => 0, // 下跌
            0.0 => 1,  // 持平
            _ => 2,    // 上涨
        };
        
        confusion_matrix[actual_idx][pred_idx] += 1;
    }
    
    // 计算准确率
    let accuracy = correct as f64 / targets.len() as f64;
    
    // 计算F1分数、精确率和召回率(针对上涨类别)
    let true_positives = confusion_matrix[2][2]; // 实际上涨，预测上涨
    let false_positives = confusion_matrix[0][2] + confusion_matrix[1][2]; // 实际非上涨，预测上涨
    let false_negatives = confusion_matrix[2][0] + confusion_matrix[2][1]; // 实际上涨，预测非上涨
    
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };
    
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    // 保存一些预测示例
    let mut prediction_examples = Vec::new();
    let max_examples = std::cmp::min(10, targets.len());
    
    for i in 0..max_examples {
        prediction_examples.push(PredictionExample {
            actual: targets[i],
            predicted: y_pred_vec[i] as f64,
            features: features_matrix[i].clone(),
        });
    }
    
    // 返回评估结果
    Ok(EvaluationResult {
        accuracy,
        confusion_matrix,
        precision,
        recall,
        f1_score,
        prediction_examples,
    })
}

// 导出的命令
pub async fn list_stock_prediction_models(symbol: String) -> Vec<ModelInfo> {
    list_models(&symbol)
}

pub async fn delete_stock_prediction_model(model_id: String) -> std::result::Result<(), String> {
    delete_model(&model_id).map_err(|e| format!("删除模型失败: {}", e))
}
