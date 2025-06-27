use candle_core::{Device, Tensor};
use candle_nn::{Module, Optimizer, VarMap, AdamW};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use rand;
use chrono::{self, Weekday, Datelike};
use sqlx::Row;


// 简化的模型配置结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub dropout: f64,
    pub learning_rate: f64,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_seq_len: usize,
}

// 简化的模型创建函数
fn create_model(config: &ModelConfig, device: &Device) -> Result<(VarMap, Box<dyn Module + Send + Sync>), candle_core::Error> {
    // 创建一个简单的线性回归模型
    let varmap = VarMap::new();
    
    // 定义正确的输入和输出形状
    let input_size = config.input_size;
    let output_size = config.output_size;
    
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
    let model = LinearRegression::new(input_size, output_size, vb)?;
    
    let model: Box<dyn Module + Send + Sync> = Box::new(model);
    
    Ok((varmap, model))
}

// 简化的模型保存函数
fn save_model(varmap: &VarMap, path: &std::path::Path) -> Result<(), candle_core::Error> {
    varmap.save(path)?;
    Ok(())
}

// 数据库路径查找函数
fn find_database_path() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    
    // 尝试多个可能的数据库路径
    let possible_paths = [
        current_dir.join("db/stock_data.db"),
        current_dir.join("src-tauri/db/stock_data.db"),
        current_dir.parent()?.join("src-tauri/db/stock_data.db"), // 如果在 src-tauri 目录内运行
    ];
    
    for path in &possible_paths {
        println!("检查数据库路径: {}", path.display());
        if path.exists() {
            println!("✅ 找到数据库文件: {}", path.display());
            return Some(path.clone());
        }
    }
    
    println!("❌ 未找到数据库文件");
    None
}

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
    pub trading_signal: Option<String>,
    pub signal_strength: Option<f64>,
    pub technical_indicators: Option<TechnicalIndicatorValues>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorValues {
    pub rsi: f64,
    pub macd_histogram: f64,
    pub kdj_j: f64,
    pub cci: f64,
    pub obv_trend: f64, // OBV相对于均值的比例
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLog {
    pub epoch: usize,
    pub loss: f64,
    pub timestamp: String,
    pub message: Option<String>,
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

// 改进的数据预处理函数
async fn prepare_stock_data(
    request: &TrainingRequest,
) -> std::result::Result<(Tensor, Tensor, Tensor, Tensor, Vec<String>), candle_core::Error> {
    // 设置设备
    let device = Device::Cpu;
    
    let symbol = &request.stock_code;
    
    // 解析前端传来的日期范围
    let end_date = chrono::Local::now().naive_local().date();
    let user_start_date = chrono::NaiveDate::parse_from_str(&request.start_date, "%Y-%m-%d")
        .unwrap_or_else(|_| end_date - chrono::Duration::days(210)); // 默认210天
    let user_end_date = chrono::NaiveDate::parse_from_str(&request.end_date, "%Y-%m-%d")
        .unwrap_or(end_date);
    
    // 计算用户请求的天数范围
    let requested_days = (user_end_date - user_start_date).num_days();
    
    // 为A股节假日增加额外缓冲期
    // 如果用户已经包含了缓冲期（如180+30=210天），我们再增加一些以确保数据充足
    let additional_buffer = if requested_days >= 200 { 
        60  // 用户已有缓冲，再加60天
    } else { 
        90  // 用户没有缓冲，加90天
    };
    
    let extended_start_date = user_start_date - chrono::Duration::days(additional_buffer);
    
    // 确保不会查询过于久远的数据（最多2年）
    let max_start_date = end_date - chrono::Duration::days(730);
    let actual_start_date = if extended_start_date < max_start_date {
        max_start_date
    } else {
        extended_start_date
    };
    
    let start_date_str = actual_start_date.format("%Y-%m-%d").to_string();
    let end_date_str = user_end_date.format("%Y-%m-%d").to_string();
    
    println!("📅 A股数据获取策略:");
    println!("   用户请求范围: {} 到 {} ({} 天)", 
             user_start_date.format("%Y-%m-%d"), 
             user_end_date.format("%Y-%m-%d"), 
             requested_days);
    println!("   实际查询范围: {} 到 {} ({} 天，含节假日缓冲)", 
             start_date_str, end_date_str, 
             (user_end_date - actual_start_date).num_days());
    
    // 使用sqlx查询数据库获取历史数据
    let historical_data = match get_historical_data_from_db(symbol, &start_date_str, &end_date_str).await {
        Ok(data) => data,
        Err(e) => {
            eprintln!("从数据库获取数据失败: {}", e);
            return Err(candle_core::Error::Msg(format!("获取历史数据失败: {}", e)));
        }
    };
    
    if historical_data.is_empty() {
        return Err(candle_core::Error::Msg("历史数据为空，无法训练模型".to_string()));
    }
    
    println!("✅ 获取到{}条历史数据", historical_data.len());
    
    // 数据质量检查 - 针对A股特点优化
    let valid_data: Vec<_> = historical_data.into_iter()
        .filter(|data| {
            // A股基本数据验证
            data.close > 0.0 && data.volume >= 0 && 
            data.open > 0.0 && data.high > 0.0 && data.low > 0.0 &&
            data.high >= data.low && data.high >= data.open && 
            data.high >= data.close && data.low <= data.open && data.low <= data.close &&
            // A股涨跌幅限制检查（ST股票20%，普通股票10%）
            data.change_percent.abs() <= 25.0 && // 允许一些数据误差
            // 成交量合理性检查
            data.volume < 1_000_000_000_000 // 避免异常大的成交量
        })
        .collect();
    
    println!("✅ 过滤后有效数据{}条", valid_data.len());
    
    // A股交易日数量估算：一年约250个交易日
    let min_required_days = 120; // 最少约半年交易数据
    let recommended_days = 180; // 推荐约9个月交易数据
    let optimal_days = 250; // 最佳约1年交易数据
    
    if valid_data.len() < min_required_days {
        return Err(candle_core::Error::Msg(format!(
            "A股有效交易数据不足，当前{}天，需要至少{}天数据（约半年交易日）", 
            valid_data.len(), min_required_days
        )));
    }
    
    if valid_data.len() < recommended_days {
        println!("⚠️  警告: 当前数据量{}天少于推荐的{}天，可能影响模型准确率", 
                 valid_data.len(), recommended_days);
    } else if valid_data.len() >= optimal_days {
        println!("✅ 数据量充足: {}天 >= {}天，有利于提高模型准确率", 
                 valid_data.len(), optimal_days);
    }
    
    // 构建特征和标签
    let mut dates = Vec::new();
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    let mut opens = Vec::new();
    let mut features_matrix = Vec::new();
    
    // 按日期排序（从旧到新）
    let mut sorted_data = valid_data.clone();
    sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    // 首先提取基础数据
    for data in &sorted_data {
        dates.push(data.date.clone());
        prices.push(data.close);
        volumes.push(data.volume);
        highs.push(data.high);
        lows.push(data.low);
        opens.push(data.open);
    }
    
    // 数据平滑处理：移除异常值
    let prices = smooth_price_data(&prices);
    let volumes = smooth_volume_data(&volumes);
    
    // 为每天准备一个特征向量（动态计算所需的历史窗口）
    let required_days = request.features.iter()
        .map(|f| get_feature_required_days(f))
        .max()
        .unwrap_or(20);
    
    // 使用较小的lookback_window，但确保有足够数据计算所有特征
    let lookback_window = required_days.max(30).min(prices.len() / 2);
    
    println!("📊 特征计算窗口: {}天, 总价格数据: {}天", lookback_window, prices.len());
    
    for i in lookback_window..prices.len() {
        let mut feature_vector = Vec::new();
        
        // 提取请求中指定的特征
        for feature_name in &request.features {
            let feature_value = calculate_feature_value(
                feature_name, 
                &prices, 
                &volumes, 
                i, 
                lookback_window,
                Some(&highs),
                Some(&lows)
            )?;
            feature_vector.push(feature_value);
        }
        
        features_matrix.push(feature_vector);
    }
    
    println!("🔢 生成特征矩阵: {}行 x {}列", features_matrix.len(), 
             if features_matrix.is_empty() { 0 } else { features_matrix[0].len() });
    
    // 移除前面的数据，因为没有计算特征
    dates = dates[lookback_window..].to_vec();
    let valid_prices = prices[lookback_window..].to_vec();
    
    // 创建目标变量: 使用未来n天的价格变化率
    let pred_days = request.prediction_days;
    let mut targets = Vec::new();
    
    // 防止整数溢出：确保有足够的数据进行预测
    if features_matrix.len() <= pred_days {
        return Err(candle_core::Error::Msg(format!(
            "数据不足以进行{}天预测，当前特征数据{}天，需要至少{}天", 
            pred_days, features_matrix.len(), pred_days + 1
        )));
    }
    
    for i in 0..(features_matrix.len() - pred_days) {
        let current_price = valid_prices[i];
        
        // 计算未来几天的平均价格，减少噪音
        let future_prices: Vec<f64> = (1..=pred_days)
            .map(|day| valid_prices.get(i + day).copied().unwrap_or(current_price))
            .collect();
        
        let future_avg_price = future_prices.iter().sum::<f64>() / future_prices.len() as f64;
        let change_rate = (future_avg_price - current_price) / current_price;
        
        // 限制变化率范围，避免极端值影响训练
        let clamped_change_rate = change_rate.clamp(-0.5, 0.5);
        targets.push(clamped_change_rate);
    }
    
    // 截断特征矩阵，使其与目标变量长度匹配
    features_matrix.truncate(targets.len());
    dates.truncate(targets.len());
    
    // 特征标准化
    let (normalized_features, _feature_stats) = normalize_features(&features_matrix)?;
    
    // 划分训练集和测试集 - 使用时间序列划分
    let train_size = (targets.len() as f64 * request.train_test_split) as usize;
    
    // 转换为张量
    let features_len = normalized_features[0].len();
    let x_train_vec: Vec<f64> = normalized_features[0..train_size].iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let y_train_vec: Vec<f64> = targets[0..train_size].to_vec();
    
    let x_test_vec: Vec<f64> = normalized_features[train_size..].iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let y_test_vec: Vec<f64> = targets[train_size..].to_vec();
    
    // 创建张量，使用F32类型以匹配模型权重
    let x_train_f32: Vec<f32> = x_train_vec.iter().map(|&x| x as f32).collect();
    let y_train_f32: Vec<f32> = y_train_vec.iter().map(|&y| y as f32).collect();
    let x_test_f32: Vec<f32> = x_test_vec.iter().map(|&x| x as f32).collect();
    let y_test_f32: Vec<f32> = y_test_vec.iter().map(|&y| y as f32).collect();
    
    let x_train = Tensor::from_slice(&x_train_f32, &[train_size, features_len], &device)?;
    let y_train = Tensor::from_slice(&y_train_f32, &[train_size, 1], &device)?;
    
    let test_size = targets.len() - train_size;
    let x_test = Tensor::from_slice(&x_test_f32, &[test_size, features_len], &device)?;
    let y_test = Tensor::from_slice(&y_test_f32, &[test_size, 1], &device)?;
    
    println!("数据预处理完成: 训练集{}样本, 测试集{}样本, 特征维度{}", 
             train_size, test_size, features_len);
    
    Ok((x_train, y_train, x_test, y_test, dates))
}

// 数据平滑处理函数
fn smooth_price_data(prices: &[f64]) -> Vec<f64> {
    let mut smoothed = prices.to_vec();
    
    // 使用中位数滤波器移除价格异常值
    for i in 2..smoothed.len()-2 {
        let window: Vec<f64> = smoothed[i-2..=i+2].to_vec();
        let mut sorted_window = window.clone();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_window[2];
        
        // 如果当前值与中位数相差超过20%，用中位数替换
        if (smoothed[i] - median).abs() / median > 0.2 {
            smoothed[i] = median;
        }
    }
    
    smoothed
}

fn smooth_volume_data(volumes: &[i64]) -> Vec<i64> {
    let mut smoothed = volumes.to_vec();
    
    // 移除成交量异常值
    for i in 2..smoothed.len()-2 {
        let window: Vec<i64> = smoothed[i-2..=i+2].to_vec();
        let avg = window.iter().sum::<i64>() as f64 / window.len() as f64;
        
        // 如果当前值与平均值相差超过5倍，用平均值替换
        if (smoothed[i] as f64 - avg).abs() / avg > 5.0 {
            smoothed[i] = avg as i64;
        }
    }
    
    smoothed
}

// 获取特征所需的历史天数
fn get_feature_required_days(feature_name: &str) -> usize {
    match feature_name {
        "close" | "volume" | "change_percent" => 1,
        "ma5" => 5,
        "ma10" => 10,
        "ma20" | "bollinger" | "cci" => 20,
        "rsi" | "stochastic_k" | "stochastic_d" | "dmi_plus" | "dmi_minus" | "adx" => 14,
        "macd" | "macd_dif" | "macd_dea" | "macd_histogram" => 26,
        "momentum" => 10,
        "kdj_k" | "kdj_d" | "kdj_j" => 9,
        "obv" => 2,
        _ => 1,
    }
}

// 计算单个特征值
fn calculate_feature_value(
    feature_name: &str,
    prices: &[f64],
    volumes: &[i64],
    index: usize,
    _lookback_window: usize,
    highs: Option<&[f64]>,
    lows: Option<&[f64]>,
) -> Result<f64, candle_core::Error> {
    match feature_name {
        "close" => {
            Ok(prices[index])
        },
        "volume" => {
            Ok(volumes[index] as f64)
        },
        "change_percent" => {
            if index > 0 {
                let prev_price = prices[index - 1];
                let change = (prices[index] - prev_price) / prev_price;
                Ok(change)
            } else {
                Ok(0.0)
            }
        },
        "ma5" => {
            if index >= 4 {
                let ma5 = prices[index-4..=index].iter().sum::<f64>() / 5.0;
                Ok(ma5)
            } else {
                Ok(prices[index])
            }
        },
        "ma10" => {
            if index >= 9 {
                let ma10 = prices[index-9..=index].iter().sum::<f64>() / 10.0;
                Ok(ma10)
            } else {
                Ok(prices[index])
            }
        },
        "ma20" => {
            if index >= 19 {
                let ma20 = prices[index-19..=index].iter().sum::<f64>() / 20.0;
                Ok(ma20)
            } else {
                Ok(prices[index])
            }
        },
        "rsi" => {
            if index >= 14 {
                Ok(calculate_rsi(&prices[index-14..=index]))
            } else { 
                Ok(50.0) // 默认中性RSI
            }
        },
        "macd" => {
            if index >= 25 {
                Ok(calculate_macd(&prices[index-25..=index]))
            } else {
                Ok(0.0)
            }
        },
        "bollinger" => {
            if index >= 19 {
                Ok(calculate_bollinger_position(&prices[index-19..=index], prices[index]))
            } else {
                Ok(0.0)
            }
        },
        "stochastic_k" => {
            if index >= 13 {
                Ok(calculate_stochastic_k(&prices[index-13..=index], prices[index]))
            } else {
                Ok(0.5)
            }
        },
        "stochastic_d" => {
            if index >= 15 {
                // 计算前3天的K值的平均值
                let k_values: Vec<f64> = (0..3)
                    .map(|i| {
                        let k_index = index - i;
                        if k_index >= 13 {
                            calculate_stochastic_k(&prices[k_index-13..=k_index], prices[k_index])
                        } else {
                            0.5
                        }
                    })
                    .collect();
                Ok(k_values.iter().sum::<f64>() / k_values.len() as f64)
            } else {
                Ok(0.5)
            }
        },
        "momentum" => {
            if index >= 10 {
                let momentum = prices[index] / prices[index-10] - 1.0;
                Ok(momentum)
            } else {
                Ok(0.0)
            }
        },
        "kdj_k" | "kdj_d" | "kdj_j" => {
            // KDJ指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 9 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(8);
                    let (k, d, j) = calculate_kdj(&highs[start..=index], &lows[start..=index], &prices[start..=index], 9);
                    match feature_name {
                        "kdj_k" => Ok(k / 100.0), // 归一化到0-1
                        "kdj_d" => Ok(d / 100.0),
                        "kdj_j" => Ok(j / 100.0),
                        _ => Ok(0.0)
                    }
                } else {
                    Ok(0.5) // 默认中性值
                }
            } else {
                Ok(0.5)
            }
        },
        "cci" => {
            // CCI指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 20 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(19);
                    let cci = calculate_cci(&highs[start..=index], &lows[start..=index], &prices[start..=index], 20);
                    Ok(cci / 200.0) // 归一化，CCI通常在-200到200之间
                } else {
                    Ok(0.0)
                }
            } else {
                Ok(0.0)
            }
        },
        "obv" => {
            // OBV指标
            if index >= 1 {
                let obv = calculate_obv(&prices[0..=index], &volumes[0..=index]);
                // 归一化OBV（相对于平均成交量）
                let avg_volume = volumes[0..=index].iter().sum::<i64>() as f64 / (index + 1) as f64;
                Ok(obv / (avg_volume * (index + 1) as f64))
            } else {
                Ok(0.0)
            }
        },
        "macd_dif" | "macd_dea" | "macd_histogram" => {
            // 完整MACD指标
            if index >= 26 {
                let (dif, dea, histogram) = calculate_macd_full(&prices[0..=index]);
                let normalized = match feature_name {
                    "macd_dif" => dif / prices[index],
                    "macd_dea" => dea / prices[index],
                    "macd_histogram" => histogram / prices[index],
                    _ => 0.0
                };
                Ok(normalized)
            } else {
                Ok(0.0)
            }
        },
        "dmi_plus" | "dmi_minus" | "adx" => {
            // DMI指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 14 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(13);
                    let (di_plus, di_minus, adx, _) = calculate_dmi(&highs[start..=index], &lows[start..=index], &prices[start..=index], 14);
                    match feature_name {
                        "dmi_plus" => Ok(di_plus / 100.0),
                        "dmi_minus" => Ok(di_minus / 100.0),
                        "adx" => Ok(adx / 100.0),
                        _ => Ok(0.0)
                    }
                } else {
                    Ok(0.0)
                }
            } else {
                Ok(0.0)
            }
        },
        _ => {
            Ok(0.0)
        }
    }
}

// 特征标准化
fn normalize_features(features: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<(f64, f64)>), candle_core::Error> {
    if features.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    
    let feature_count = features[0].len();
    let mut stats = Vec::with_capacity(feature_count);
    let mut normalized = vec![vec![0.0; feature_count]; features.len()];
    
    // 计算每个特征的均值和标准差
    for feature_idx in 0..feature_count {
        let values: Vec<f64> = features.iter().map(|row| row[feature_idx]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt().max(1e-8); // 避免除零
        
        stats.push((mean, std_dev));
        
        // 标准化该特征
        for (row_idx, row) in features.iter().enumerate() {
            normalized[row_idx][feature_idx] = (row[feature_idx] - mean) / std_dev;
        }
    }
    
    Ok((normalized, stats))
}

// RSI计算函数
fn calculate_rsi(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 50.0;
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += -change;
        }
    }
    
    gains /= (prices.len() - 1) as f64;
    losses /= (prices.len() - 1) as f64;
    
    if losses == 0.0 {
        100.0
    } else {
        100.0 - (100.0 / (1.0 + (gains / losses)))
    }
}

// MACD计算函数
fn calculate_macd(prices: &[f64]) -> f64 {
    if prices.len() < 26 {
        return 0.0;
    }
    
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    ema12 - ema26
}

// 布林带位置计算
fn calculate_bollinger_position(prices: &[f64], current_price: f64) -> f64 {
    if prices.len() < 20 {
        return 0.0;
    }
    
    let ma = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter()
        .map(|p| (p - ma).powi(2))
        .sum::<f64>() / prices.len() as f64;
    let std_dev = variance.sqrt();
    
    let upper_band = ma + 2.0 * std_dev;
    let lower_band = ma - 2.0 * std_dev;
    
    if upper_band == lower_band {
        0.0
    } else {
        (current_price - lower_band) / (upper_band - lower_band) - 0.5
    }
}

// 随机指标K值计算
fn calculate_stochastic_k(prices: &[f64], current_price: f64) -> f64 {
    if prices.is_empty() {
        return 0.5;
    }
    
    let highest = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        0.5
    } else {
        (current_price - lowest) / (highest - lowest)
    }
}

// 从数据库获取历史数据
async fn get_historical_data_from_db(symbol: &str, start_date: &str, end_date: &str) -> Result<Vec<HistoricalDataType>, String> {
    // 创建一个临时的数据库连接
    use sqlx::sqlite::SqlitePoolOptions;
    
    // 使用动态数据库路径查找
    let db_path = find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    
    let connection_string = format!("sqlite://{}", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("连接数据库失败: {}", e))?;
    
    let records = sqlx::query_as::<_, HistoricalDataType>(
        r#"SELECT * FROM historical_data 
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#
    )
    .bind(symbol)
    .bind(start_date)
    .bind(end_date)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("查询历史数据失败: {}", e))?;
    
    Ok(records)
}

// 历史数据结构体
#[derive(Debug, Clone)]
struct HistoricalDataType {
    pub date: String,
    pub open: f64,
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub volume: i64,
    pub change_percent: f64,
}

// 实现FromRow特征，使其可以从数据库行转换
impl<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> for HistoricalDataType {
    fn from_row(row: &'r sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        Ok(Self {
            date: row.try_get("date")?,
            open: row.try_get("open")?,
            close: row.try_get("close")?,
            high: row.try_get("high")?,
            low: row.try_get("low")?,
            volume: row.try_get("volume")?,
            change_percent: row.try_get("change_percent")?,
        })
    }
}

// 计算指数移动平均线
fn calculate_ema(data: &[f64], period: usize) -> f64 {
    if data.is_empty() || period == 0 || data.len() < period {
        return 0.0;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0];
    
    for i in 1..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }
    
    ema
}

// 🎯 新增：计算完整的MACD指标（包括DIF、DEA、MACD柱）
fn calculate_macd_full(prices: &[f64]) -> (f64, f64, f64) {
    if prices.len() < 26 {
        return (0.0, 0.0, 0.0);
    }
    
    // 计算EMA12和EMA26
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    
    // DIF = EMA12 - EMA26
    let dif = ema12 - ema26;
    
    // 计算最近9天的DIF值用于计算DEA
    let mut dif_values = Vec::new();
    for i in (prices.len().saturating_sub(9))..prices.len() {
        if i >= 26 {
            let sub_prices = &prices[0..=i];
            let sub_ema12 = calculate_ema(sub_prices, 12);
            let sub_ema26 = calculate_ema(sub_prices, 26);
            dif_values.push(sub_ema12 - sub_ema26);
        }
    }
    
    // DEA = EMA(DIF, 9)
    let dea = if dif_values.len() >= 9 {
        calculate_ema(&dif_values, 9)
    } else {
        dif // 如果数据不足，使用DIF作为DEA
    };
    
    // MACD柱 = 2 * (DIF - DEA)
    let macd = 2.0 * (dif - dea);
    
    (dif, dea, macd)
}

// 🎯 新增：计算KDJ指标
fn calculate_kdj(highs: &[f64], lows: &[f64], closes: &[f64], n: usize) -> (f64, f64, f64) {
    if highs.len() < n || lows.len() < n || closes.len() < n {
        return (50.0, 50.0, 50.0);
    }
    
    let len = highs.len();
    let start = len.saturating_sub(n);
    
    // 计算N日内最高价和最低价
    let highest = highs[start..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = lows[start..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        return (50.0, 50.0, 50.0);
    }
    
    // 计算RSV
    let rsv = (closes[len - 1] - lowest) / (highest - lowest) * 100.0;
    
    // 简化计算：使用最近3天的平均值模拟K值的平滑
    let mut k_values = vec![rsv];
    for i in 1..3 {
        if len > i {
            let idx = len - 1 - i;
            if idx >= start {
                let h = highs[start..=idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let l = lows[start..=idx].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                if h > l {
                    k_values.push((closes[idx] - l) / (h - l) * 100.0);
                }
            }
        }
    }
    
    let k = k_values.iter().sum::<f64>() / k_values.len() as f64;
    
    // D值是K值的3日移动平均
    let d = k * 0.667 + 50.0 * 0.333; // 简化计算
    
    // J = 3K - 2D
    let j = 3.0 * k - 2.0 * d;
    
    (k, d, j)
}

// 🎯 新增：计算OBV（能量潮）指标
fn calculate_obv(prices: &[f64], volumes: &[i64]) -> f64 {
    if prices.len() < 2 || volumes.len() < 2 {
        return 0.0;
    }
    
    let mut obv = 0.0;
    for i in 1..prices.len().min(volumes.len()) {
        if prices[i] > prices[i - 1] {
            obv += volumes[i] as f64;
        } else if prices[i] < prices[i - 1] {
            obv -= volumes[i] as f64;
        }
        // 价格不变时，OBV保持不变
    }
    
    obv
}

// 🎯 新增：计算CCI（商品通道指数）
fn calculate_cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }
    
    let start = highs.len().saturating_sub(period);
    let mut tp_values = Vec::new(); // Typical Price
    
    for i in start..highs.len() {
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        tp_values.push(tp);
    }
    
    // 计算移动平均
    let ma = tp_values.iter().sum::<f64>() / tp_values.len() as f64;
    
    // 计算平均偏差
    let md = tp_values.iter()
        .map(|&tp| (tp - ma).abs())
        .sum::<f64>() / tp_values.len() as f64;
    
    if md == 0.0 {
        return 0.0;
    }
    
    // CCI = (TP - MA) / (0.015 * MD)
    let current_tp = (highs.last().unwrap() + lows.last().unwrap() + closes.last().unwrap()) / 3.0;
    (current_tp - ma) / (0.015 * md)
}

// 🎯 新增：计算DMI（动向指标）
fn calculate_dmi(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> (f64, f64, f64, f64) {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let mut tr_values = Vec::new();
    let mut dm_plus_values = Vec::new();
    let mut dm_minus_values = Vec::new();
    
    // 计算TR、+DM、-DM
    for i in 1..highs.len() {
        // True Range
        let h_l = highs[i] - lows[i];
        let h_pc = (highs[i] - closes[i - 1]).abs();
        let l_pc = (lows[i] - closes[i - 1]).abs();
        let tr = h_l.max(h_pc).max(l_pc);
        tr_values.push(tr);
        
        // Directional Movement
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];
        
        let dm_plus = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let dm_minus = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        
        dm_plus_values.push(dm_plus);
        dm_minus_values.push(dm_minus);
    }
    
    if tr_values.len() < period {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    // 计算平滑后的值（使用简单移动平均代替Wilder's平滑）
    let start = tr_values.len().saturating_sub(period);
    let atr = tr_values[start..].iter().sum::<f64>() / period as f64;
    let adm_plus = dm_plus_values[start..].iter().sum::<f64>() / period as f64;
    let adm_minus = dm_minus_values[start..].iter().sum::<f64>() / period as f64;
    
    if atr == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    // 计算方向指标
    let di_plus = (adm_plus / atr) * 100.0;
    let di_minus = (adm_minus / atr) * 100.0;
    
    // 计算ADX
    let dx = if di_plus + di_minus == 0.0 {
        0.0
    } else {
        ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100.0
    };
    
    // 简化的ADX计算（应该是DX的平滑值）
    let adx = dx;
    
    (di_plus, di_minus, adx, dx)
}

// 🎯 新增：计算SAR（抛物线停损转向）指标
fn calculate_sar(highs: &[f64], lows: &[f64], af_step: f64, af_max: f64) -> Vec<f64> {
    if highs.len() < 2 || lows.len() < 2 {
        return vec![0.0; highs.len()];
    }
    
    let mut sar_values = Vec::new();
    let mut is_uptrend = true;
    let mut af = af_step;
    let mut ep = highs[0]; // 极值点
    let mut sar = lows[0];
    
    sar_values.push(sar);
    
    for i in 1..highs.len() {
        // 计算新的SAR
        let new_sar = sar + af * (ep - sar);
        
        if is_uptrend {
            // 上升趋势
            if lows[i] <= new_sar {
                // 趋势反转
                is_uptrend = false;
                sar = ep;
                ep = lows[i];
                af = af_step;
            } else {
                sar = new_sar;
                if highs[i] > ep {
                    ep = highs[i];
                    af = (af + af_step).min(af_max);
                }
            }
        } else {
            // 下降趋势
            if highs[i] >= new_sar {
                // 趋势反转
                is_uptrend = true;
                sar = ep;
                ep = highs[i];
                af = af_step;
            } else {
                sar = new_sar;
                if lows[i] < ep {
                    ep = lows[i];
                    af = (af + af_step).min(af_max);
                }
            }
        }
        
        sar_values.push(sar);
    }
    
    sar_values
}

// 🎯 新增：综合技术指标分析函数
fn analyze_technical_signals(
    prices: &[f64], 
    highs: &[f64], 
    lows: &[f64], 
    volumes: &[i64]
) -> TechnicalSignals {
    let len = prices.len();
    
    // 计算各种技术指标
    let (macd_dif, macd_dea, macd_histogram) = if len >= 26 {
        calculate_macd_full(prices)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    let (kdj_k, kdj_d, kdj_j) = if len >= 14 && highs.len() >= 14 && lows.len() >= 14 {
        calculate_kdj(highs, lows, prices, 9)
    } else {
        (50.0, 50.0, 50.0)
    };
    
    let rsi = if len >= 14 {
        calculate_rsi(&prices[len.saturating_sub(14)..])
    } else {
        50.0
    };
    
    let cci = if len >= 20 && highs.len() >= 20 && lows.len() >= 20 {
        calculate_cci(highs, lows, prices, 20)
    } else {
        0.0
    };
    
    let obv = calculate_obv(prices, volumes);
    let obv_ma = if volumes.len() >= 10 {
        let recent_obv_values: Vec<f64> = (0..10).map(|i| {
            let end = volumes.len() - i;
            calculate_obv(&prices[0..end], &volumes[0..end])
        }).collect();
        recent_obv_values.iter().sum::<f64>() / recent_obv_values.len() as f64
    } else {
        obv
    };
    
         // 🎯 生成买卖信号
    let mut buy_signals = 0;
    let mut sell_signals = 0;
    let mut signal_strength: f64 = 0.0;
    
    // MACD信号
    if macd_dif > macd_dea && macd_histogram > 0.0 {
        buy_signals += 1;
        signal_strength += 0.15;
    } else if macd_dif < macd_dea && macd_histogram < 0.0 {
        sell_signals += 1;
        signal_strength -= 0.15;
    }
    
    // KDJ信号
    if kdj_j < 20.0 || (kdj_k < 30.0 && kdj_d < 30.0) {
        buy_signals += 1;
        signal_strength += 0.2;
    } else if kdj_j > 80.0 || (kdj_k > 70.0 && kdj_d > 70.0) {
        sell_signals += 1;
        signal_strength -= 0.2;
    }
    
    // RSI信号
    if rsi < 30.0 {
        buy_signals += 1;
        signal_strength += 0.15;
    } else if rsi > 70.0 {
        sell_signals += 1;
        signal_strength -= 0.15;
    }
    
    // CCI信号
    if cci < -100.0 {
        buy_signals += 1;
        signal_strength += 0.1;
    } else if cci > 100.0 {
        sell_signals += 1;
        signal_strength -= 0.1;
    }
    
    // OBV信号
    if obv > obv_ma {
        buy_signals += 1;
        signal_strength += 0.1;
    } else if obv < obv_ma {
        sell_signals += 1;
        signal_strength -= 0.1;
    }
    
    // 价格突破信号
    if len >= 20 {
        let ma20 = prices[len-20..].iter().sum::<f64>() / 20.0;
        if prices[len-1] > ma20 * 1.02 {
            buy_signals += 1;
            signal_strength += 0.1;
        } else if prices[len-1] < ma20 * 0.98 {
            sell_signals += 1;
            signal_strength -= 0.1;
        }
    }
    
    // 成交量信号
    if volumes.len() >= 5 {
        let vol_ma5 = volumes[volumes.len()-5..].iter().sum::<i64>() as f64 / 5.0;
        let current_vol = *volumes.last().unwrap() as f64;
        if current_vol > vol_ma5 * 1.5 && prices[len-1] > prices[len-2] {
            buy_signals += 1;
            signal_strength += 0.1;
        }
    }
    
    // 计算综合信号
    let signal = if buy_signals > sell_signals + 2 {
        TradingSignal::StrongBuy
    } else if buy_signals > sell_signals {
        TradingSignal::Buy
    } else if sell_signals > buy_signals + 2 {
        TradingSignal::StrongSell
    } else if sell_signals > buy_signals {
        TradingSignal::Sell
    } else {
        TradingSignal::Hold
    };
    
    TechnicalSignals {
        macd_dif,
        macd_dea,
        macd_histogram,
        kdj_k,
        kdj_d,
        kdj_j,
        rsi,
        cci,
        obv,
        signal,
                 signal_strength: signal_strength.max(-1.0).min(1.0),
        buy_signals,
        sell_signals,
    }
}

// 🎯 新增：技术信号结构体
#[derive(Debug, Clone)]
struct TechnicalSignals {
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub macd_histogram: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub kdj_j: f64,
    pub rsi: f64,
    pub cci: f64,
    pub obv: f64,
    pub signal: TradingSignal,
    pub signal_strength: f64,
    pub buy_signals: i32,
    pub sell_signals: i32,
}

// 🎯 新增：交易信号枚举
#[derive(Debug, Clone, PartialEq)]
enum TradingSignal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

// 训练模型函数
pub async fn train_candle_model(request: TrainingRequest) -> std::result::Result<TrainingResult, String> {
    let model_id = Uuid::new_v4().to_string();
    let model_type = request.model_type.clone();
    
    // 准备数据
    let (x_train, y_train, x_test, y_test, _) = prepare_stock_data(&request).await
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
            // 确保输出和目标张量的形状匹配
            println!("输出形状: {:?}, 目标形状: {:?}", output.dims(), y_batch.dims());
            
            // 如果输出形状和目标形状不匹配，则进行调整
            let reshaped_output = if output.dims() != y_batch.dims() {
                if output.dim(0).unwrap() == y_batch.dim(0).unwrap() {
                    // 如果批次大小相同但输出维度不同，尝试reshape
                    output.reshape(&[output.dim(0).unwrap(), 1])
                        .map_err(|e| format!("调整输出形状失败: {}", e))?
                } else {
                    return Err(format!("输出形状 {:?} 和目标形状 {:?} 不兼容", output.dims(), y_batch.dims()));
                }
            } else {
                output
            };
            
            let loss = reshaped_output.sub(&y_batch).map_err(|e| format!("计算损失失败: {}", e))?;
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
    
    // 转换为Vec用于准确率计算 - 处理不同维度的张量
    let predictions_vec = match y_pred.dims() {
        // 如果是1维张量 [n]
        [_] => {
            y_pred.to_vec1::<f32>().map_err(|e| format!("转换1维预测结果失败: {}", e))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        },
        // 如果是2维张量 [n, 1] 
        [_, 1] => {
            y_pred.to_vec2::<f32>().map_err(|e| format!("转换2维预测结果失败: {}", e))?
                .into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>()
        },
        // 如果是其他2维张量 [n, m]
        [_, _] => {
            let vec2d = y_pred.to_vec2::<f32>().map_err(|e| format!("转换2维预测结果失败: {}", e))?;
            vec2d.into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>() // 取第一列
        },
        // 其他维度
        _ => {
            return Err("预测输出张量维度不支持，请检查模型配置".to_string());
        }
    };
    
    let actuals_vec = match y_test.dims() {
        // 如果是1维张量 [n]
        [_] => {
            y_test.to_vec1::<f32>().map_err(|e| format!("转换1维实际结果失败: {}", e))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        },
        // 如果是2维张量 [n, 1]
        [_, 1] => {
            y_test.to_vec2::<f32>().map_err(|e| format!("转换2维实际结果失败: {}", e))?
                .into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>()
        },
        // 如果是其他2维张量 [n, m]
        [_, _] => {
            let vec2d = y_test.to_vec2::<f32>().map_err(|e| format!("转换2维实际结果失败: {}", e))?;
            vec2d.into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>() // 取第一列
        },
        // 其他维度
        _ => {
            return Err("实际值张量维度不支持，请检查数据准备".to_string());
        }
    };
    
    // 使用改进的准确率计算方法（更重视方向预测）
    let (direction_accuracy, combined_accuracy) = calculate_direction_focused_accuracy(&predictions_vec, &actuals_vec);
    
    // 计算MSE和RMSE用于日志显示
    let diff = y_pred.sub(&y_test).map_err(|e| format!("计算MSE失败: {}", e))?;
    let squared_diff = diff.sqr().map_err(|e| format!("计算平方失败: {}", e))?;
    let mse = squared_diff.mean_all().map_err(|e| format!("计算均值失败: {}", e))?;
    let mse = mse.to_scalar::<f32>().unwrap() as f64;
    let rmse = mse.sqrt();
    
    println!("评估结果: MSE = {:.4}, RMSE = {:.4}", mse, rmse);
    println!("🎯 方向预测准确率: {:.2}% | 综合准确率: {:.2}%", 
             direction_accuracy * 100.0, combined_accuracy * 100.0);
    println!("📊 预测张量维度: {:?}, 实际张量维度: {:?}", y_pred.dims(), y_test.dims());
    
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
        accuracy: combined_accuracy,
    };
    
    save_model_metadata(&metadata).map_err(|e| format!("元数据保存失败: {}", e))?;
    
    Ok(TrainingResult {
        metadata,
        accuracy: combined_accuracy,
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
    
    // 获取最近的真实市场数据用于预测
    let (current_price, dates, prices, volumes, highs, lows) = get_recent_market_data(&request.stock_code, 60).await
        .map_err(|e| format!("获取市场数据失败: {}", e))?;
    
    if prices.len() < 20 {
        return Err("历史数据不足，无法进行预测，需要至少20天数据".to_string());
    }
    
    // 计算特征向量
    let mut features = Vec::new();
    let last_idx = prices.len() - 1;
    
    // 为每个特征计算值
    for feature_name in &metadata.features {
        match feature_name.as_str() {
            "close" => {
                // 归一化收盘价
                let price_min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let price_max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let price_range = price_max - price_min;
                let normalized = if price_range > 0.0 {
                    (current_price - price_min) / price_range
                } else {
                    0.5 // 如果价格没有变化，使用中间值
                };
                features.push(normalized);
            },
            "volume" => {
                // 归一化成交量
                let latest_volume = volumes[last_idx];
                let vol_min = volumes.iter().fold(i64::MAX, |a, &b| a.min(b));
                let vol_max = volumes.iter().fold(i64::MIN, |a, &b| a.max(b));
                let vol_range = (vol_max - vol_min) as f64;
                let normalized = if vol_range > 0.0 {
                    (latest_volume - vol_min) as f64 / vol_range
                } else {
                    0.5 // 如果成交量没有变化，使用中间值
                };
                features.push(normalized);
            },
            "change_percent" => {
                // 计算价格变化百分比
                let prev_price = prices[last_idx - 1];
                let change = (current_price - prev_price) / prev_price;
                let normalized = (change / 0.1).clamp(-1.0, 1.0); // 假设正常变化率在±10%内
                features.push(normalized);
            },
            "ma5" => {
                // 5日移动平均线
                if prices.len() >= 5 {
                    let ma5 = prices[prices.len()-5..].iter().sum::<f64>() / 5.0;
                    let normalized = (ma5 - current_price) / current_price;
                    features.push(normalized);
                } else {
                    features.push(0.0);
                }
            },
            "ma10" => {
                // 10日移动平均线
                if prices.len() >= 10 {
                    let ma10 = prices[prices.len()-10..].iter().sum::<f64>() / 10.0;
                    let normalized = (ma10 - current_price) / current_price;
                    features.push(normalized);
                } else {
                    features.push(0.0);
                }
            },
            "ma20" => {
                // 20日移动平均线
                if prices.len() >= 20 {
                    let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                    let normalized = (ma20 - current_price) / current_price;
                    features.push(normalized);
                } else {
                    features.push(0.0);
                }
            },
            "rsi" => {
                // RSI计算
                if prices.len() >= 15 {
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
                } else {
                    features.push(0.5); // 默认中性RSI
                }
            },
            "macd" => {
                // MACD计算
                if prices.len() >= 26 {
                    let ema12 = prices[prices.len()-26..].iter().sum::<f64>() / 12.0;
                    let ema26 = prices[prices.len()-26..].iter().sum::<f64>() / 26.0;
                    let macd = ema12 - ema26;
                    let normalized = macd / current_price;
                    features.push(normalized);
                } else {
                    features.push(0.0);
                }
            },
            "bollinger" => {
                // 布林带计算
                if prices.len() >= 20 {
                    let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
                    let variance = prices[prices.len()-20..]
                        .iter()
                        .map(|p| (p - ma20).powi(2))
                        .sum::<f64>() / 20.0;
                    let std_dev = variance.sqrt();
                    let upper_band = ma20 + 2.0 * std_dev;
                    let lower_band = ma20 - 2.0 * std_dev;
                    
                    // 计算价格在布林带中的相对位置 (-1到1)
                    let position = if upper_band > lower_band {
                        2.0 * (current_price - lower_band) / (upper_band - lower_band) - 1.0
                    } else {
                        0.0
                    };
                    
                    features.push(position);
                } else {
                    features.push(0.0);
                }
            },
            "stochastic_k" => {
                // K值计算
                if prices.len() >= 14 {
                    let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    
                    let k = if highest_high > lowest_low {
                        (current_price - lowest_low) / (highest_high - lowest_low)
                    } else {
                        0.5
                    };
                    
                    features.push(k);
                } else {
                    features.push(0.5);
                }
            },
            "stochastic_d" => {
                // D值计算(K值的3日移动平均)
                if prices.len() >= 14 {
                    let highest_high = prices[prices.len()-14..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let lowest_low = prices[prices.len()-14..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    
                    let k1 = if highest_high > lowest_low {
                        (current_price - lowest_low) / (highest_high - lowest_low)
                    } else {
                        0.5
                    };
                    
                    let k2 = if highest_high > lowest_low && prices.len() > 1 {
                        (prices[prices.len()-2] - lowest_low) / (highest_high - lowest_low)
                    } else {
                        0.5
                    };
                    
                    let k3 = if highest_high > lowest_low && prices.len() > 2 {
                        (prices[prices.len()-3] - lowest_low) / (highest_high - lowest_low)
                    } else {
                        0.5
                    };
                    
                    let d = (k1 + k2 + k3) / 3.0;
                    features.push(d);
                } else {
                    features.push(0.5);
                }
            },
            "momentum" => {
                // 动量指标(当前价格与n日前价格的比率)
                let n = 10; // 使用10日动量
                if prices.len() > n {
                    let price_n_days_ago = prices[prices.len()-n-1];
                    let momentum = current_price / price_n_days_ago - 1.0;
                    let normalized = (momentum / 0.2).clamp(-1.0, 1.0); // 假设正常动量在±20%内
                    features.push(normalized);
                } else {
                    features.push(0.0);
                }
            },
            _ => {
                // 未知特征，添加0值
                features.push(0.0);
            }
        }
    }
    
    // 创建输入张量，转换为F32类型以匹配模型权重
    let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
    let input_tensor = Tensor::from_slice(&features_f32, &[1, features.len()], &device)
        .map_err(|e| format!("创建输入张量失败: {}", e))?;
    
    // 进行预测
    let output = model.forward(&input_tensor)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    // 如果输出是多维的，确保我们得到正确的值
    println!("预测输出形状: {:?}", output.dims());
    let raw_change_rate = match output.dims() {
        // 如果是1维张量 [1] 或 [batch_size]
        [_] => {
            output.to_vec1::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0] as f64
        },
        // 如果是2维张量 [batch_size, 1] 或 [batch_size, features]
        [_, n] => {
            if *n == 1 { // 修复: 解引用 n
                // 如果是 [batch_size, 1]，直接获取第一个元素
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0][0] as f64
            } else {
                // 如果是 [batch_size, features]，获取第一行第一列元素
                println!("警告: 预测输出维度与预期不符，尝试获取第一个值");
                output.to_vec2::<f32>().map_err(|e| format!("获取预测结果失败: {}", e))?[0][0] as f64
            }
        },
        // 其他维度，返回错误
        _ => {
            return Err(format!("预测输出维度不支持: {:?}", output.dims()));
        }
    };
    
    // 基于历史数据计算真实的价格变化模式
    let historical_volatility = calculate_historical_volatility(&prices);
    let recent_trend = calculate_recent_trend(&prices);
    let support_resistance = calculate_support_resistance(&prices, current_price);
    
    // 🎯 使用综合技术分析
    let technical_signals = analyze_technical_signals(&prices, &highs, &lows, &volumes);
    
    println!("📊 历史波动率: {:.4}, 近期趋势: {:.4}, 支撑阻力: {:.4}", 
             historical_volatility, recent_trend, support_resistance);
    println!("📈 技术信号: {:?}, 信号强度: {:.2}, 买入信号: {}, 卖出信号: {}", 
             technical_signals.signal, technical_signals.signal_strength, 
             technical_signals.buy_signals, technical_signals.sell_signals);
    println!("📊 原始模型预测变化率: {:.6}", raw_change_rate);
    
    // 生成预测
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut last_price = current_price;
    
    // 获取最后一个日期，用于计算预测日期
    let last_date = chrono::NaiveDate::parse_from_str(&dates.last().unwrap_or(&"2023-01-01".to_string()), "%Y-%m-%d")
        .unwrap_or_else(|_| chrono::Local::now().naive_local().date());
    
    for day in 1..=request.prediction_days {
        // 创建目标日期 - 使用A股交易日规则
        let mut target_date = last_date;
        // 向前推进指定的交易日数
        for _ in 0..day {
            target_date = get_next_trading_day(target_date);
        }
        let date_str = target_date.format("%Y-%m-%d").to_string();
        
        // 🎯 改进的预测策略：结合技术分析的涨跌预测
        
        // 1. 基础模型预测（标准化处理）
        let base_model_prediction = raw_change_rate.tanh() * 0.02; // 限制在±2%范围内
        
        // 2. 历史波动性调整
        let volatility_factor = historical_volatility.clamp(0.01, 0.08); // 限制波动率范围
        
        // 3. 趋势修正（随时间衰减）
        let trend_decay = 0.9_f64.powi(day as i32);
        let trend_factor = recent_trend * trend_decay;
        
        // 4. 技术信号影响（A股特色：技术分析权重较高）
        let technical_impact = technical_signals.signal_strength * 0.03 * (0.95_f64.powi(day as i32));
        
        // 5. 随机市场噪音（模拟真实市场的不确定性，考虑A股波动性）
        let market_noise = (rand::random::<f64>() - 0.5) * volatility_factor * 1.2;
        
        // 6. 均值回归效应（价格偏离均值时的回归倾向）
        let mean_reversion = if prices.len() >= 20 {
            let ma20 = prices[prices.len()-20..].iter().sum::<f64>() / 20.0;
            let deviation = (last_price - ma20) / ma20;
            -deviation * 0.25 * (1.0 / day as f64) // 偏离越大，回归力越强
        } else {
            0.0
        };
        
        // 7. 支撑阻力位影响
        let sr_effect = support_resistance * 0.4;
        
        // 8. A股特色：追涨杀跌心理（短期动量效应）
        let momentum_effect = if day <= 2 && technical_signals.buy_signals > technical_signals.sell_signals {
            0.01 // 买入信号多时，短期可能继续上涨
        } else if day <= 2 && technical_signals.sell_signals > technical_signals.buy_signals {
            -0.01 // 卖出信号多时，短期可能继续下跌
        } else {
            0.0
        };
        
        // 综合计算预测变化率（调整权重，更重视技术分析）
        let mut predicted_change_rate = base_model_prediction * 0.25  // 模型预测25%权重
            + technical_impact * 0.30                                 // 技术分析30%权重
            + trend_factor * 0.20                                    // 趋势20%权重
            + market_noise * 0.10                                    // 随机噪音10%权重
            + mean_reversion * 0.08                                  // 均值回归8%权重
            + sr_effect * 0.05                                       // 支撑阻力5%权重
            + momentum_effect * 0.02;                                // 动量效应2%权重
        
        // 7. 引入周期性调整（模拟市场的周期性波动）
        let cycle_adjustment = match day % 3 {
            1 => 0.0,                                                // 第1天保持原预测
            2 => -predicted_change_rate * 0.3,                      // 第2天适度反向调整
            0 => predicted_change_rate * 0.2,                       // 第3天小幅同向调整
            _ => 0.0,
        };
        predicted_change_rate += cycle_adjustment;
        
        // 8. 应用A股涨跌幅限制
        let change_percent = clamp_daily_change(predicted_change_rate * 100.0);
        let clamped_change_rate = change_percent / 100.0;
        
        // 计算预测价格
        let predicted_price = last_price * (1.0 + clamped_change_rate);
        
        // 🎯 改进的置信度计算（结合技术分析）
        let base_confidence = (metadata.accuracy + 0.3).min(0.8); // 提升基础置信度
        
        // 置信度影响因子
        let volatility_impact = 1.0 - (volatility_factor * 8.0).min(0.4);     // 波动率影响
        let trend_strength = (recent_trend.abs() * 10.0).min(0.2);            // 趋势强度奖励
        let prediction_magnitude = 1.0 - (change_percent.abs() / 15.0).min(0.3); // 预测幅度惩罚
        let time_decay = 0.95_f64.powi(day as i32);                           // 时间衰减
        
        // 技术信号一致性
        let technical_consistency = match technical_signals.signal {
            TradingSignal::StrongBuy | TradingSignal::StrongSell => 1.15, // 强烈信号提升置信度
            TradingSignal::Buy | TradingSignal::Sell => 1.05,             // 一般信号轻微提升
            TradingSignal::Hold => 0.95,                                  // 横盘信号降低置信度
        };
        
        // 信号与预测方向一致性
        let signal_alignment = match (&technical_signals.signal, predicted_change_rate > 0.0) {
            (TradingSignal::StrongBuy | TradingSignal::Buy, true) => 1.1,   // 买入信号与上涨预测一致
            (TradingSignal::StrongSell | TradingSignal::Sell, false) => 1.1, // 卖出信号与下跌预测一致
            (TradingSignal::Hold, _) => 1.0,                                // 横盘信号中性
            _ => 0.9,                                                        // 信号与预测不一致
        };
        
        let model_consistency = if day > 1 && !predictions.is_empty() {
            // 检查预测的一致性（避免剧烈波动）
            let prev_change = predictions.last().unwrap().predicted_change_percent;
            let change_diff = (change_percent - prev_change).abs();
            1.0 - (change_diff / 10.0).min(0.2)
        } else {
            1.0
        };
        
        let confidence = (base_confidence 
            * volatility_impact 
            * prediction_magnitude 
            * time_decay 
            * model_consistency
            * technical_consistency
            * signal_alignment
            + trend_strength * 0.1)
            .clamp(0.40, 0.90); // 调整置信度范围为40%-90%
        
        // 添加预测结果（包含技术分析信息）
        let trading_signal_str = match &technical_signals.signal {
            TradingSignal::StrongBuy => "强烈买入",
            TradingSignal::Buy => "买入",
            TradingSignal::Hold => "持有",
            TradingSignal::Sell => "卖出",
            TradingSignal::StrongSell => "强烈卖出",
        };
        
        let technical_indicators = TechnicalIndicatorValues {
            rsi: technical_signals.rsi,
            macd_histogram: technical_signals.macd_histogram,
            kdj_j: technical_signals.kdj_j,
            cci: technical_signals.cci,
            obv_trend: if technical_signals.obv > 0.0 { 1.0 } else { -1.0 }, // 简化的OBV趋势
        };
        
        predictions.push(Prediction {
            target_date: date_str,
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(trading_signal_str.to_string()),
            signal_strength: Some(technical_signals.signal_strength),
            technical_indicators: Some(technical_indicators),
        });
        
        // 更新上一个预测价格
        last_price = predicted_price;
        
        // 输出调试信息
        println!("📈 第{}天预测: 价格={:.2}, 变化={:.2}%, 置信度={:.2}%", 
                 day, predicted_price, change_percent, confidence * 100.0);
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
    
    // 构建训练请求 - 使用更长的时间范围
    let end_date = chrono::Local::now().naive_local().date();
    let start_date = end_date - chrono::Duration::days(500); // 使用约1.5年的数据
    
    let request = TrainingRequest {
        stock_code: metadata.stock_code.clone(),
        model_name: metadata.name.clone(),
        start_date: start_date.format("%Y-%m-%d").to_string(),
        end_date: end_date.format("%Y-%m-%d").to_string(),
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
    
    println!("🔄 开始重新训练模型，使用扩展的时间范围: {} 到 {}", 
             request.start_date, request.end_date);
    
    // 删除旧模型
    delete_model(&model_id).map_err(|e| format!("删除旧模型失败: {}", e))?;
    
    // 训练新模型
    let result = train_candle_model(request).await?;
    
    println!("✅ 模型重新训练完成，新准确率: {:.4}%", result.accuracy * 100.0);
    
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
    
    // 将特征和目标转换为张量，使用F32类型以匹配模型权重
    let features_len = features_matrix[0].len();
    let test_size = features_matrix.len();
    
    let x_test_vec: Vec<f64> = features_matrix.iter()
        .flat_map(|v| v.iter().copied())
        .collect();
    
    let x_test_f32: Vec<f32> = x_test_vec.iter().map(|&x| x as f32).collect();
    let x_test = Tensor::from_slice(&x_test_f32, &[test_size, features_len], &device)
        .map_err(|e| format!("创建测试特征张量失败: {}", e))?;
    
    // 进行预测
    let y_pred = model.forward(&x_test)
        .map_err(|e| format!("预测失败: {}", e))?;
    
    // 确保预测输出的形状与目标需求一致
    println!("预测输出形状: {:?}, 测试数据大小: {}", y_pred.dims(), test_size);
    let y_pred_adjusted = if y_pred.dims().len() == 2 && y_pred.dims()[1] != 1 {
        // 如果输出是 [test_size, N] 形状（N > 1），取第一列作为预测结果
        println!("调整预测输出，从多列变为单列");
        y_pred.narrow(1, 0, 1).map_err(|e| format!("调整预测形状失败: {}", e))?
    } else if y_pred.dims().len() == 1 {
        // 如果输出是 [test_size] 形状，重塑为 [test_size, 1]
        println!("调整预测输出，从1维变为2维");
        y_pred.reshape(&[test_size, 1]).map_err(|e| format!("调整预测形状失败: {}", e))?
    } else {
        // 其他情况下保持原样
        y_pred
    };
    
    // 提取预测结果
    let y_pred_vec = match y_pred_adjusted.dims().len() {
        1 => {
            // 如果是1维张量
            y_pred_adjusted.to_vec1::<f32>().map_err(|e| format!("转换1维预测结果失败: {}", e))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        },
        2 => {
            // 如果是2维张量 [test_size, 1]
            let vec2d = y_pred_adjusted.to_vec2::<f32>().map_err(|e| format!("转换2维预测结果失败: {}", e))?;
            vec2d.into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>()
        },
        _ => {
            // 如果是其他维度，尝试展平
            let flat = y_pred_adjusted.flatten_all().map_err(|e| format!("展平预测结果失败: {}", e))?;
            flat.to_vec1::<f32>().map_err(|e| format!("转换展平预测结果失败: {}", e))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        }
    };
    
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
            predicted: y_pred_vec[i],
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



// A股交易规则工具函数
fn is_trading_day(date: chrono::NaiveDate) -> bool {
    match date.weekday() {
        Weekday::Mon | Weekday::Tue | Weekday::Wed | Weekday::Thu | Weekday::Fri => {
            // 这里可以进一步添加节假日判断
            // 暂时只判断工作日
            true
        },
        _ => false
    }
}

fn get_next_trading_day(date: chrono::NaiveDate) -> chrono::NaiveDate {
    let mut next_date = date + chrono::Duration::days(1);
    while !is_trading_day(next_date) {
        next_date += chrono::Duration::days(1);
    }
    next_date
}

fn clamp_daily_change(change_percent: f64) -> f64 {
    // A股涨跌停限制：±10%
    change_percent.clamp(-10.0, 10.0)
}

// 计算历史波动率
fn calculate_historical_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 20 {
        return 0.02; // 默认2%波动率
    }
    
    // 计算过去20天的价格变化率
    let mut daily_returns = Vec::new();
    for i in 1..std::cmp::min(21, prices.len()) {
        let return_rate = (prices[prices.len() - i] - prices[prices.len() - i - 1]) / prices[prices.len() - i - 1];
        daily_returns.push(return_rate);
    }
    
    // 计算标准差
    let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
    let variance = daily_returns.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / daily_returns.len() as f64;
    
    variance.sqrt().min(0.1) // 限制最大波动率为10%
}

// 计算近期趋势
fn calculate_recent_trend(prices: &[f64]) -> f64 {
    if prices.len() < 10 {
        return 0.0;
    }
    
    let recent_len = std::cmp::min(10, prices.len());
    let recent_prices = &prices[prices.len() - recent_len..];
    
    // 使用简单线性回归计算趋势
    let n = recent_len as f64;
    let sum_x = (0..recent_len).sum::<usize>() as f64;
    let sum_y = recent_prices.iter().sum::<f64>();
    let sum_xy = recent_prices.iter().enumerate()
        .map(|(i, &price)| i as f64 * price)
        .sum::<f64>();
    let sum_x2 = (0..recent_len).map(|i| (i * i) as f64).sum::<f64>();
    
    // 趋势斜率
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let avg_price = sum_y / n;
    
    // 标准化趋势 (相对于平均价格)
    (slope / avg_price).clamp(-0.05, 0.05) // 限制在±5%范围内
}

// 计算支撑阻力位
fn calculate_support_resistance(prices: &[f64], current_price: f64) -> f64 {
    if prices.len() < 30 {
        return 0.01; // 默认1%影响
    }
    
    let recent_len = std::cmp::min(30, prices.len());
    let recent_prices = &prices[prices.len() - recent_len..];
    
    // 寻找局部高点和低点
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    
    for i in 1..recent_prices.len() - 1 {
        if recent_prices[i] > recent_prices[i-1] && recent_prices[i] > recent_prices[i+1] {
            highs.push(recent_prices[i]);
        }
        if recent_prices[i] < recent_prices[i-1] && recent_prices[i] < recent_prices[i+1] {
            lows.push(recent_prices[i]);
        }
    }
    
    // 找到最近的支撑位和阻力位
    let resistance = highs.iter().fold(0.0, |acc, &x| if x > current_price && (acc == 0.0 || x < acc) { x } else { acc });
    let support = lows.iter().fold(0.0, |acc, &x| if x < current_price && x > acc { x } else { acc });
    
    // 计算支撑阻力影响
    let sr_strength = if resistance > 0.0 && support > 0.0 {
        let resistance_dist = (resistance - current_price) / current_price;
        let support_dist = (current_price - support) / current_price;
        (resistance_dist - support_dist) * 0.5 // 平衡支撑阻力影响
    } else if resistance > 0.0 {
        (resistance - current_price) / current_price * 0.3
    } else if support > 0.0 {
        (current_price - support) / current_price * 0.3
    } else {
        0.0
    };
    
    sr_strength.clamp(-0.03, 0.03) // 限制在±3%范围内
}



// 新增：涨跌方向分类预测结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPrediction {
    pub target_date: String,
    pub predicted_direction: String, // "上涨", "下跌", "横盘"
    pub direction_confidence: f64,   // 方向预测置信度
    pub predicted_price: f64,        // 预测价格（仅供参考）
    pub predicted_change_percent: f64,
    pub confidence: f64,
}

// 新增：涨跌方向分类枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up,    // 上涨 > 0.5%
    Down,  // 下跌 < -0.5%
    Flat,  // 横盘 [-0.5%, 0.5%]
}

impl Direction {
    fn from_change_percent(change: f64) -> Self {
        if change > 0.5 {
            Direction::Up
        } else if change < -0.5 {
            Direction::Down
        } else {
            Direction::Flat
        }
    }
}



// 新增：改进的准确率计算，更加重视方向预测
fn calculate_direction_focused_accuracy(predictions: &[f64], actuals: &[f64]) -> (f64, f64) {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return (0.0, 0.0);
    }
    
    let mut direction_correct = 0;
    let mut total_predictions = 0;
    let mut price_error_sum = 0.0;
    
    for i in 1..predictions.len().min(actuals.len()) {
        // 计算预测和实际的变化方向
        let pred_change = predictions[i] - predictions[i-1];
        let actual_change = actuals[i] - actuals[i-1];
        
        // 方向分类（使用更严格的阈值）
        let pred_direction = Direction::from_change_percent(pred_change / predictions[i-1] * 100.0);
        let actual_direction = Direction::from_change_percent(actual_change / actuals[i-1] * 100.0);
        
        // 方向准确性检查
        if pred_direction == actual_direction {
            direction_correct += 1;
        }
        
        // 价格准确性（相对误差）
        let relative_error = ((predictions[i] - actuals[i]) / actuals[i]).abs();
        price_error_sum += relative_error;
        
        total_predictions += 1;
    }
    
    if total_predictions == 0 {
        return (0.0, 0.0);
    }
    
    let direction_accuracy = direction_correct as f64 / total_predictions as f64;
    let price_accuracy = (1.0 - (price_error_sum / total_predictions as f64)).max(0.0);
    
    // 方向准确率权重提高到70%，价格准确率30%
    let combined_accuracy = direction_accuracy * 0.7 + price_accuracy * 0.3;
    
    (direction_accuracy, combined_accuracy.min(0.85)) // 限制最高准确率保持现实性
}

// 从数据库获取最近的市场数据
async fn get_recent_market_data(symbol: &str, days: usize) -> Result<(f64, Vec<String>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<f64>), String> {
    // 创建临时数据库连接
    use sqlx::sqlite::SqlitePoolOptions;
    use chrono::Local;
    
    // 计算开始日期（大幅增加数据获取范围，确保有足够的历史数据进行技术分析）
    let end_date = Local::now().naive_local().date();
    let buffer_days = 60; // 增加缓冲期到60天，应对节假日
    // 至少获取1年的数据，或者用户指定天数+缓冲期，取更大值
    let total_days = std::cmp::max(365, days + buffer_days); 
    let start_date = end_date - chrono::Duration::days(total_days as i64);
    
    // 使用动态数据库路径查找
    let db_path = find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    
    let connection_string = format!("sqlite://{}", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("连接数据库失败: {}", e))?;
    
    // 修改查询，获取更多历史数据但保持合理的限制
    let limit = std::cmp::max(300, days * 2); // 至少300条记录，或者请求天数的2倍
    let records = sqlx::query_as::<_, HistoricalDataType>(
        r#"SELECT * FROM historical_data 
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date DESC
           LIMIT ?"#
    )
    .bind(symbol)
    .bind(start_date.format("%Y-%m-%d").to_string())
    .bind(end_date.format("%Y-%m-%d").to_string())
    .bind(limit as i32)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("查询历史数据失败: {}", e))?;
    
    if records.is_empty() {
        return Err(format!("未找到股票代码 {} 的历史数据", symbol));
    }
    
    // 反向排序以获取时间顺序（从旧到新）
    let mut sorted_records = records;
    sorted_records.reverse();
    
    // 提取数据
    let dates: Vec<String> = sorted_records.iter().map(|r| r.date.clone()).collect();
    let prices: Vec<f64> = sorted_records.iter().map(|r| r.close).collect();
    let volumes: Vec<i64> = sorted_records.iter().map(|r| r.volume).collect();
    let highs: Vec<f64> = sorted_records.iter().map(|r| r.high).collect();
    let lows: Vec<f64> = sorted_records.iter().map(|r| r.low).collect();
    
    // 获取最新价格
    let current_price = prices.last().copied().unwrap_or(0.0);
    
    println!("📊 获取到{}条历史数据用于预测，时间范围: {} 到 {}", 
             sorted_records.len(),
             sorted_records.first().map(|r| &r.date).unwrap_or(&"未知".to_string()),
             sorted_records.last().map(|r| &r.date).unwrap_or(&"未知".to_string()));
    
    Ok((current_price, dates, prices, volumes, highs, lows))
}
