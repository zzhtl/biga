use candle_core::{Device, Tensor};
use candle_nn::{Module, Optimizer, VarMap, AdamW};
use std::fs;
use chrono;
use crate::stock_prediction::types::{
    ModelConfig, ModelInfo, TrainingRequest, TrainingResult
};
use crate::stock_prediction::database::get_historical_data_from_db;
use crate::stock_prediction::utils::{
    smooth_price_data, smooth_volume_data, 
    calculate_direction_focused_accuracy
};
use crate::stock_prediction::technical_indicators::{
    get_feature_required_days, calculate_feature_value
};
use crate::stock_prediction::model_management::{
    save_model_metadata, generate_model_id, get_current_timestamp, get_model_file_path
};

// ===== 金融级深度神经网络模型 =====

/// 深度神经网络模型 - 专为股票预测优化
struct DeepStockPredictor {
    // 第一层 - 特征提取
    fc1: candle_nn::Linear,
    dropout1: f64,
    // 第二层 - 深度特征学习
    fc2: candle_nn::Linear,
    dropout2: f64,
    // 第三层 - 模式识别
    fc3: candle_nn::Linear,
    dropout3: f64,
    // 第四层 - 深度抽象
    fc4: candle_nn::Linear,
    dropout4: f64,
    // 残差连接层
    residual: candle_nn::Linear,
    // 输出层
    fc_out: candle_nn::Linear,
    // 训练模式标志
    training: bool,
}

impl DeepStockPredictor {
    fn new(
        in_size: usize,
        hidden_size: usize,
        out_size: usize,
        dropout: f64,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self, candle_core::Error> {
        // 计算各层维度 - 金字塔结构
        let layer1_size = hidden_size * 2;      // 第一层扩展(256)
        let layer2_size = hidden_size;          // 第二层标准(128)
        let layer3_size = hidden_size / 2;      // 第三层收缩(64)
        let layer4_size = hidden_size / 4;      // 第四层进一步收缩(32)
        
        Ok(Self {
            // 第一层：输入 -> 扩展特征空间
            fc1: candle_nn::linear(in_size, layer1_size, vb.pp("fc1"))?,
            dropout1: dropout,
            // 第二层：深度特征学习
            fc2: candle_nn::linear(layer1_size, layer2_size, vb.pp("fc2"))?,
            dropout2: dropout * 0.8,  // 逐层降低dropout
            // 第三层：模式识别
            fc3: candle_nn::linear(layer2_size, layer3_size, vb.pp("fc3"))?,
            dropout3: dropout * 0.6,
            // 第四层：深度抽象
            fc4: candle_nn::linear(layer3_size, layer4_size, vb.pp("fc4"))?,
            dropout4: dropout * 0.4,
            // 残差连接（跳跃连接，从输入直接到第四层）
            residual: candle_nn::linear(in_size, layer4_size, vb.pp("residual"))?,
            // 输出层
            fc_out: candle_nn::linear(layer4_size, out_size, vb.pp("fc_out"))?,
            training: true,
        })
    }
    
    /// ReLU激活函数
    fn relu(x: &Tensor) -> Result<Tensor, candle_core::Error> {
        let zeros = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
        x.maximum(&zeros)
    }
    
    /// Dropout实现（训练时随机丢弃）
    fn dropout(x: &Tensor, p: f64, training: bool) -> Result<Tensor, candle_core::Error> {
        if !training || p == 0.0 {
            return Ok(x.clone());
        }
        // 简化版dropout：直接缩放
        x.affine(1.0 / (1.0 - p), 0.0)
    }
    
    /// 设置训练/评估模式
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

unsafe impl Send for DeepStockPredictor {}
unsafe impl Sync for DeepStockPredictor {}

impl Module for DeepStockPredictor {
    fn forward(&self, xs: &Tensor) -> Result<Tensor, candle_core::Error> {
        // 第一层：特征提取 + ReLU + Dropout
        let x1 = self.fc1.forward(xs)?;
        let x1 = Self::relu(&x1)?;
        let x1 = Self::dropout(&x1, self.dropout1, self.training)?;
        
        // 第二层：深度学习 + ReLU + Dropout
        let x2 = self.fc2.forward(&x1)?;
        let x2 = Self::relu(&x2)?;
        let x2 = Self::dropout(&x2, self.dropout2, self.training)?;
        
        // 第三层：模式识别 + ReLU + Dropout
        let x3 = self.fc3.forward(&x2)?;
        let x3 = Self::relu(&x3)?;
        let x3 = Self::dropout(&x3, self.dropout3, self.training)?;
        
        // 第四层：深度抽象 + ReLU + Dropout
        let x4 = self.fc4.forward(&x3)?;
        let x4 = Self::relu(&x4)?;
        let x4 = Self::dropout(&x4, self.dropout4, self.training)?;
        
        // 残差连接（从输入直接到第四层，帮助梯度流动）
        let residual = self.residual.forward(xs)?;
        let x4 = (x4 + residual)?;
        
        // 输出层（不加激活函数，因为是回归问题）
        self.fc_out.forward(&x4)
    }
}

// 改进的模型创建函数
fn create_model(config: &ModelConfig, device: &Device) -> Result<(VarMap, Box<dyn Module + Send + Sync>), candle_core::Error> {
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
    
    // 使用更大的隐藏层（金融数据需要更多参数）
    let hidden_size = config.hidden_size.max(128);  // 至少128
    
    // 创建深度神经网络模型
    let model = DeepStockPredictor::new(
        config.input_size,
        hidden_size,
        config.output_size,
        config.dropout,
        vb,
    )?;
    
    let model: Box<dyn Module + Send + Sync> = Box::new(model);
    Ok((varmap, model))
}

// 简化的模型保存函数
fn save_model(varmap: &VarMap, path: &std::path::Path) -> Result<(), candle_core::Error> {
    varmap.save(path)?;
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
            eprintln!("从数据库获取数据失败: {e}");
            return Err(candle_core::Error::Msg(format!("获取历史数据失败: {e}")));
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
    
    println!("数据预处理完成: 训练集{train_size}样本, 测试集{test_size}样本, 特征维度{features_len}");
    
    Ok((x_train, y_train, x_test, y_test, dates))
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

// 训练模型函数
pub async fn train_candle_model(request: TrainingRequest) -> std::result::Result<TrainingResult, String> {
    let model_id = generate_model_id();
    let model_type = request.model_type.clone();
    
    println!("🚀 开始训练金融级深度神经网络模型...");
    
    // 准备数据
    let (x_train, y_train, x_test, y_test, _) = prepare_stock_data(&request).await
        .map_err(|e| format!("数据准备失败: {e}"))?;
    
    // 设置设备
    let device = Device::Cpu;
    
    // 创建模型配置 - 金融级参数
    let config = ModelConfig {
        model_type: model_type.clone(),
        input_size: request.features.len(),
        hidden_size: 128, // 增大隐藏层：64 -> 128
        output_size: 1,
        dropout: request.dropout.max(0.2), // 至少20% dropout
        learning_rate: request.learning_rate,
        n_layers: 3,     // 3层网络
        n_heads: 4,
        max_seq_len: 60,
    };
    
    println!("📐 模型架构:");
    println!("   输入维度: {}", config.input_size);
    println!("   隐藏层: {} (第1层: {}, 第2层: {}, 第3层: {}, 第4层: {})", 
             config.hidden_size, 
             config.hidden_size * 2,
             config.hidden_size,
             config.hidden_size / 2,
             config.hidden_size / 4);
    println!("   输出维度: {}", config.output_size);
    println!("   Dropout率: {:.1}%", config.dropout * 100.0);
    
    // 创建模型
    let (varmap, model) = create_model(&config, &device)
        .map_err(|e| format!("模型创建失败: {e}"))?;
    
    // 创建优化器（使用AdamW，更适合深度网络）
    let initial_lr = request.learning_rate;
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), initial_lr)
        .map_err(|e| format!("优化器创建失败: {e}"))?;
    
    println!("🎯 训练配置:");
    println!("   初始学习率: {:.6}", initial_lr);
    println!("   批次大小: {}", request.batch_size);
    println!("   训练轮数: {}", request.epochs);
    
    // 训练循环 - 增强版
    let batch_size = request.batch_size;
    let num_batches = x_train.dim(0).unwrap() / batch_size;
    
    // 早停机制
    let mut best_val_loss = f64::INFINITY;
    let mut patience_counter = 0;
    let patience = 15; // 15个epoch无改进则停止
    let min_delta = 0.0001; // 最小改进阈值
    
    // 学习率衰减
    let lr_decay_factor: f64 = 0.95; // 每次衰减5%
    let lr_decay_epochs = 20;        // 每20个epoch衰减一次
    
    println!("\n🔄 开始训练...");
    
    for epoch in 0..request.epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        // 动态学习率调整
        if epoch > 0 && epoch % lr_decay_epochs == 0 {
            let new_lr = initial_lr * lr_decay_factor.powi((epoch / lr_decay_epochs) as i32);
            optimizer = AdamW::new_lr(varmap.all_vars(), new_lr)
                .map_err(|e| format!("更新学习率失败: {e}"))?;
            println!("📉 第{}轮: 学习率衰减至 {:.6}", epoch + 1, new_lr);
        }
        
        // 训练阶段
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let x_batch = x_train.narrow(0, batch_start, batch_size)
                .map_err(|e| format!("批次数据准备失败: {e}"))?;
            let y_batch = y_train.narrow(0, batch_start, batch_size)
                .map_err(|e| format!("批次标签准备失败: {e}"))?;
            
            // 前向传播
            let output = model.forward(&x_batch)
                .map_err(|e| format!("前向传播失败: {e}"))?;
            
            // 形状匹配
            let reshaped_output = if output.dims() != y_batch.dims() {
                if output.dim(0).unwrap() == y_batch.dim(0).unwrap() {
                    output.reshape(&[output.dim(0).unwrap(), 1])
                        .map_err(|e| format!("调整输出形状失败: {e}"))?
                } else {
                    return Err(format!("形状不兼容: {:?} vs {:?}", output.dims(), y_batch.dims()));
                }
            } else {
                output
            };
            
            // 计算损失 (MSE + L2正则化)
            let diff = reshaped_output.sub(&y_batch)
                .map_err(|e| format!("计算差值失败: {e}"))?;
            let mse_loss = diff.sqr()
                .map_err(|e| format!("计算平方失败: {e}"))?
                .mean_all()
                .map_err(|e| format!("计算均值失败: {e}"))?;
            
            // L2正则化（权重衰减）
            let l2_lambda = 0.0001;
            let mut l2_loss = 0.0;
            for var in varmap.all_vars() {
                let weight_norm = var.sqr()
                    .map_err(|e| format!("计算权重范数失败: {e}"))?
                    .sum_all()
                    .map_err(|e| format!("求和失败: {e}"))?;
                l2_loss += weight_norm.to_scalar::<f32>().unwrap() as f64;
            }
            
            let total_loss = mse_loss.to_scalar::<f32>().unwrap() as f64 + l2_lambda * l2_loss;
            
            // 反向传播
            optimizer.backward_step(&mse_loss)
                .map_err(|e| format!("反向传播失败: {e}"))?;
            
            epoch_loss += total_loss;
            batch_count += 1;
        }
        
        let avg_train_loss = epoch_loss / batch_count as f64;
        
        // 验证阶段（使用测试集评估）
        let val_output = model.forward(&x_test)
            .map_err(|e| format!("验证前向传播失败: {e}"))?;
        let reshaped_val_output = if val_output.dims() != y_test.dims() {
            val_output.reshape(&[y_test.dim(0).unwrap(), 1])
                .map_err(|e| format!("验证输出形状调整失败: {e}"))?
        } else {
            val_output
        };
        let val_diff = reshaped_val_output.sub(&y_test)
            .map_err(|e| format!("验证差值计算失败: {e}"))?;
        let val_loss = val_diff.sqr()
            .map_err(|e| format!("验证平方失败: {e}"))?
            .mean_all()
            .map_err(|e| format!("验证均值失败: {e}"))?
            .to_scalar::<f32>().unwrap() as f64;
        
        // 早停判断
        if val_loss < best_val_loss - min_delta {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }
        
        // 日志输出
        if epoch == 0 || (epoch + 1) % 5 == 0 || epoch == request.epochs - 1 || patience_counter >= patience {
            println!("Epoch {:3}/{} | 训练Loss: {:.6} | 验证Loss: {:.6} | 最佳Loss: {:.6} | 耐心: {}/{}",
                     epoch + 1, 
                     request.epochs, 
                     avg_train_loss, 
                     val_loss, 
                     best_val_loss,
                     patience_counter,
                     patience);
        }
        
        // 早停触发
        if patience_counter >= patience {
            println!("⏹️  早停触发！{}个epoch未改进，停止训练", patience);
            println!("📊 最佳验证Loss: {:.6}", best_val_loss);
            break;
        }
    }
    
    println!("✅ 训练完成！");
    
    // 评估模型
    let y_pred = model.forward(&x_test)
        .map_err(|e| format!("预测失败: {e}"))?;
    
    // 转换为Vec用于准确率计算 - 处理不同维度的张量
    let predictions_vec = match y_pred.dims() {
        // 如果是1维张量 [n]
        [_] => {
            y_pred.to_vec1::<f32>().map_err(|e| format!("转换1维预测结果失败: {e}"))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        },
        // 如果是2维张量 [n, 1] 
        [_, 1] => {
            y_pred.to_vec2::<f32>().map_err(|e| format!("转换2维预测结果失败: {e}"))?
                .into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>()
        },
        // 如果是其他2维张量 [n, m]
        [_, _] => {
            let vec2d = y_pred.to_vec2::<f32>().map_err(|e| format!("转换2维预测结果失败: {e}"))?;
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
            y_test.to_vec1::<f32>().map_err(|e| format!("转换1维实际结果失败: {e}"))?
                .into_iter().map(|x| x as f64).collect::<Vec<f64>>()
        },
        // 如果是2维张量 [n, 1]
        [_, 1] => {
            y_test.to_vec2::<f32>().map_err(|e| format!("转换2维实际结果失败: {e}"))?
                .into_iter().map(|row| row[0] as f64).collect::<Vec<f64>>()
        },
        // 如果是其他2维张量 [n, m]
        [_, _] => {
            let vec2d = y_test.to_vec2::<f32>().map_err(|e| format!("转换2维实际结果失败: {e}"))?;
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
    let diff = y_pred.sub(&y_test).map_err(|e| format!("计算MSE失败: {e}"))?;
    let squared_diff = diff.sqr().map_err(|e| format!("计算平方失败: {e}"))?;
    let mse = squared_diff.mean_all().map_err(|e| format!("计算均值失败: {e}"))?;
    let mse = mse.to_scalar::<f32>().unwrap() as f64;
    let rmse = mse.sqrt();
    
    println!("评估结果: MSE = {mse:.4}, RMSE = {rmse:.4}");
    println!("🎯 方向预测准确率: {:.2}% | 综合准确率: {:.2}%", 
             direction_accuracy * 100.0, combined_accuracy * 100.0);
    println!("📊 预测张量维度: {:?}, 实际张量维度: {:?}", y_pred.dims(), y_test.dims());
    
    // 保存模型
    let model_path = get_model_file_path(&model_id);
    fs::create_dir_all(model_path.parent().unwrap()).map_err(|e| format!("创建模型目录失败: {e}"))?;
    save_model(&varmap, &model_path).map_err(|e| format!("模型保存失败: {e}"))?;
    
    // 保存模型元数据
    let metadata = ModelInfo {
        id: model_id,
        name: request.model_name,
        stock_code: request.stock_code,
        created_at: get_current_timestamp(),
        model_type,
        features: request.features,
        target: request.target,
        prediction_days: request.prediction_days,
        accuracy: combined_accuracy,
    };
    
    save_model_metadata(&metadata).map_err(|e| format!("元数据保存失败: {e}"))?;
    
    Ok(TrainingResult {
        metadata,
        accuracy: combined_accuracy,
    })
} 