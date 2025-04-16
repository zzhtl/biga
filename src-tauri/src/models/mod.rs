use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa_trees::RandomForest;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};

// 导出特征提取器模块
pub mod feature_extractor;
pub use feature_extractor::FeatureExtractor;

// 支持的模型类型
pub enum ModelType {
    Linear,
    RandomForest, 
    Lstm,
}

// 统一的模型接口
pub struct Model {
    model_type: ModelType,
    linear_model: Option<LinearRegression>,
    random_forest_model: Option<RandomForest>,
    lstm_model: Option<Vec<f64>>, // 简化版LSTM模型表示，实际应使用专用库
}

// 模型管理器
pub struct ModelManager {
    feature_extractor: FeatureExtractor,
}

impl ModelManager {
    pub fn new() -> Self {
        ModelManager {
            feature_extractor: FeatureExtractor::new(),
        }
    }
    
    // 获取特征提取器引用
    pub fn feature_extractor(&self) -> &FeatureExtractor {
        &self.feature_extractor
    }
    
    // 从股票数据中提取特征用于训练
    pub fn prepare_stock_data(
        &self,
        stock_data: &[crate::db::StockData],
        features: &[String],
        target: &str,
        prediction_days: u32
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        // 使用特征提取器提取特征
        let (mut features_data, mut targets) = self.feature_extractor.extract_features(
            stock_data,
            features,
            target,
            prediction_days
        )?;
        
        // 对特征和目标进行归一化处理
        self.feature_extractor.normalize_features(&mut features_data)?;
        let (min_target, range_target) = self.feature_extractor.normalize_targets(&mut targets)?;
        
        Ok((features_data, targets))
    }
    
    // 准备预测数据
    pub fn prepare_prediction_data(
        &self,
        stock_data: &[crate::db::StockData],
        features: &[String],
        prediction_days: u32
    ) -> Result<(Vec<Vec<f64>>, (f64, f64))> {
        // 使用特征提取器提取预测特征
        let mut features_data = self.feature_extractor.extract_prediction_features(
            stock_data,
            features,
            prediction_days
        )?;
        
        // 对特征进行归一化处理
        self.feature_extractor.normalize_features(&mut features_data)?;
        
        // 计算目标的归一化参数（用于后期反归一化）
        let mut targets = stock_data.iter()
            .map(|data| data.close as f64)
            .collect::<Vec<f64>>();
        
        if targets.is_empty() {
            return Err(anyhow::anyhow!("无法获取股票收盘价数据"));
        }
        
        let min_close = targets.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_close = targets.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range_close = max_close - min_close;
        
        Ok((features_data, (min_close, range_close)))
    }
    
    // 反归一化预测结果
    pub fn denormalize_prediction(&self, prediction: f64, min_value: f64, range: f64) -> f64 {
        self.feature_extractor.denormalize_prediction(prediction, min_value, range)
    }
    
    // 使用股票数据训练线性回归模型
    pub fn train_linear_regression_with_stock_data(
        &mut self,
        stock_data: &[crate::db::StockData],
        features: &[String],
        target: &str,
        prediction_days: u32
    ) -> Result<(Model, f64)> {
        // 使用特征提取器准备数据
        let (feature_vectors, target_values) = self.prepare_stock_data(
            stock_data,
            features,
            target,
            prediction_days
        )?;
        
        // 使用提取的特征训练模型
        self.train_linear_regression(&feature_vectors, &target_values)
    }
    
    // 使用股票数据训练随机森林模型
    pub fn train_random_forest_with_stock_data(
        &mut self,
        stock_data: &[crate::db::StockData],
        features: &[String],
        target: &str,
        prediction_days: u32
    ) -> Result<(Model, f64)> {
        // 使用特征提取器准备数据
        let (feature_vectors, target_values) = self.prepare_stock_data(
            stock_data,
            features,
            target,
            prediction_days
        )?;
        
        // 使用提取的特征训练模型
        self.train_random_forest(&feature_vectors, &target_values)
    }
    
    // 使用股票数据训练LSTM模型
    pub fn train_lstm_with_stock_data(
        &mut self,
        stock_data: &[crate::db::StockData],
        features: &[String],
        target: &str,
        prediction_days: u32
    ) -> Result<(Model, f64)> {
        // 使用特征提取器准备数据
        let (feature_vectors, target_values) = self.prepare_stock_data(
            stock_data,
            features,
            target,
            prediction_days
        )?;
        
        // 使用提取的特征训练模型
        self.train_lstm(&feature_vectors, &target_values, prediction_days)
    }
    
    // 使用股票数据进行预测
    pub fn predict_with_stock_data(
        &self,
        model: &Model,
        stock_data: &[crate::db::StockData],
        features: &[String],
        prediction_days: u32
    ) -> Result<f64> {
        // 准备预测数据
        let (feature_vectors, (min_value, range)) = self.prepare_prediction_data(
            stock_data,
            features,
            prediction_days
        )?;
        
        // 根据模型类型进行预测
        let normalized_prediction = match model.model_type {
            ModelType::Linear => self.predict_linear(model, &feature_vectors)?,
            ModelType::RandomForest => self.predict_random_forest(model, &feature_vectors)?,
            ModelType::Lstm => self.predict_lstm(model, &feature_vectors, prediction_days)?,
        };
        
        // 反归一化预测结果
        let prediction = self.denormalize_prediction(normalized_prediction, min_value, range);
        
        Ok(prediction)
    }
    
    // 训练线性回归模型
    pub fn train_linear_regression(
        &mut self, 
        features: &Vec<Vec<f64>>, 
        targets: &Vec<f64>
    ) -> Result<(Model, f64)> {
        // 转换输入数据为ndarray格式
        let (x, y) = self.prepare_data(features, targets)?;
        
        // 拆分训练集和测试集
        let (train_size, _) = x.dim();
        let test_split = train_size * 8 / 10; // 80% 训练数据
        
        let x_train = x.slice(s![0..test_split, ..]).to_owned();
        let y_train = y.slice(s![0..test_split]).to_owned();
        let x_test = x.slice(s![test_split.., ..]);
        let y_test = y.slice(s![test_split..]);
        
        // 训练线性回归模型
        let linear_model = LinearRegression::default()
            .fit(&DatasetBase::new(x_train, y_train))
            .context("线性回归模型训练失败")?;
        
        // 计算模型准确度
        let predictions = linear_model.predict(&x_test);
        let accuracy = self.calculate_accuracy(&predictions.as_slice().unwrap(), &y_test.as_slice().unwrap());
        
        // 创建模型对象
        let model = Model {
            model_type: ModelType::Linear,
            linear_model: Some(linear_model),
            random_forest_model: None,
            lstm_model: None,
        };
        
        Ok((model, accuracy))
    }
    
    // 训练随机森林模型
    pub fn train_random_forest(
        &mut self, 
        features: &Vec<Vec<f64>>, 
        targets: &Vec<f64>
    ) -> Result<(Model, f64)> {
        // 转换输入数据为ndarray格式
        let (x, y) = self.prepare_data(features, targets)?;
        
        // 拆分训练集和测试集
        let (train_size, _) = x.dim();
        let test_split = train_size * 8 / 10; // 80% 训练数据
        
        let x_train = x.slice(s![0..test_split, ..]).to_owned();
        let y_train = y.slice(s![0..test_split]).to_owned();
        let x_test = x.slice(s![test_split.., ..]);
        let y_test = y.slice(s![test_split..]);
        
        // 训练随机森林模型
        let random_forest = RandomForest::params()
            .max_depth(10)
            .n_trees(100)
            .fit(&DatasetBase::new(x_train, y_train))
            .context("随机森林模型训练失败")?;
        
        // 计算模型准确度
        let predictions = random_forest.predict(&x_test);
        let accuracy = self.calculate_accuracy(&predictions.as_slice().unwrap(), &y_test.as_slice().unwrap());
        
        // 创建模型对象
        let model = Model {
            model_type: ModelType::RandomForest,
            linear_model: None,
            random_forest_model: Some(random_forest),
            lstm_model: None,
        };
        
        Ok((model, accuracy))
    }
    
    // 训练LSTM模型 (简化版，实际应使用专用库如tch-rs)
    pub fn train_lstm(
        &mut self, 
        features: &Vec<Vec<f64>>, 
        targets: &Vec<f64>,
        prediction_days: u32
    ) -> Result<(Model, f64)> {
        // 注意：这是一个简化版的LSTM实现
        // 实际项目中应使用专业的深度学习库如tch-rs (PyTorch)
        
        // 模拟训练LSTM模型
        // 这里仅返回一个包含权重的向量作为模型表示
        let mut lstm_weights = Vec::new();
        
        // 计算一些简单的权重（实际应该是训练得到的）
        for (i, target) in targets.iter().enumerate() {
            if i < features.len() - 1 {
                let weight = target / (1.0 + features[i][0].abs());
                lstm_weights.push(weight);
            }
        }
        
        // 添加额外的预测日权重
        lstm_weights.push(prediction_days as f64 / 100.0);
        
        // 创建模型对象
        let model = Model {
            model_type: ModelType::Lstm,
            linear_model: None,
            random_forest_model: None,
            lstm_model: Some(lstm_weights),
        };
        
        // 模拟准确度计算
        let accuracy = 0.75; // 假设的准确度
        
        Ok((model, accuracy))
    }
    
    // 保存模型到文件
    pub fn save_model(&self, model: &Model, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path).context("创建模型文件失败")?;
        let mut writer = BufWriter::new(file);
        
        // 写入模型类型
        let model_type = match model.model_type {
            ModelType::Linear => 0u8,
            ModelType::RandomForest => 1u8,
            ModelType::Lstm => 2u8,
        };
        writer.write_all(&[model_type])?;
        
        // 根据模型类型保存相应数据
        match model.model_type {
            ModelType::Linear => {
                if let Some(ref linear_model) = model.linear_model {
                    // 保存线性模型的系数
                    let coefficients = linear_model.params().as_slice().unwrap();
                    let n_coeffs = coefficients.len() as u32;
                    writer.write_all(&n_coeffs.to_le_bytes())?;
                    
                    for coeff in coefficients {
                        writer.write_all(&coeff.to_le_bytes())?;
                    }
                    
                    // 保存截距
                    let intercept = linear_model.intercept();
                    writer.write_all(&intercept.to_le_bytes())?;
                }
            },
            ModelType::RandomForest => {
                // 简化：实际应该序列化随机森林模型
                // 这里仅写入一个标记
                writer.write_all(&[1u8])?;
            },
            ModelType::Lstm => {
                if let Some(ref lstm_weights) = model.lstm_model {
                    // 保存LSTM权重
                    let n_weights = lstm_weights.len() as u32;
                    writer.write_all(&n_weights.to_le_bytes())?;
                    
                    for weight in lstm_weights {
                        writer.write_all(&weight.to_le_bytes())?;
                    }
                }
            }
        }
        
        writer.flush()?;
        Ok(())
    }
    
    // 从文件加载模型
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<Model> {
        let file = File::open(path).context("打开模型文件失败")?;
        let mut reader = BufReader::new(file);
        
        // 读取模型类型
        let mut model_type_buf = [0u8];
        reader.read_exact(&mut model_type_buf)?;
        
        match model_type_buf[0] {
            0 => {
                // 线性回归模型
                // 读取系数数量
                let mut n_coeffs_buf = [0u8; 4];
                reader.read_exact(&mut n_coeffs_buf)?;
                let n_coeffs = u32::from_le_bytes(n_coeffs_buf) as usize;
                
                // 读取系数
                let mut coefficients = Vec::with_capacity(n_coeffs);
                for _ in 0..n_coeffs {
                    let mut coeff_buf = [0u8; 8];
                    reader.read_exact(&mut coeff_buf)?;
                    coefficients.push(f64::from_le_bytes(coeff_buf));
                }
                
                // 读取截距
                let mut intercept_buf = [0u8; 8];
                reader.read_exact(&mut intercept_buf)?;
                let intercept = f64::from_le_bytes(intercept_buf);
                
                // 创建线性回归模型
                let linear_model = LinearRegression::new(
                    Array1::from_vec(coefficients),
                    intercept
                );
                
                Ok(Model {
                    model_type: ModelType::Linear,
                    linear_model: Some(linear_model),
                    random_forest_model: None,
                    lstm_model: None,
                })
            },
            1 => {
                // 随机森林模型
                // 简化：实际应该反序列化随机森林模型
                Ok(Model {
                    model_type: ModelType::RandomForest,
                    linear_model: None,
                    random_forest_model: None, // 简化处理
                    lstm_model: None,
                })
            },
            2 => {
                // LSTM模型
                // 读取权重数量
                let mut n_weights_buf = [0u8; 4];
                reader.read_exact(&mut n_weights_buf)?;
                let n_weights = u32::from_le_bytes(n_weights_buf) as usize;
                
                // 读取权重
                let mut weights = Vec::with_capacity(n_weights);
                for _ in 0..n_weights {
                    let mut weight_buf = [0u8; 8];
                    reader.read_exact(&mut weight_buf)?;
                    weights.push(f64::from_le_bytes(weight_buf));
                }
                
                Ok(Model {
                    model_type: ModelType::Lstm,
                    linear_model: None,
                    random_forest_model: None,
                    lstm_model: Some(weights),
                })
            },
            _ => Err(anyhow::anyhow!("未知的模型类型"))
        }
    }
    
    // 使用线性回归模型进行预测
    pub fn predict_linear(&self, model: &Model, features: &Vec<Vec<f64>>) -> Result<f64> {
        if let Some(ref linear_model) = model.linear_model {
            // 准备特征数据
            let x = self.prepare_features(features)?;
            
            // 取最后一行特征用于预测
            let last_features = x.slice(s![x.nrows() - 1, ..]);
            
            // 预测
            let prediction = linear_model.predict(&last_features.insert_axis(Axis(0)));
            let predicted_value = prediction.into_raw_vec()[0];
            
            Ok(predicted_value)
        } else {
            Err(anyhow::anyhow!("模型类型不匹配"))
        }
    }
    
    // 使用随机森林模型进行预测
    pub fn predict_random_forest(&self, model: &Model, features: &Vec<Vec<f64>>) -> Result<f64> {
        if let Some(ref random_forest) = model.random_forest_model {
            // 准备特征数据
            let x = self.prepare_features(features)?;
            
            // 取最后一行特征用于预测
            let last_features = x.slice(s![x.nrows() - 1, ..]);
            
            // 预测
            let prediction = random_forest.predict(&last_features.insert_axis(Axis(0)));
            let predicted_value = prediction.into_raw_vec()[0];
            
            Ok(predicted_value)
        } else {
            Err(anyhow::anyhow!("模型类型不匹配"))
        }
    }
    
    // 使用LSTM模型进行预测
    pub fn predict_lstm(&self, model: &Model, features: &Vec<Vec<f64>>, days_ahead: u32) -> Result<f64> {
        if let Some(ref lstm_weights) = model.lstm_model {
            // 简化的LSTM预测
            // 实际应使用专业的深度学习库
            
            // 使用最近的特征和权重计算预测值
            let last_features = features.last().ok_or_else(|| anyhow::anyhow!("特征数据为空"))?;
            
            let mut prediction = 0.0;
            for (i, &feature) in last_features.iter().enumerate() {
                if i < lstm_weights.len() - 1 {
                    prediction += feature * lstm_weights[i];
                }
            }
            
            // 添加时间因子调整
            let time_factor = lstm_weights.last().unwrap_or(&0.01) * days_ahead as f64;
            prediction *= (1.0 + time_factor * 0.01);
            
            Ok(prediction)
        } else {
            Err(anyhow::anyhow!("模型类型不匹配"))
        }
    }
    
    // 辅助方法：准备输入数据
    fn prepare_data(
        &self, 
        features: &Vec<Vec<f64>>, 
        targets: &Vec<f64>
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        if features.is_empty() || targets.is_empty() {
            return Err(anyhow::anyhow!("输入数据为空"));
        }
        
        if features.len() != targets.len() {
            return Err(anyhow::anyhow!("特征和目标数量不匹配"));
        }
        
        let n_samples = features.len();
        let n_features = features[0].len();
        
        // 创建特征矩阵
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        for feature_vec in features {
            if feature_vec.len() != n_features {
                return Err(anyhow::anyhow!("特征维度不一致"));
            }
            x_data.extend_from_slice(feature_vec);
        }
        
        let x = Array2::from_shape_vec((n_samples, n_features), x_data)
            .context("创建特征矩阵失败")?;
        
        // 创建目标向量
        let y = Array1::from_vec(targets.clone());
        
        Ok((x, y))
    }
    
    // 辅助方法：仅准备特征数据
    fn prepare_features(&self, features: &Vec<Vec<f64>>) -> Result<Array2<f64>> {
        if features.is_empty() {
            return Err(anyhow::anyhow!("输入特征为空"));
        }
        
        let n_samples = features.len();
        let n_features = features[0].len();
        
        // 创建特征矩阵
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        for feature_vec in features {
            if feature_vec.len() != n_features {
                return Err(anyhow::anyhow!("特征维度不一致"));
            }
            x_data.extend_from_slice(feature_vec);
        }
        
        let x = Array2::from_shape_vec((n_samples, n_features), x_data)
            .context("创建特征矩阵失败")?;
        
        Ok(x)
    }
    
    // 计算模型准确度
    fn calculate_accuracy(&self, predictions: &[f64], actual: &[f64]) -> f64 {
        if predictions.is_empty() || actual.is_empty() || predictions.len() != actual.len() {
            return 0.0;
        }
        
        // 计算均方误差
        let mut total_squared_error = 0.0;
        let mut total_variance = 0.0;
        
        // 计算实际值的平均值
        let mean = actual.iter().sum::<f64>() / actual.len() as f64;
        
        for i in 0..predictions.len() {
            let error = predictions[i] - actual[i];
            total_squared_error += error * error;
            
            let variance = actual[i] - mean;
            total_variance += variance * variance;
        }
        
        if total_variance == 0.0 {
            return 0.0;
        }
        
        // 计算R²值
        let r_squared = 1.0 - (total_squared_error / total_variance);
        
        // 限制R²的范围
        r_squared.max(0.0).min(1.0)
    }
} 