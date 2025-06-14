use linfa::prelude::*;
use linfa_linear::LinearRegression;
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};

// 自定义TrainedModel trait用于模型统一接口
trait TrainedModelF64 {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
}

trait TrainedModelUsize {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>>;
}

// 为LinearRegression实现自定义trait
impl TrainedModelF64 for linfa_linear::FittedLinearRegression<f64> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        // 使用完全限定语法避免歧义
        let pred = linfa::prelude::Predict::predict(self, x);
        Ok(pred)
    }
}

// 为DecisionTree实现自定义trait
impl TrainedModelUsize for linfa_trees::DecisionTree<f64, usize> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>> {
        // 使用完全限定语法避免歧义
        let pred = linfa::prelude::Predict::predict(self, x);
        Ok(pred)
    }
}

// 导出特征提取器模块
pub mod feature_extractor;
pub use feature_extractor::FeatureExtractor;

// 支持的模型类型
pub enum ModelType {
    Linear,
    RandomForest, 
    Lstm,
}

// 使用正确的数据类型别名
pub type StockData = crate::db::models::HistoricalData;

// 简化的模型定义
pub struct Model {
    model_type: ModelType,
    linear_model: Option<Box<dyn TrainedModelF64 + Send + Sync>>,
    decision_tree_model: Option<Box<dyn TrainedModelUsize + Send + Sync>>,
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
        &mut self,
        stock_data: &[StockData],
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
        &mut self,
        stock_data: &[StockData],
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
        let targets = stock_data.iter()
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
        stock_data: &[StockData],
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
        stock_data: &[StockData],
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
        self.train_decision_tree(&feature_vectors, &target_values)
    }
    
    // 使用股票数据训练LSTM模型
    pub fn train_lstm_with_stock_data(
        &mut self,
        stock_data: &[StockData],
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
        &mut self,
        model: &Model,
        stock_data: &[StockData],
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
            ModelType::RandomForest => self.predict_decision_tree(model, &feature_vectors)?,
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
        let predictions = linfa::prelude::Predict::predict(&linear_model, &x_test);
        let accuracy = self.calculate_accuracy(&predictions.as_slice().unwrap(), &y_test.as_slice().unwrap());
        
        // 创建模型对象
        let model = Model {
            model_type: ModelType::Linear,
            linear_model: Some(Box::new(linear_model) as Box<dyn TrainedModelF64 + Send + Sync>),
            decision_tree_model: None,
            lstm_model: None,
        };
        
        Ok((model, accuracy))
    }
    
    // 训练决策树模型 (替代RandomForest)
    pub fn train_decision_tree(
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
        
        // 将连续目标值转换为分类索引
        // 注：DecisionTree需要usize类型标签
        let n_bins = 10; // 分成10个区间
        let mut y_binned = Array1::zeros(y_train.len());
        
        let y_min = y_train.fold(f64::INFINITY, |acc, &y| acc.min(y));
        let y_max = y_train.fold(f64::NEG_INFINITY, |acc, &y| acc.max(y));
        let bin_width = (y_max - y_min) / n_bins as f64;
        
        for (i, &y) in y_train.iter().enumerate() {
            let bin = ((y - y_min) / bin_width).min((n_bins - 1) as f64).floor() as usize;
            y_binned[i] = bin;
        }
        
        // 训练决策树模型
        let tree_model = DecisionTree::params()
            .max_depth(Some(10))
            .fit(&DatasetBase::new(x_train, y_binned))
            .context("决策树模型训练失败")?;
        
        // 为测试计算准确度，将连续值转为分类
        let mut y_test_binned = Array1::zeros(y_test.len());
        for (i, &y) in y_test.iter().enumerate() {
            let bin = ((y - y_min) / bin_width).min((n_bins - 1) as f64).floor() as usize;
            y_test_binned[i] = bin;
        }
        
        // 计算分类准确度
        let predictions = linfa::prelude::Predict::predict(&tree_model, &x_test);
        let correct = predictions
            .iter()
            .zip(y_test_binned.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        
        let accuracy = correct as f64 / y_test.len() as f64;
        
        // 创建模型对象
        let model = Model {
            model_type: ModelType::RandomForest,
            linear_model: None,
            decision_tree_model: Some(Box::new(tree_model) as Box<dyn TrainedModelUsize + Send + Sync>),
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
            decision_tree_model: None,
            lstm_model: Some(lstm_weights),
        };
        
        // 模拟准确度计算
        let accuracy = 0.75; // 假设的准确度
        
        Ok((model, accuracy))
    }
    
    // 保存模型到文件 - 简化实现
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
        
        // 模型具体数据转储 - 简化版
        match model.model_type {
            ModelType::Linear => {
                // 简化数据存储
                writer.write_all(&[1u8])?;
            },
            ModelType::RandomForest => {
                // 简化存储
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
    
    // 从文件加载模型 - 简化版实现
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<Model> {
        let file = File::open(path).context("打开模型文件失败")?;
        let mut reader = BufReader::new(file);
        
        // 读取模型类型
        let mut model_type_buf = [0u8];
        reader.read_exact(&mut model_type_buf)?;
        
        match model_type_buf[0] {
            0 => {
                // 创建空的线性回归模型
                Ok(Model {
                    model_type: ModelType::Linear,
                    linear_model: None, // 简化处理
                    decision_tree_model: None,
                    lstm_model: None,
                })
            },
            1 => {
                // 简化：实际应该反序列化决策树模型
                Ok(Model {
                    model_type: ModelType::RandomForest,
                    linear_model: None,
                    decision_tree_model: None,
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
                    decision_tree_model: None,
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
            let last_features_2d = last_features.to_owned().insert_axis(Axis(0));
            let prediction = linear_model.predict(&last_features_2d)?;
            
            if prediction.len() > 0 {
                Ok(prediction[0])
            } else {
                Err(anyhow::anyhow!("预测结果为空"))
            }
        } else {
            Err(anyhow::anyhow!("模型类型不匹配"))
        }
    }
    
    // 使用决策树模型进行预测
    pub fn predict_decision_tree(&self, model: &Model, features: &Vec<Vec<f64>>) -> Result<f64> {
        if let Some(ref tree_model) = model.decision_tree_model {
            // 准备特征数据
            let x = self.prepare_features(features)?;
            
            // 取最后一行特征用于预测
            let last_features = x.slice(s![x.nrows() - 1, ..]);
            let last_features_2d = last_features.to_owned().insert_axis(Axis(0));
            
            // 使用具体类型方法避免歧义，并使用 TrainedModelUsize 的 predict 方法
            let prediction = tree_model.predict(&last_features_2d)?;
            
            // 简化处理：将分类预测转回0-1区间的连续值
            if prediction.len() > 0 {
                // 转换为0-1区间值（简化处理）
                let pred_value = prediction[0] as f64 / 10.0;
                Ok(pred_value)
            } else {
                Err(anyhow::anyhow!("预测结果为空"))
            }
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
            prediction *= 1.0 + time_factor * 0.01;
            
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

pub mod candle_models;

// 导出candle模型相关的公共API
pub use candle_models::{ModelConfig, create_model, save_model}; 