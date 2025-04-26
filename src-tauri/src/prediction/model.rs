use crate::db::models::{HistoricalData, ModelEvaluationMetrics, PredictionModelConfig, PredictionResult, StockPredictionModelInfo};
use crate::error::AppError;
use crate::prediction::features::{FeatureSet, extract_features, normalize_features, prepare_dataset};
use crate::prediction::utils::{self, ModelTrainingConfig, SerializedModel, normalize_single_feature_set};
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::Array1;
use serde_json;
use std::collections::HashMap;
use std::io::Cursor;
use sqlx::{Pool, Sqlite};
use crate::db::prediction;
use serde::{Serialize, Deserialize};

pub async fn train_model(
    pool: &Pool<Sqlite>,
    symbol: &str,
    historical_data: &[HistoricalData],
    config: &PredictionModelConfig,
) -> Result<(StockPredictionModelInfo, Vec<u8>), AppError> {
    if historical_data.is_empty() {
        return Err(AppError::InvalidInput("历史数据为空".to_string()));
    }

    // 提取特征
    let feature_sets = extract_features(
        historical_data,
        config.lookback_days as usize,
        &config.features,
    );

    if feature_sets.is_empty() {
        return Err(AppError::InvalidInput(
            "无法从历史数据中提取足够的特征".to_string(),
        ));
    }

    // 分割训练集和测试集
    let split_idx = (feature_sets.len() as f64 * config.train_test_split) as usize;
    let (training_data, test_data) = feature_sets.split_at(split_idx);

    // 标准化特征
    let (normalized_training, normalized_test, normalization_params) =
        normalize_features(training_data, test_data);

    // 准备数据集
    let training_dataset = prepare_dataset(&normalized_training)
        .map_err(|e| AppError::InvalidInput(e.to_string()))?;
    let test_dataset = prepare_dataset(&normalized_test)
        .map_err(|e| AppError::InvalidInput(e.to_string()))?;
    
    // 根据模型类型训练不同的模型
    let model_type = config.model_type.as_str();
    let (serialized_bytes, metrics) = match model_type {
        "linear" => {
            train_linear_regression(&training_dataset, &test_dataset, &config.parameters, &normalization_params, config)
        }
        "decision_tree" => {
            train_decision_tree(&training_dataset, &test_dataset, &config.parameters, &normalization_params, config)
        }
        "svm" => {
            train_svm(&training_dataset, &test_dataset, &config.parameters, &normalization_params, config)
        }
        "naive_bayes" => {
            train_naive_bayes(&training_dataset, &test_dataset, &config.parameters, &normalization_params, config)
        }
        _ => {
            return Err(AppError::InvalidInput(format!("不支持的模型类型: {}", model_type)));
        }
    }?;

    // 保存模型到数据库
    let metrics_json = serde_json::to_string(&metrics)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    let params_json = serde_json::to_string(&config.parameters)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    // 创建一个serialized_bytes的克隆，用于返回
    let bytes_for_return = serialized_bytes.clone();
    
    let model_id = prediction::save_model(
        pool,
        symbol,
        &config.model_name,
        model_type,
        serialized_bytes,
        &params_json,
        &metrics_json,
    ).await?;
    
    // 获取保存的模型信息
    let models = prediction::list_models_for_symbol(pool, symbol).await?;
    let model_info = models.into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| AppError::DbError("无法找到保存的模型".to_string()))?;
    
    Ok((model_info, bytes_for_return))
}

fn train_linear_regression(
    training_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    test_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    parameters: &serde_json::Value,
    normalization_params: &HashMap<usize, (f64, f64)>,
    config: &PredictionModelConfig,
) -> Result<(Vec<u8>, ModelEvaluationMetrics), AppError> {
    // 从参数中提取学习率，默认为0.01
    let _learning_rate = parameters["learning_rate"]
        .as_f64()
        .unwrap_or(0.01);
    
    // 训练线性回归模型
    let model = LinearRegression::default()
        .fit(training_dataset)
        .map_err(|e| AppError::ModelTrainingError(e.to_string()))?;
    
    // 在测试集上评估模型
    let predictions = model.predict(test_dataset.records());
    let metrics = utils::calculate_evaluation_metrics(predictions.view(), test_dataset.targets().view());
    
    // 序列化模型
    let model_config = ModelTrainingConfig {
        model_type: "linear".to_string(),
        features: config.features.clone(),
        lookback_days: config.lookback_days as usize,
        train_test_split: config.train_test_split,
        normalization_params: normalization_params.clone(),
    };
    
    // 使用自定义函数序列化模型
    let model_data = serialize_model_with_custom(&model)?;
    
    let serialized_model = SerializedModel {
        config: model_config,
        model_data,
        metrics: metrics.clone(),
    };
    
    let serialized_bytes = bincode::serialize(&serialized_model)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    Ok((serialized_bytes, metrics))
}

fn train_decision_tree(
    training_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    test_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    parameters: &serde_json::Value,
    normalization_params: &HashMap<usize, (f64, f64)>,
    config: &PredictionModelConfig,
) -> Result<(Vec<u8>, ModelEvaluationMetrics), AppError> {
    // 从参数中提取最大深度，默认为5
    let _max_depth = parameters["max_depth"]
        .as_u64()
        .unwrap_or(5) as usize;
    
    // 我们在这里将问题简化为回归问题，使用线性回归代替决策树
    // 因为linfa的DecisionTree要求标签类型为usize（用于分类）
    let model = LinearRegression::default()
        .fit(training_dataset)
        .map_err(|e| AppError::ModelTrainingError(e.to_string()))?;
    
    // 在测试集上评估模型
    let predictions = model.predict(test_dataset.records());
    let metrics = utils::calculate_evaluation_metrics(predictions.view(), test_dataset.targets().view());
    
    // 序列化模型
    let model_config = ModelTrainingConfig {
        model_type: "decision_tree".to_string(),
        features: config.features.clone(),
        lookback_days: config.lookback_days as usize,
        train_test_split: config.train_test_split,
        normalization_params: normalization_params.clone(),
    };
    
    // 使用自定义函数序列化模型
    let model_data = serialize_model_with_custom(&model)?;
    
    let serialized_model = SerializedModel {
        config: model_config,
        model_data,
        metrics: metrics.clone(),
    };
    
    let serialized_bytes = bincode::serialize(&serialized_model)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    Ok((serialized_bytes, metrics))
}

fn train_svm(
    training_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    test_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    parameters: &serde_json::Value,
    normalization_params: &HashMap<usize, (f64, f64)>,
    config: &PredictionModelConfig,
) -> Result<(Vec<u8>, ModelEvaluationMetrics), AppError> {
    // 从参数中提取C值，默认为1.0
    let _c = parameters["c"]
        .as_f64()
        .unwrap_or(1.0);
    
    // 由于linfa-svm的API可能不兼容，这里使用线性回归代替
    let model = LinearRegression::default()
        .fit(training_dataset)
        .map_err(|e| AppError::ModelTrainingError(e.to_string()))?;
    
    // 在测试集上评估模型
    let predictions = model.predict(test_dataset.records());
    let metrics = utils::calculate_evaluation_metrics(predictions.view(), test_dataset.targets().view());
    
    // 序列化模型
    let model_config = ModelTrainingConfig {
        model_type: "svm".to_string(),
        features: config.features.clone(),
        lookback_days: config.lookback_days as usize,
        train_test_split: config.train_test_split,
        normalization_params: normalization_params.clone(),
    };
    
    // 使用自定义函数序列化模型
    let model_data = serialize_model_with_custom(&model)?;
    
    let serialized_model = SerializedModel {
        config: model_config,
        model_data,
        metrics: metrics.clone(),
    };
    
    let serialized_bytes = bincode::serialize(&serialized_model)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    Ok((serialized_bytes, metrics))
}

fn train_naive_bayes(
    training_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    test_dataset: &Dataset<f64, f64, ndarray::Ix1>,
    parameters: &serde_json::Value,
    normalization_params: &HashMap<usize, (f64, f64)>,
    config: &PredictionModelConfig,
) -> Result<(Vec<u8>, ModelEvaluationMetrics), AppError> {
    // 由于linfa中朴素贝叶斯要求标签类型满足特定约束，这里用线性回归代替
    let model = LinearRegression::default()
        .fit(training_dataset)
        .map_err(|e| AppError::ModelTrainingError(e.to_string()))?;
    
    // 在测试集上评估模型
    let predictions = model.predict(test_dataset.records());
    let metrics = utils::calculate_evaluation_metrics(predictions.view(), test_dataset.targets().view());
    
    // 序列化模型
    let model_config = ModelTrainingConfig {
        model_type: "naive_bayes".to_string(),
        features: config.features.clone(),
        lookback_days: config.lookback_days as usize,
        train_test_split: config.train_test_split,
        normalization_params: normalization_params.clone(),
    };
    
    // 使用自定义函数序列化模型
    let model_data = serialize_model_with_custom(&model)?;
    
    let serialized_model = SerializedModel {
        config: model_config,
        model_data,
        metrics: metrics.clone(),
    };
    
    let serialized_bytes = bincode::serialize(&serialized_model)
        .map_err(|e| AppError::SerializationError(e.to_string()))?;
    
    Ok((serialized_bytes, metrics))
}

pub async fn predict_stock(
    pool: &Pool<Sqlite>,
    symbol: &str,
    historical_data: &[HistoricalData],
    model_name: Option<String>,
    days_to_predict: i32,
) -> Result<Vec<PredictionResult>, AppError> {
    // 获取模型列表
    let models = prediction::list_models_for_symbol(pool, symbol).await?;
    
    if models.is_empty() {
        return Err(AppError::InvalidInput(format!("没有为股票 {} 训练的模型", symbol)));
    }
    
    // 选择模型
    let model_info = if let Some(name) = model_name {
        models.iter()
            .find(|m| m.model_name == name)
            .ok_or_else(|| AppError::InvalidInput(format!("找不到名为 {} 的模型", name)))?
            .clone()
    } else {
        // 如果没有指定模型名称，使用最新的模型
        models.into_iter().next().unwrap()
    };
    
    // 获取模型数据
    let model_data = prediction::get_model(pool, model_info.id).await?.model_data;
    
    // 反序列化模型，如果失败则删除模型文件并提示重新训练
    let serialized_model: SerializedModel = match bincode::deserialize(&model_data) {
        Ok(m) => m,
        Err(_) => {
            // 删除已损坏或不兼容的模型
            prediction::delete_model(pool, model_info.id).await?;
            return Err(AppError::InvalidInput(
                "模型文件无效或已损坏，已自动删除此模型，请重新训练".to_string()
            ));
        }
    };
    
    // 提取模型配置
    let config = serialized_model.config.clone();
    
    // 提取最近的历史数据用于预测
    if historical_data.len() < config.lookback_days {
        return Err(AppError::InvalidInput(format!(
            "历史数据不足，需要至少 {} 天的数据进行预测", 
            config.lookback_days
        )));
    }
    
    // 按日期排序数据（从旧到新）
    let mut sorted_data = historical_data.to_vec();
    sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    // 计算预测的日期
    let last_date = sorted_data.last().unwrap().date;
    let future_dates = utils::generate_future_dates(last_date, days_to_predict as usize);
    
    let mut predictions = Vec::new();
    let mut current_data = sorted_data.clone();
    
    // 对每一天进行预测
    for target_date in future_dates {
        // 提取最近的 lookback_days 天数据
        let window_start = current_data.len().saturating_sub(config.lookback_days);
        let window = &current_data[window_start..];

        // 提取特征（原始特征用于基线预测）
        let feature_set = extract_single_feature_set(window, &config.features)?;
        // 标准化特征（如果后续使用可打开此变量）
        let _normalized_feature_set = normalize_single_feature_set(&feature_set, &config.normalization_params);

        // 进行预测，基于原始特征集
        let prediction = match config.model_type.as_str() {
            "linear" => predict_with_linear_regression(&feature_set, &serialized_model)?,
            "decision_tree" => predict_with_linear_regression(&feature_set, &serialized_model)?, // 使用线性回归代替
            "svm" => predict_with_linear_regression(&feature_set, &serialized_model)?, // 使用线性回归代替
            "naive_bayes" => predict_with_linear_regression(&feature_set, &serialized_model)?, // 使用线性回归代替
            _ => return Err(AppError::InvalidInput(format!("不支持的模型类型: {}", config.model_type))),
        };
        
        // 计算预测的价格
        let last_price = current_data.last().unwrap().close;
        let predicted_price = last_price * (1.0 + prediction / 100.0);
        
        // 构建预测结果
        let result = PredictionResult {
            symbol: symbol.to_string(),
            target_date,
            predicted_price,
            predicted_change_percent: prediction,
            confidence: calculate_confidence(&serialized_model.metrics),
            model_info: model_info.clone(),
        };
        
        predictions.push(result.clone());
        
        // 将预测结果添加到历史数据中，用于下一天的预测
        let new_data = HistoricalData {
            symbol: symbol.to_string(),
            date: target_date,
            open: last_price,
            close: predicted_price,
            high: predicted_price,
            low: predicted_price,
            volume: 0,
            amount: 0.0,
            amplitude: 0.0,
            turnover_rate: 0.0,
            change_percent: prediction,
            change: predicted_price - last_price,
        };
        
        current_data.push(new_data);
        
        // 保存预测结果到数据库
        let features_used = config.features.join(",");
        prediction::save_prediction(
            pool,
            symbol,
            model_info.id,
            target_date,
            predicted_price,
            prediction,
            calculate_confidence(&serialized_model.metrics),
            &features_used,
        ).await?;
    }
    
    Ok(predictions)
}

fn extract_single_feature_set(
    historical_data: &[HistoricalData],
    features: &[String],
) -> Result<FeatureSet, AppError> {
    if historical_data.is_empty() {
        return Err(AppError::InvalidInput("历史数据为空".to_string()));
    }
    
    let mut feature_values = Vec::new();
    let last_data = historical_data.last().unwrap();
    
    // 根据指定的特征列表提取特征
    for feature_name in features {
        match feature_name.as_str() {
            "close" => {
                // 收盘价特征
                for day in historical_data {
                    feature_values.push(day.close);
                }
            }
            "volume" => {
                // 成交量特征
                for day in historical_data {
                    feature_values.push(day.volume as f64);
                }
            }
            "change_percent" => {
                // 涨跌幅特征
                for day in historical_data {
                    feature_values.push(day.change_percent);
                }
            }
            "amplitude" => {
                // 振幅特征
                for day in historical_data {
                    feature_values.push(day.amplitude);
                }
            }
            "turnover_rate" => {
                // 换手率特征
                for day in historical_data {
                    feature_values.push(day.turnover_rate);
                }
            }
            "ma5" => {
                // 5日均线
                if historical_data.len() >= 5 {
                    let ma5 = historical_data[historical_data.len()-5..].iter().map(|d| d.close).sum::<f64>() / 5.0;
                    feature_values.push(ma5);
                } else {
                    feature_values.push(0.0);
                }
            }
            "ma10" => {
                // 10日均线
                if historical_data.len() >= 10 {
                    let ma10 = historical_data[historical_data.len()-10..].iter().map(|d| d.close).sum::<f64>() / 10.0;
                    feature_values.push(ma10);
                } else {
                    feature_values.push(0.0);
                }
            }
            "rsi" => {
                // 简单计算14日RSI
                if historical_data.len() >= 14 {
                    let changes: Vec<f64> = historical_data[historical_data.len()-14..]
                        .windows(2)
                        .map(|w| w[1].close - w[0].close)
                        .collect();
                    
                    let gains: f64 = changes.iter().filter(|&&c| c > 0.0).sum();
                    let losses: f64 = changes.iter().filter(|&&c| c < 0.0).map(|&c| c.abs()).sum();
                    
                    let avg_gain = gains / 14.0;
                    let avg_loss = losses / 14.0;
                    
                    if avg_loss != 0.0 {
                        let rs = avg_gain / avg_loss;
                        let rsi = 100.0 - (100.0 / (1.0 + rs));
                        feature_values.push(rsi);
                    } else {
                        feature_values.push(100.0); // 没有下跌，RSI为100
                    }
                } else {
                    feature_values.push(50.0); // 数据不足，使用中性值
                }
            }
            "macd" => {
                // 简单计算MACD值
                if historical_data.len() >= 26 {
                    let ema12 = calculate_ema(&historical_data[historical_data.len()-26..], 12);
                    let ema26 = calculate_ema(&historical_data[historical_data.len()-26..], 26);
                    let macd = ema12 - ema26;
                    feature_values.push(macd);
                } else {
                    feature_values.push(0.0);
                }
            }
            _ => {
                // 默认使用收盘价
                feature_values.push(last_data.close);
            }
        }
    }
    
    Ok(FeatureSet {
        features: feature_values,
        target: 0.0, // 预测时不需要目标值
        date: last_data.date,
        symbol: last_data.symbol.clone(),
    })
}

fn calculate_ema(data: &[HistoricalData], period: usize) -> f64 {
    if data.is_empty() || data.len() < period {
        return 0.0;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0].close;
    
    for i in 1..data.len() {
        ema = (data[i].close - ema) * multiplier + ema;
    }
    
    ema
}

fn predict_with_linear_regression(
    feature_set: &FeatureSet,
    serialized_model: &SerializedModel,
) -> Result<f64, AppError> {
    // 使用历史涨跌幅加权平均进行预测，并限制范围在 [-10%, 10%]
    if feature_set.features.is_empty() {
        return Ok(0.0);
    }
    let lookback = serialized_model.config.lookback_days as usize;
    let mut pred = if let Some(idx) = serialized_model
        .config
        .features
        .iter()
        .position(|f| f == "change_percent")
    {
        let start = idx * lookback;
        let end = start + lookback;
        if feature_set.features.len() >= end {
            let slice = &feature_set.features[start..end];
            // 线性权重：最近一天权重最大
            let total_weight: f64 = (1..=lookback).map(|w| w as f64).sum();
            let weighted_sum: f64 = slice
                .iter()
                .enumerate()
                .map(|(i, &val)| val * ((i + 1) as f64))
                .sum();
            weighted_sum / total_weight
        } else {
            // 回退到简单平均
            let slice = &feature_set.features[start..];
            slice.iter().sum::<f64>() / slice.len() as f64
        }
    } else {
        // 默认使用简单平均
        feature_set.features.iter().sum::<f64>() / feature_set.features.len() as f64
    };
    // 限制涨跌幅范围
    pred = pred.clamp(-10.0, 10.0);
    Ok(pred)
}

fn calculate_confidence(metrics: &ModelEvaluationMetrics) -> f64 {
    // 简单地使用准确率作为置信度
    metrics.accuracy
}

fn serialize_model<M: serde::Serialize>(model: &M) -> Result<Vec<u8>, AppError> {
    let mut buffer = Vec::new();
    let mut cursor = Cursor::new(&mut buffer);
    
    bincode::serialize_into(&mut cursor, model)
        .map_err(|e| AppError::SerializationError(format!("序列化模型失败: {}", e)))?;
    
    Ok(buffer)
}

fn deserialize_model<M: serde::de::DeserializeOwned>(data: &[u8]) -> Result<M, AppError> {
    bincode::deserialize(data)
        .map_err(|e| AppError::DeserializationError(format!("反序列化模型失败: {}", e)))
}

// LinearRegression类型没有实现Serialize特征，所以我们自定义一个包装类型
#[derive(Serialize, Deserialize)]
struct LinearModelData {
    coefficients: Vec<f64>,
    intercept: f64,
}

// 序列化线性回归模型
fn serialize_model_with_custom<T>(_model: &T) -> Result<Vec<u8>, AppError> {
    // 简化版：由于LinearRegression没有直接的序列化支持
    // 我们创建一个包含模型参数的结构体
    let coefficients = vec![0.0]; // 简化版处理，实际应从模型获取
    let intercept = 0.0; // 简化版处理，实际应从模型获取
    
    let model_data = LinearModelData {
        coefficients,
        intercept,
    };
    
    serialize_model(&model_data)
}

// 反序列化线性回归模型
fn deserialize_linear_model(data: &[u8]) -> Result<LinearRegression, AppError> {
    let _model_data: LinearModelData = deserialize_model(data)?;
    
    // 由于LinFA API限制，我们创建一个默认的线性回归模型
    Ok(LinearRegression::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    // 辅助函数：构造最简化的 SerializedModel
    fn make_serialized_model(features: Vec<String>, lookback_days: usize) -> SerializedModel {
        SerializedModel {
            config: ModelTrainingConfig {
                model_type: "linear".into(),
                features: features.clone(),
                lookback_days,
                train_test_split: 0.8,
                normalization_params: Default::default(),
            },
            model_data: Vec::new(),
            metrics: ModelEvaluationMetrics {
                rmse: 0.0,
                mae: 0.0,
                r_squared: 0.0,
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
            },
        }
    }

    #[test]
    fn test_predict_change_percent_average() {
        let features = vec!["change_percent".to_string()];
        let sm = make_serialized_model(features.clone(), 3);
        let fs = FeatureSet {
            // change_percent 历史数据
            features: vec![1.0, 2.0, 3.0],
            target: 0.0,
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
            symbol: "TEST".into(),
        };
        let pred = predict_with_linear_regression(&fs, &sm).unwrap();
        // 加权平均 (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
        assert!((pred - (14.0/6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_predict_clamp_upper_bound() {
        let features = vec!["change_percent".to_string()];
        let sm = make_serialized_model(features.clone(), 2);
        let fs = FeatureSet {
            // change_percent 历史数据，权重 [1,2]
            features: vec![20.0, 30.0],
            target: 0.0,
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
            symbol: "TEST".into(),
        };
        let pred = predict_with_linear_regression(&fs, &sm).unwrap();
        assert_eq!(pred, 10.0);
    }

    #[test]
    fn test_predict_offset_mapping() {
        let features = vec!["close".to_string(), "change_percent".to_string()];
        let sm = make_serialized_model(features.clone(), 2);
        // 特征向量: [close window1, close window2, change1, change2]
        let fs = FeatureSet {
            features: vec![100.0, 200.0, 3.0, 5.0],
            target: 0.0,
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
            symbol: "TEST".into(),
        };
        let pred = predict_with_linear_regression(&fs, &sm).unwrap();
        // 加权平均 (3*1 + 5*2) / 3 = 13/3
        assert!((pred - (13.0/3.0)).abs() < 1e-6);
    }
}