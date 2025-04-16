use crate::db::models::{ModelEvaluationMetrics, StockPredictionModelInfo};
use crate::prediction::features::FeatureSet;
use chrono::{Duration, NaiveDate, Local, Datelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ndarray::ArrayView1;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelTrainingConfig {
    pub model_type: String,
    pub parameters: serde_json::Value,
    pub features: Vec<String>,
    pub lookback_days: usize,
    pub train_test_split: f64,
    pub normalization_params: HashMap<usize, (f64, f64)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedModel {
    pub config: ModelTrainingConfig,
    pub model_data: Vec<u8>,
    pub metrics: ModelEvaluationMetrics,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelPredictionInput {
    pub model_info: StockPredictionModelInfo,
    pub model_data: Vec<u8>,
    pub normalization_params: HashMap<usize, (f64, f64)>,
    pub features: Vec<String>,
    pub lookback_days: usize,
    pub feature_set: FeatureSet,
}

pub fn calculate_evaluation_metrics(
    predictions: ArrayView1<f64>,
    targets: ArrayView1<f64>,
) -> ModelEvaluationMetrics {
    let n = predictions.len() as f64;
    
    // 计算 RMSE (均方根误差)
    let squared_errors: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &actual)| (pred - actual).powi(2))
        .sum();
    let rmse = (squared_errors / n).sqrt();
    
    // 计算 MAE (平均绝对误差)
    let absolute_errors: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &actual)| (pred - actual).abs())
        .sum();
    let mae = absolute_errors / n;
    
    // 计算 R-squared (决定系数)
    let mean_actual = targets.mean().unwrap_or(0.0);
    let total_sum_squares: f64 = targets
        .iter()
        .map(|&actual| (actual - mean_actual).powi(2))
        .sum();
    let r_squared = 1.0 - (squared_errors / total_sum_squares);
    
    // 计算分类指标 (假设我们预测的是涨跌方向)
    let true_positives = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &actual)| pred > 0.0 && actual > 0.0)
        .count() as f64;
    
    let true_negatives = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &actual)| pred <= 0.0 && actual <= 0.0)
        .count() as f64;
    
    let false_positives = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &actual)| pred > 0.0 && actual <= 0.0)
        .count() as f64;
    
    let false_negatives = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &actual)| pred <= 0.0 && actual > 0.0)
        .count() as f64;
    
    // 计算准确率
    let accuracy = (true_positives + true_negatives) / n;
    
    // 计算精确率
    let precision = if true_positives + false_positives > 0.0 {
        true_positives / (true_positives + false_positives)
    } else {
        0.0
    };
    
    // 计算召回率
    let recall = if true_positives + false_negatives > 0.0 {
        true_positives / (true_positives + false_negatives)
    } else {
        0.0
    };
    
    // 计算 F1 分数
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    ModelEvaluationMetrics {
        rmse,
        mae,
        r_squared,
        accuracy,
        precision,
        recall,
        f1_score,
    }
}

pub fn generate_future_dates(start_date: NaiveDate, days: usize) -> Vec<NaiveDate> {
    let mut dates = Vec::with_capacity(days);
    let mut current_date = start_date;
    
    for _ in 0..days {
        current_date = current_date.checked_add_days(chrono::Days::new(1)).unwrap_or(current_date);
        // 跳过周末
        let weekday_num = current_date.weekday().num_days_from_monday();
        while weekday_num >= 5 { // 5 是周六, 6 是周日
            current_date = current_date.checked_add_days(chrono::Days::new(1)).unwrap_or(current_date);
            let new_weekday = current_date.weekday().num_days_from_monday();
            if new_weekday < 5 {
                break;
            }
        }
        dates.push(current_date);
    }
    
    dates
}

pub fn normalize_single_feature_set(
    feature_set: &FeatureSet,
    normalization_params: &HashMap<usize, (f64, f64)>,
) -> FeatureSet {
    let normalized_features = feature_set
        .features
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            if let Some(&(min, max)) = normalization_params.get(&i) {
                if max - min != 0.0 {
                    (value - min) / (max - min)
                } else {
                    0.0
                }
            } else {
                value // 如果没有标准化参数，保持原值
            }
        })
        .collect();

    FeatureSet {
        features: normalized_features,
        target: feature_set.target,
        date: feature_set.date,
        symbol: feature_set.symbol.clone(),
    }
} 