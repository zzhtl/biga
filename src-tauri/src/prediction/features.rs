use crate::db::models::HistoricalData;
use chrono::NaiveDate;
use std::collections::HashMap;
use linfa::{self, prelude::*, Dataset};
use ndarray::{Array2, ArrayView1, Axis, Array1};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub features: Vec<f64>,
    pub target: f64,
    pub date: NaiveDate,
    pub symbol: String,
}

pub fn extract_features(
    historical_data: &[HistoricalData],
    lookback_days: usize,
    features: &[String],
) -> Vec<FeatureSet> {
    if historical_data.len() <= lookback_days {
        return Vec::new();
    }

    // 按日期排序数据（从旧到新）
    let mut sorted_data = historical_data.to_vec();
    sorted_data.sort_by(|a, b| a.date.cmp(&b.date));

    let mut feature_sets = Vec::new();

    // 为每个可能的时间窗口创建特征
    for i in lookback_days..sorted_data.len() {
        let window = &sorted_data[(i - lookback_days)..i];
        let current_data = &sorted_data[i];
        
        // 计算目标值：未来一天的涨跌百分比
        let target = if i < sorted_data.len() - 1 {
            sorted_data[i + 1].change_percent
        } else {
            // 对于最后一个数据点，我们没有下一天的数据，所以跳过
            continue;
        };

        let mut feature_values = Vec::new();
        
        // 根据指定的特征列表提取特征
        for feature_name in features {
            match feature_name.as_str() {
                "close" => {
                    // 收盘价特征
                    for day in window {
                        feature_values.push(day.close);
                    }
                }
                "volume" => {
                    // 成交量特征
                    for day in window {
                        feature_values.push(day.volume as f64);
                    }
                }
                "change_percent" => {
                    // 涨跌幅特征
                    for day in window {
                        feature_values.push(day.change_percent);
                    }
                }
                "amplitude" => {
                    // 振幅特征
                    for day in window {
                        feature_values.push(day.amplitude);
                    }
                }
                "turnover_rate" => {
                    // 换手率特征
                    for day in window {
                        feature_values.push(day.turnover_rate);
                    }
                }
                "ma5" => {
                    // 5日均线
                    if i >= 5 {
                        let ma5 = sorted_data[(i-5)..i].iter().map(|d| d.close).sum::<f64>() / 5.0;
                        feature_values.push(ma5);
                    } else {
                        feature_values.push(0.0);
                    }
                }
                "ma10" => {
                    // 10日均线
                    if i >= 10 {
                        let ma10 = sorted_data[(i-10)..i].iter().map(|d| d.close).sum::<f64>() / 10.0;
                        feature_values.push(ma10);
                    } else {
                        feature_values.push(0.0);
                    }
                }
                "rsi" => {
                    // 简单计算14日RSI
                    if i >= 14 {
                        let changes: Vec<f64> = sorted_data[(i-14)..i]
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
                    if i >= 26 {
                        let ema12 = calculate_ema(&sorted_data[(i-26)..i], 12);
                        let ema26 = calculate_ema(&sorted_data[(i-26)..i], 26);
                        let macd = ema12 - ema26;
                        feature_values.push(macd);
                    } else {
                        feature_values.push(0.0);
                    }
                }
                _ => {
                    // 默认使用收盘价
                    feature_values.push(current_data.close);
                }
            }
        }

        feature_sets.push(FeatureSet {
            features: feature_values,
            target,
            date: current_data.date,
            symbol: current_data.symbol.clone(),
        });
    }

    feature_sets
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

pub fn normalize_features(
    training_data: &[FeatureSet],
    test_data: &[FeatureSet],
) -> (Vec<FeatureSet>, Vec<FeatureSet>, HashMap<usize, (f64, f64)>) {
    if training_data.is_empty() || test_data.is_empty() {
        return (Vec::new(), Vec::new(), HashMap::new());
    }

    // 计算训练集的每个特征维度的最小值和最大值
    let feature_count = training_data[0].features.len();
    let mut min_values = vec![f64::MAX; feature_count];
    let mut max_values = vec![f64::MIN; feature_count];

    for feature_set in training_data {
        for (i, &value) in feature_set.features.iter().enumerate() {
            min_values[i] = min_values[i].min(value);
            max_values[i] = max_values[i].max(value);
        }
    }

    // 创建标准化参数映射，用于保存模型
    let mut normalization_params = HashMap::new();
    for i in 0..feature_count {
        normalization_params.insert(i, (min_values[i], max_values[i]));
    }

    // 标准化训练集
    let normalized_training = training_data
        .iter()
        .map(|set| {
            let normalized_features = set
                .features
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    if max_values[i] - min_values[i] != 0.0 {
                        (value - min_values[i]) / (max_values[i] - min_values[i])
                    } else {
                        0.0
                    }
                })
                .collect();

            FeatureSet {
                features: normalized_features,
                target: set.target,
                date: set.date,
                symbol: set.symbol.clone(),
            }
        })
        .collect();

    // 使用相同的参数标准化测试集
    let normalized_test = test_data
        .iter()
        .map(|set| {
            let normalized_features = set
                .features
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    if max_values[i] - min_values[i] != 0.0 {
                        (value - min_values[i]) / (max_values[i] - min_values[i])
                    } else {
                        0.0
                    }
                })
                .collect();

            FeatureSet {
                features: normalized_features,
                target: set.target,
                date: set.date,
                symbol: set.symbol.clone(),
            }
        })
        .collect();

    (normalized_training, normalized_test, normalization_params)
}

pub fn prepare_dataset(
    feature_sets: &[FeatureSet],
) -> Result<Dataset<f64, f64, ndarray::Ix1>, &'static str> {
    if feature_sets.is_empty() {
        return Err("特征集不能为空");
    }

    let feature_count = feature_sets[0].features.len();
    let sample_count = feature_sets.len();

    // 创建特征矩阵
    let mut features = Array2::zeros((sample_count, feature_count));
    let mut targets = Vec::with_capacity(sample_count);

    for (i, set) in feature_sets.iter().enumerate() {
        for (j, &value) in set.features.iter().enumerate() {
            features[[i, j]] = value;
        }
        targets.push(set.target);
    }

    let targets = Array1::from(targets);
    
    Ok(Dataset::new(features, targets))
}

pub fn compute_direction_accuracy(
    predictions: ArrayView1<f64>,
    targets: ArrayView1<f64>,
) -> f64 {
    let correct_direction = predictions
        .iter()
        .zip(targets.iter())
        .filter(|(&pred, &actual)| (pred > 0.0 && actual > 0.0) || (pred < 0.0 && actual < 0.0))
        .count();

    correct_direction as f64 / predictions.len() as f64
} 