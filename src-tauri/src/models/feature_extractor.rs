use crate::db::models::HistoricalData as StockData;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use ndarray::Array1;
use std::collections::HashMap;

/// 特征提取器，用于从股票数据生成训练数据
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new() -> Self {
        FeatureExtractor
    }

    /// 根据传入的特征列表，从股票数据中提取特征
    /// 
    /// # 参数
    /// 
    /// * `data` - 原始股票数据
    /// * `features` - 要提取的特征列表
    /// * `target` - 目标特征
    /// * `prediction_days` - 预测天数，用于计算滞后特征
    /// 
    /// # 返回值
    /// 
    /// 返回处理后的特征矩阵和目标向量
    pub fn extract_features(
        &self,
        data: &[StockData],
        features: &[String],
        target: &str,
        prediction_days: u32,
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("股票数据为空"));
        }

        // 对数据按日期排序
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
        
        // 提取所有可能的特征
        let mut feature_vectors: Vec<Vec<f64>> = Vec::new();
        let mut target_values: Vec<f64> = Vec::new();
        
        for i in prediction_days as usize..sorted_data.len() {
            // 计算当前日期的特征
            let mut feature_vec = Vec::new();
            
            // 为每个请求的特征提取值
            for feature_name in features {
                match feature_name.as_str() {
                    "close" => {
                        feature_vec.push(sorted_data[i].close as f64);
                    },
                    "open" => {
                        feature_vec.push(sorted_data[i].open as f64);
                    },
                    "high" => {
                        feature_vec.push(sorted_data[i].high as f64);
                    },
                    "low" => {
                        feature_vec.push(sorted_data[i].low as f64);
                    },
                    "volume" => {
                        feature_vec.push(sorted_data[i].volume as f64);
                    },
                    "price_change" => {
                        let change = sorted_data[i].close - sorted_data[i].open;
                        feature_vec.push(change as f64);
                    },
                    "price_change_pct" => {
                        let change_pct = if sorted_data[i].open > 0.0 {
                            (sorted_data[i].close - sorted_data[i].open) / sorted_data[i].open
                        } else {
                            0.0
                        };
                        feature_vec.push(change_pct as f64);
                    },
                    // 移动平均线特征
                    "ma5" => {
                        if i >= 5 {
                            let ma5 = (0..5).map(|j| sorted_data[i - j].close as f64).sum::<f64>() / 5.0;
                            feature_vec.push(ma5);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64);
                        }
                    },
                    "ma10" => {
                        if i >= 10 {
                            let ma10 = (0..10).map(|j| sorted_data[i - j].close as f64).sum::<f64>() / 10.0;
                            feature_vec.push(ma10);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64);
                        }
                    },
                    "ma20" => {
                        if i >= 20 {
                            let ma20 = (0..20).map(|j| sorted_data[i - j].close as f64).sum::<f64>() / 20.0;
                            feature_vec.push(ma20);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64);
                        }
                    },
                    // 相对强弱指标 (RSI)
                    "rsi14" => {
                        if i >= 14 {
                            feature_vec.push(self.calculate_rsi(&sorted_data[i-14..=i], 14));
                        } else {
                            feature_vec.push(50.0); // 默认中间值
                        }
                    },
                    // 布林带
                    "bollinger_upper" => {
                        if i >= 20 {
                            let (upper, _, _) = self.calculate_bollinger_bands(&sorted_data[i-20..=i], 20, 2.0);
                            feature_vec.push(upper);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64 * 1.1); // 简单估计
                        }
                    },
                    "bollinger_middle" => {
                        if i >= 20 {
                            let (_, middle, _) = self.calculate_bollinger_bands(&sorted_data[i-20..=i], 20, 2.0);
                            feature_vec.push(middle);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64);
                        }
                    },
                    "bollinger_lower" => {
                        if i >= 20 {
                            let (_, _, lower) = self.calculate_bollinger_bands(&sorted_data[i-20..=i], 20, 2.0);
                            feature_vec.push(lower);
                        } else {
                            feature_vec.push(sorted_data[i].close as f64 * 0.9); // 简单估计
                        }
                    },
                    // 过去N天的收盘价
                    _ if feature_name.starts_with("close_lag") => {
                        if let Some(lag_days) = feature_name[9..].parse::<usize>().ok() {
                            if i >= lag_days {
                                feature_vec.push(sorted_data[i - lag_days].close as f64);
                            } else {
                                feature_vec.push(sorted_data[0].close as f64);
                            }
                        } else {
                            return Err(anyhow::anyhow!(format!("无效的特征名: {}", feature_name)));
                        }
                    },
                    // 默认情况
                    _ => {
                        return Err(anyhow::anyhow!(format!("未知的特征名: {}", feature_name)));
                    }
                }
            }
            
            // 提取目标值(未来prediction_days天的价格)
            if i + prediction_days as usize <= sorted_data.len() - 1 {
                let future_idx = i + prediction_days as usize;
                let target_value = match target {
                    "close" => sorted_data[future_idx].close as f64,
                    "open" => sorted_data[future_idx].open as f64,
                    "high" => sorted_data[future_idx].high as f64,
                    "low" => sorted_data[future_idx].low as f64,
                    "price_change" => (sorted_data[future_idx].close - sorted_data[future_idx].open) as f64,
                    "price_change_pct" => {
                        if sorted_data[future_idx].open > 0.0 {
                            ((sorted_data[future_idx].close - sorted_data[future_idx].open) / sorted_data[future_idx].open) as f64
                        } else {
                            0.0
                        }
                    },
                    _ => {
                        return Err(anyhow::anyhow!(format!("未知的目标特征: {}", target)));
                    }
                };
                
                feature_vectors.push(feature_vec);
                target_values.push(target_value);
            }
        }
        
        if feature_vectors.is_empty() || target_values.is_empty() {
            return Err(anyhow::anyhow!("无法生成足够的特征或目标数据"));
        }
        
        Ok((feature_vectors, target_values))
    }
    
    /// 用于预测的特征提取，不包括目标值提取
    pub fn extract_prediction_features(
        &self,
        data: &[StockData],
        features: &[String],
        prediction_days: u32,
    ) -> Result<Vec<Vec<f64>>> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("股票数据为空"));
        }

        // 对数据按日期排序
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
        
        // 获取最后N天的数据用于预测
        let required_days = prediction_days as usize + 20; // 确保有足够的数据计算各种指标
        let start_idx = if sorted_data.len() > required_days {
            sorted_data.len() - required_days
        } else {
            0
        };
        
        let prediction_data = &sorted_data[start_idx..];
        
        // 提取特征
        let mut feature_vectors: Vec<Vec<f64>> = Vec::new();
        
        for i in prediction_days as usize..prediction_data.len() {
            // 计算当前日期的特征
            let mut feature_vec = Vec::new();
            
            // 为每个请求的特征提取值
            for feature_name in features {
                match feature_name.as_str() {
                    "close" => {
                        feature_vec.push(prediction_data[i].close as f64);
                    },
                    "open" => {
                        feature_vec.push(prediction_data[i].open as f64);
                    },
                    "high" => {
                        feature_vec.push(prediction_data[i].high as f64);
                    },
                    "low" => {
                        feature_vec.push(prediction_data[i].low as f64);
                    },
                    "volume" => {
                        feature_vec.push(prediction_data[i].volume as f64);
                    },
                    "price_change" => {
                        let change = prediction_data[i].close - prediction_data[i].open;
                        feature_vec.push(change as f64);
                    },
                    "price_change_pct" => {
                        let change_pct = if prediction_data[i].open > 0.0 {
                            (prediction_data[i].close - prediction_data[i].open) / prediction_data[i].open
                        } else {
                            0.0
                        };
                        feature_vec.push(change_pct as f64);
                    },
                    // 移动平均线特征
                    "ma5" => {
                        if i >= 5 {
                            let ma5 = (0..5).map(|j| prediction_data[i - j].close as f64).sum::<f64>() / 5.0;
                            feature_vec.push(ma5);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64);
                        }
                    },
                    "ma10" => {
                        if i >= 10 {
                            let ma10 = (0..10).map(|j| prediction_data[i - j].close as f64).sum::<f64>() / 10.0;
                            feature_vec.push(ma10);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64);
                        }
                    },
                    "ma20" => {
                        if i >= 20 {
                            let ma20 = (0..20).map(|j| prediction_data[i - j].close as f64).sum::<f64>() / 20.0;
                            feature_vec.push(ma20);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64);
                        }
                    },
                    // 相对强弱指标 (RSI)
                    "rsi14" => {
                        if i >= 14 {
                            feature_vec.push(self.calculate_rsi(&prediction_data[i-14..=i], 14));
                        } else {
                            feature_vec.push(50.0); // 默认中间值
                        }
                    },
                    // 布林带
                    "bollinger_upper" => {
                        if i >= 20 {
                            let (upper, _, _) = self.calculate_bollinger_bands(&prediction_data[i-20..=i], 20, 2.0);
                            feature_vec.push(upper);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64 * 1.1); // 简单估计
                        }
                    },
                    "bollinger_middle" => {
                        if i >= 20 {
                            let (_, middle, _) = self.calculate_bollinger_bands(&prediction_data[i-20..=i], 20, 2.0);
                            feature_vec.push(middle);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64);
                        }
                    },
                    "bollinger_lower" => {
                        if i >= 20 {
                            let (_, _, lower) = self.calculate_bollinger_bands(&prediction_data[i-20..=i], 20, 2.0);
                            feature_vec.push(lower);
                        } else {
                            feature_vec.push(prediction_data[i].close as f64 * 0.9); // 简单估计
                        }
                    },
                    // 过去N天的收盘价
                    _ if feature_name.starts_with("close_lag") => {
                        if let Some(lag_days) = feature_name[9..].parse::<usize>().ok() {
                            if i >= lag_days {
                                feature_vec.push(prediction_data[i - lag_days].close as f64);
                            } else {
                                feature_vec.push(prediction_data[0].close as f64);
                            }
                        } else {
                            return Err(anyhow::anyhow!(format!("无效的特征名: {}", feature_name)));
                        }
                    },
                    // 默认情况
                    _ => {
                        return Err(anyhow::anyhow!(format!("未知的特征名: {}", feature_name)));
                    }
                }
            }
            
            feature_vectors.push(feature_vec);
        }
        
        if feature_vectors.is_empty() {
            return Err(anyhow::anyhow!("无法生成足够的特征数据"));
        }
        
        Ok(feature_vectors)
    }
    
    /// 计算相对强弱指标 (RSI)
    fn calculate_rsi(&self, data: &[StockData], period: usize) -> f64 {
        if data.len() <= period {
            return 50.0; // 默认中间值
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..=period {
            let change = data[i].close - data[i-1].close;
            if change > 0.0 {
                gains += change as f64;
            } else {
                losses -= change as f64; // 取绝对值
            }
        }
        
        if losses == 0.0 {
            return 100.0;
        }
        
        let rs = gains / losses;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        
        rsi
    }
    
    /// 计算布林带 (Bollinger Bands)
    fn calculate_bollinger_bands(&self, data: &[StockData], period: usize, num_std_dev: f64) -> (f64, f64, f64) {
        if data.len() <= period {
            let close = data.last().map(|d| d.close).unwrap_or(0.0) as f64;
            return (close * 1.1, close, close * 0.9);
        }
        
        // 计算移动平均线
        let ma = data[data.len()-period..].iter()
            .map(|d| d.close as f64)
            .sum::<f64>() / period as f64;
        
        // 计算标准差
        let variance = data[data.len()-period..].iter()
            .map(|d| {
                let diff = d.close as f64 - ma;
                diff * diff
            })
            .sum::<f64>() / period as f64;
        
        let std_dev = variance.sqrt();
        
        // 计算布林带上下轨
        let upper_band = ma + (std_dev * num_std_dev);
        let lower_band = ma - (std_dev * num_std_dev);
        
        (upper_band, ma, lower_band)
    }
    
    /// 标准化数据(Min-Max标准化)
    pub fn normalize_features(&self, features: &mut Vec<Vec<f64>>) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow::anyhow!("特征数据为空"));
        }
        
        let num_features = features[0].len();
        let num_samples = features.len();
        
        // 计算每个特征的最小值和最大值
        let mut min_values = vec![f64::MAX; num_features];
        let mut max_values = vec![f64::MIN; num_features];
        
        for sample in features.iter() {
            for (j, &value) in sample.iter().enumerate() {
                min_values[j] = min_values[j].min(value);
                max_values[j] = max_values[j].max(value);
            }
        }
        
        // 标准化数据
        for i in 0..num_samples {
            for j in 0..num_features {
                let range = max_values[j] - min_values[j];
                if range > f64::EPSILON {
                    features[i][j] = (features[i][j] - min_values[j]) / range;
                } else {
                    features[i][j] = 0.5; // 如果范围太小，设为中间值
                }
            }
        }
        
        Ok(())
    }
    
    /// 标准化目标值
    pub fn normalize_targets(&self, targets: &mut Vec<f64>) -> Result<(f64, f64)> {
        if targets.is_empty() {
            return Err(anyhow::anyhow!("目标数据为空"));
        }
        
        let min_value = targets.iter().fold(f64::MAX, |a, &b| a.min(b));
        let max_value = targets.iter().fold(f64::MIN, |a, &b| a.max(b));
        
        let range = max_value - min_value;
        if range > f64::EPSILON {
            for i in 0..targets.len() {
                targets[i] = (targets[i] - min_value) / range;
            }
        } else {
            for i in 0..targets.len() {
                targets[i] = 0.5;
            }
        }
        
        // 返回最小值和范围，以便后续反标准化
        Ok((min_value, range))
    }
    
    /// 反标准化预测值
    pub fn denormalize_prediction(&self, prediction: f64, min_value: f64, range: f64) -> f64 {
        if range > f64::EPSILON {
            prediction * range + min_value
        } else {
            min_value
        }
    }
} 