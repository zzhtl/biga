//! Candle 模型加载与预测

use super::features::{build_dataset_for_horizon, build_samples, FEATURE_DIM};
use super::network::Mlp;
use crate::db::models::HistoricalData;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use chrono::NaiveDate;
use std::path::Path;

/// 已加载的 ML 预测器
pub struct MlPredictor {
    mlp: Mlp,
    device: Device,
}

impl MlPredictor {
    /// 从 safetensors 权重文件加载
    pub fn load(path: &Path) -> Result<Self, String> {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        // 先用 VarBuilder 注册结构，再从文件加载权重
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mlp = Mlp::new(vb).map_err(|e| e.to_string())?;
        varmap.load(path).map_err(|e| e.to_string())?;
        Ok(Self { mlp, device })
    }

    /// 预测一组特征对应的模型训练周期收益率（%）。
    pub fn predict(&self, features: &[f32]) -> Result<f64, String> {
        let rows = features.len() / FEATURE_DIM;
        if rows == 0 {
            return Err("特征为空".to_string());
        }
        let x = Tensor::from_vec(features.to_vec(), (rows, FEATURE_DIM), &self.device)
            .map_err(|e| e.to_string())?;
        let pred = self.mlp.forward(&x).map_err(|e| e.to_string())?;
        let v: Vec<f32> = pred
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        Ok(*v.last().unwrap_or(&0.0) as f64)
    }
}

/// 在给定历史数据上评估已加载模型，返回 (方向准确率, mae, rmse, 样本数)
pub fn evaluate_on(historical: &[HistoricalData], predictor: &MlPredictor) -> (f64, f64, f64, usize) {
    evaluate_on_horizon(historical, predictor, 1)
}

/// 在给定历史数据上按指定 horizon 评估已加载模型。
pub fn evaluate_on_horizon(
    historical: &[HistoricalData],
    predictor: &MlPredictor,
    horizon: usize,
) -> (f64, f64, f64, usize) {
    let (features, labels, n) = build_dataset_for_horizon(historical, horizon);
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }

    evaluate_predictions(&features, &labels, n, |feat| predictor.predict(feat))
}

/// 在训练结束日之后的样本上按指定 horizon 评估已加载模型。
pub fn evaluate_on_horizon_after(
    historical: &[HistoricalData],
    predictor: &MlPredictor,
    horizon: usize,
    min_feature_date: NaiveDate,
) -> (f64, f64, f64, usize) {
    let (features, labels, n) =
        build_evaluation_dataset_after(historical, horizon, min_feature_date);
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }

    evaluate_predictions(&features, &labels, n, |feat| predictor.predict(feat))
}

fn build_evaluation_dataset_after(
    historical: &[HistoricalData],
    horizon: usize,
    min_feature_date: NaiveDate,
) -> (Vec<f32>, Vec<f32>, usize) {
    let samples = build_samples(historical, horizon);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for sample in samples {
        if sample.date <= min_feature_date {
            continue;
        }
        features.extend_from_slice(&sample.features);
        labels.push((sample.fwd_return * 100.0) as f32);
    }

    let n = labels.len();
    (features, labels, n)
}

fn evaluate_predictions(
    features: &[f32],
    labels: &[f32],
    n: usize,
    mut predict_one: impl FnMut(&[f32]) -> Result<f64, String>,
) -> (f64, f64, f64, usize) {
    let mut evaluated = 0usize;
    let mut direction_correct = 0usize;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    for i in 0..n {
        let feat = &features[i * FEATURE_DIM..(i + 1) * FEATURE_DIM];
        let pred = match predict_one(feat) {
            Ok(p) => p,
            Err(_) => continue,
        };
        evaluated += 1;
        let actual = labels[i] as f64;
        if (pred > 0.0 && actual > 0.0) || (pred < 0.0 && actual < 0.0) {
            direction_correct += 1;
        }
        let err = (pred - actual).abs();
        abs_sum += err;
        sq_sum += err * err;
    }
    if evaluated == 0 {
        return (0.0, 0.0, 0.0, 0);
    }
    let count = evaluated as f64;
    (
        direction_correct as f64 / count,
        abs_sum / count,
        (sq_sum / count).sqrt(),
        evaluated,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_history(days: usize) -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..days)
            .map(|i| {
                let close = 10.0 + i as f64;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 10_000 + i as i64,
                    amount: close * 10_000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    #[test]
    fn test_evaluate_predictions_counts_only_successful_predictions() {
        let features = vec![0.0; FEATURE_DIM * 3];
        let labels = vec![1.0, -1.0, -2.0];
        let mut calls = 0usize;

        let (direction_accuracy, mae, rmse, evaluated) =
            evaluate_predictions(&features, &labels, 3, |_| {
                calls += 1;
                match calls {
                    1 => Ok(2.0),
                    2 => Err("prediction failed".to_string()),
                    _ => Ok(1.0),
                }
            });

        assert_eq!(evaluated, 2);
        assert_eq!(calls, 3);
        assert!((direction_accuracy - 0.5).abs() < f64::EPSILON);
        assert!((mae - 2.0).abs() < f64::EPSILON);
        assert!((rmse - 5.0f64.sqrt()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_evaluation_dataset_after_filters_training_window() {
        let historical = make_history(30);
        let min_feature_date = historical[22].date;
        let (_, labels, n) = build_evaluation_dataset_after(&historical, 1, min_feature_date);

        assert_eq!(n, 6);
        assert_eq!(labels.len(), 6);
    }
}
