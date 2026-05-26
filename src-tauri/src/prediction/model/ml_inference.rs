//! Candle 模型加载与预测

use super::features::{build_dataset, FEATURE_DIM};
use super::network::Mlp;
use crate::db::models::HistoricalData;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
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

    /// 预测一组特征对应的次日收益率（%）
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
    let (features, labels, n) = build_dataset(historical);
    if n == 0 {
        return (0.0, 0.0, 0.0, 0);
    }

    let mut direction_correct = 0usize;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    for i in 0..n {
        let feat = &features[i * FEATURE_DIM..(i + 1) * FEATURE_DIM];
        let pred = match predictor.predict(feat) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let actual = labels[i] as f64;
        if (pred > 0.0 && actual > 0.0) || (pred < 0.0 && actual < 0.0) {
            direction_correct += 1;
        }
        let err = (pred - actual).abs();
        abs_sum += err;
        sq_sum += err * err;
    }
    let count = n as f64;
    (
        direction_correct as f64 / count,
        abs_sum / count,
        (sq_sum / count).sqrt(),
        n,
    )
}
