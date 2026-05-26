//! Candle MLP 网络定义与训练

use super::features::FEATURE_DIM;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use std::path::Path;

/// 隐藏层维度
pub const HIDDEN: usize = 16;

/// 三层感知机：FEATURE_DIM → HIDDEN → HIDDEN → 1（回归次日收益率）
pub struct Mlp {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(FEATURE_DIM, HIDDEN, vb.pp("l1"))?,
            l2: linear(HIDDEN, HIDDEN, vb.pp("l2"))?,
            out: linear(HIDDEN, 1, vb.pp("out"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

/// 训练评估结果
pub struct TrainOutcome {
    pub direction_accuracy: f64,
    pub mae: f64,
    pub rmse: f64,
    pub test_samples: usize,
}

/// 训练 MLP 并保存权重到 `save_path`（safetensors）。
///
/// - `features`：扁平 n×FEATURE_DIM
/// - `labels`：n（次日收益率%）
/// - 按 `split` 切分训练/测试，返回测试集上的方向准确率与误差。
pub fn train_and_save(
    features: &[f32],
    labels: &[f32],
    n: usize,
    epochs: usize,
    learning_rate: f64,
    split: f64,
    save_path: &Path,
) -> Result<TrainOutcome, String> {
    if n < 20 {
        return Err(format!("样本不足，无法训练（n={n}）"));
    }
    let device = Device::Cpu;
    let split = split.clamp(0.5, 0.95);
    let n_train = ((n as f64 * split) as usize).clamp(10, n - 1);
    let n_test = n - n_train;

    let to_tensor = |feats: &[f32], rows: usize| -> Result<Tensor, String> {
        Tensor::from_vec(feats.to_vec(), (rows, FEATURE_DIM), &device).map_err(|e| e.to_string())
    };
    let to_label = |labs: &[f32], rows: usize| -> Result<Tensor, String> {
        Tensor::from_vec(labs.to_vec(), (rows, 1), &device).map_err(|e| e.to_string())
    };

    let x_train = to_tensor(&features[..n_train * FEATURE_DIM], n_train)?;
    let y_train = to_label(&labels[..n_train], n_train)?;
    let x_test = to_tensor(&features[n_train * FEATURE_DIM..], n_test)?;

    // 初始化网络与优化器
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mlp = Mlp::new(vb).map_err(|e| e.to_string())?;
    let mut optimizer = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: learning_rate.max(1e-5),
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;

    // 训练循环（全批量梯度下降，MSE 损失）
    for _ in 0..epochs.max(1) {
        let pred = mlp.forward(&x_train).map_err(|e| e.to_string())?;
        let loss = candle_nn::loss::mse(&pred, &y_train).map_err(|e| e.to_string())?;
        optimizer.backward_step(&loss).map_err(|e| e.to_string())?;
    }

    // 测试集评估
    let pred_test = mlp.forward(&x_test).map_err(|e| e.to_string())?;
    let preds: Vec<f32> = pred_test
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let actuals = &labels[n_train..];

    let mut direction_correct = 0usize;
    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    for (p, a) in preds.iter().zip(actuals.iter()) {
        let (p, a) = (*p as f64, *a as f64);
        if (p > 0.0 && a > 0.0) || (p < 0.0 && a < 0.0) {
            direction_correct += 1;
        }
        let err = (p - a).abs();
        abs_sum += err;
        sq_sum += err * err;
    }
    let count = preds.len().max(1) as f64;

    // 保存权重
    if let Some(parent) = save_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    varmap.save(save_path).map_err(|e| e.to_string())?;

    Ok(TrainOutcome {
        direction_accuracy: direction_correct as f64 / count,
        mae: abs_sum / count,
        rmse: (sq_sum / count).sqrt(),
        test_samples: preds.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_save_runs() {
        // 构造一个可学习的线性关系：label ≈ feature0 * 10
        let n = 80;
        let mut features = Vec::with_capacity(n * FEATURE_DIM);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let f0 = (i as f32 / n as f32) - 0.5;
            for j in 0..FEATURE_DIM {
                features.push(if j == 0 { f0 } else { 0.0 });
            }
            labels.push(f0 * 10.0);
        }

        let path = std::env::temp_dir()
            .join(format!("biga_test_model_{}.safetensors", std::process::id()));
        let outcome =
            train_and_save(&features, &labels, n, 100, 0.05, 0.8, &path).expect("training failed");

        assert!(outcome.direction_accuracy.is_finite());
        assert!((0.0..=1.0).contains(&outcome.direction_accuracy));
        assert!(outcome.mae.is_finite());
        assert!(outcome.test_samples > 0);
        assert!(path.exists(), "权重文件应已保存");

        std::fs::remove_file(&path).ok();
    }
}
