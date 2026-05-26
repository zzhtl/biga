//! 模型训练模块
//!
//! 使用 candle 训练真实的 MLP 模型：特征显式包含量比与换手率，
//! 标签为次日收益率，按时间切分训练/测试集，返回测试集上的真实方向准确率。

use crate::db::{connection::create_temp_pool, repository::get_recent_historical_data};
use crate::prediction::model::features::{build_dataset, feature_names};
use crate::prediction::model::management::{
    generate_model_id, get_current_timestamp, get_model_file_path, save_model_metadata,
};
use crate::prediction::model::network::train_and_save;
use crate::prediction::types::{ModelInfo, TrainingRequest, TrainingResult};

/// 训练股票预测模型（真实 candle MLP）
pub async fn train_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    println!("🚀 开始训练模型: {}", request.model_name);
    println!("   股票代码: {}", request.stock_code);

    // 加载历史数据
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, 800, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    if historical.len() < 80 {
        return Err(format!(
            "历史数据不足（{}），训练至少需要 80 个交易日",
            historical.len()
        ));
    }

    // 构造数据集
    let (features, labels, n) = build_dataset(&historical);
    if n < 40 {
        return Err(format!("有效样本不足（{n}），无法训练"));
    }

    // 训练并保存权重
    let model_id = generate_model_id();
    let model_path = get_model_file_path(&model_id);
    let split = if request.train_test_split > 0.0 {
        request.train_test_split
    } else {
        0.8
    };
    let outcome = train_and_save(
        &features,
        &labels,
        n,
        request.epochs.max(50),
        request.learning_rate,
        split,
        &model_path,
    )?;

    let metadata = ModelInfo {
        id: model_id.clone(),
        name: request.model_name,
        stock_code: request.stock_code,
        created_at: get_current_timestamp(),
        model_type: "candle_mlp".to_string(),
        features: feature_names(),
        target: request.target,
        prediction_days: request.prediction_days,
        accuracy: outcome.direction_accuracy,
    };
    save_model_metadata(&metadata)?;

    println!(
        "✅ 训练完成：方向准确率 {:.1}%（测试样本 {}，MAE {:.3}）",
        outcome.direction_accuracy * 100.0,
        outcome.test_samples,
        outcome.mae
    );

    Ok(TrainingResult {
        metadata,
        accuracy: outcome.direction_accuracy,
    })
}

/// 重新训练模型：按新的超参数对同一标的重新训练并覆盖权重
pub async fn retrain_model(
    model_id: String,
    epochs: u32,
    _batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    use crate::prediction::model::management::load_model_metadata;

    let metadata = load_model_metadata(&model_id)?;

    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&metadata.stock_code, 800, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    let (features, labels, n) = build_dataset(&historical);
    if n < 40 {
        return Err(format!("有效样本不足（{n}），无法重训练"));
    }

    let model_path = get_model_file_path(&model_id);
    let outcome = train_and_save(
        &features,
        &labels,
        n,
        (epochs as usize).max(50),
        learning_rate,
        0.8,
        &model_path,
    )?;

    // 更新元数据中的准确率
    let mut updated = metadata;
    updated.accuracy = outcome.direction_accuracy;
    save_model_metadata(&updated)?;

    println!(
        "🔄 重训练完成：方向准确率 {:.1}%",
        outcome.direction_accuracy * 100.0
    );
    Ok(())
}
