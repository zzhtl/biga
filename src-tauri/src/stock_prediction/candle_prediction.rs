// 精简版candle_prediction.rs - 保持向后兼容性
// 主要功能已迁移到各自的专用模块中

use crate::stock_prediction::types::{TrainingRequest, TrainingResult, PredictionRequest, PredictionResponse};
use crate::stock_prediction::training::train_candle_model as internal_train;
use crate::stock_prediction::prediction::predict_with_candle as internal_predict;
use crate::stock_prediction::evaluation::evaluate_candle_model as internal_evaluate;
use crate::stock_prediction::types::EvaluationResult;
use crate::stock_prediction::model_management::{delete_model, load_model_metadata};

// 向后兼容的训练函数
pub async fn train_candle_model(request: TrainingRequest) -> std::result::Result<TrainingResult, String> {
    internal_train(request).await
}

// 向后兼容的预测函数
pub async fn predict_with_candle(request: PredictionRequest) -> std::result::Result<PredictionResponse, String> {
    internal_predict(request).await
}

// 向后兼容的评估函数 - 适配新接口
pub async fn evaluate_candle_model(model_id: String) -> std::result::Result<EvaluationResult, String> {
    // 直接通过ID加载模型元数据
    let model = match load_model_metadata(&model_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            eprintln!("加载模型元数据失败: {e}");
            return Err(format!("找不到模型ID: {model_id}"));
        }
    };
    
    match internal_evaluate(model.stock_code.clone(), Some(model.name.clone())).await {
        Ok(result) => {
            // 将新的EvaluationResult转换为旧的格式
            Ok(EvaluationResult {
                model_id: result.model_id,
                model_name: result.model_name,
                stock_code: result.stock_code,
                test_samples: result.test_samples,
                accuracy: result.accuracy,
                direction_accuracy: result.direction_accuracy,
                mse: result.mse,
                mae: result.mae,
                rmse: result.rmse,
                evaluation_date: result.evaluation_date,
            })
        },
        Err(e) => Err(e)
    }
}

// 重新训练模型函数 - 使用现有的训练流程
pub async fn retrain_candle_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> std::result::Result<(), String> {
    // 获取现有模型的元数据 - 修复：直接通过ID加载模型元数据
    let model = match load_model_metadata(&model_id) {
        Ok(metadata) => metadata,
        Err(e) => {
            eprintln!("加载模型元数据失败: {e}");
            return Err(format!("找不到模型ID: {model_id}"));
        }
    };
    
    println!("🔄 开始重新训练模型: {} ({})", model.name, model.stock_code);
    
    // 删除旧模型
    delete_model(&model_id).map_err(|e| format!("删除旧模型失败: {e}"))?;
    
    // 创建重新训练请求，使用现有模型的配置
    let retrain_request = TrainingRequest {
        stock_code: model.stock_code.clone(),
        model_name: format!("{}_retrained", model.name),
        start_date: "2023-01-01".to_string(), // 使用默认日期范围
        end_date: chrono::Local::now().naive_local().date().format("%Y-%m-%d").to_string(),
        features: model.features.clone(),
        target: model.target.clone(),
        prediction_days: model.prediction_days,
        model_type: model.model_type.clone(),
        epochs: epochs as usize,
        batch_size: batch_size as usize,
        learning_rate,
        dropout: 0.1, // 默认dropout
        train_test_split: 0.8, // 默认训练集比例
    };
    
    // 执行重新训练
    internal_train(retrain_request).await?;
    println!("✅ 模型重新训练完成");
    Ok(())
}
