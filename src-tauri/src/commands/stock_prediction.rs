use crate::stock_prediction::{TrainingRequest, PredictionRequest as CandlePredictionRequest, TrainingResult, ModelInfo, Prediction, PredictionResponse};

// 列出所有股票预测模型
#[tauri::command]
pub async fn list_stock_prediction_models(symbol: String) -> Result<Vec<ModelInfo>, String> {
    Ok(crate::stock_prediction::list_models(&symbol))
}

// 删除股票预测模型
#[tauri::command]
pub async fn delete_stock_prediction_model(model_id: String) -> Result<(), String> {
    crate::stock_prediction::delete_model(&model_id)
        .map_err(|e| format!("删除模型失败: {}", e))
}

// 使用Candle训练股票预测模型
#[tauri::command]
pub async fn train_candle_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    crate::stock_prediction::train_candle_model(request).await
}

// 使用Candle进行股票价格预测
#[tauri::command]
pub async fn predict_with_candle(request: CandlePredictionRequest) -> Result<PredictionResponse, String> {
    crate::stock_prediction::predict_with_candle(request).await
}

// 重新训练Candle模型
#[tauri::command]
pub async fn retrain_candle_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    crate::stock_prediction::retrain_candle_model(model_id, epochs, batch_size, learning_rate).await
}

// 评估Candle模型
#[tauri::command]
pub async fn evaluate_candle_model(model_id: String) -> Result<crate::stock_prediction::EvaluationResult, String> {
    crate::stock_prediction::evaluate_candle_model(model_id).await
}

// 训练股票预测模型 - 向后兼容的简化版本
#[tauri::command]
pub async fn train_stock_prediction_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    train_candle_model(request).await
}

// 进行股票价格预测 - 向后兼容的简化版本
#[tauri::command]
pub async fn predict_stock_price(request: CandlePredictionRequest) -> Result<PredictionResponse, String> {
    predict_with_candle(request).await
} 