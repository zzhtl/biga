//! 预测服务

use crate::prediction::{
    types::*,
    model::{training, inference, management},
    strategy::multi_timeframe,
};
use crate::db::{connection::create_temp_pool, repository::get_recent_historical_data};

/// 训练模型
pub async fn train_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    training::train_model(request).await
}

/// 重训练模型
pub async fn retrain_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    training::retrain_model(model_id, epochs, batch_size, learning_rate).await
}

/// 进行预测
pub async fn predict(request: PredictionRequest) -> Result<PredictionResponse, String> {
    inference::predict(request).await
}

/// 简化策略预测
pub async fn predict_simple(request: PredictionRequest) -> Result<PredictionResponse, String> {
    inference::predict_simple(request).await
}

/// 评估模型
pub async fn evaluate_model(model_id: String) -> Result<EvaluationResult, String> {
    inference::evaluate_model(model_id).await
}

/// 列出模型
pub fn list_models(stock_code: &str) -> Vec<ModelInfo> {
    management::list_models(stock_code)
}

/// 删除模型
pub fn delete_model(model_id: &str) -> Result<(), String> {
    management::delete_model(model_id)
}

/// 获取多周期信号
pub async fn get_multi_timeframe_signals(symbol: String) -> Result<Vec<multi_timeframe::MultiTimeframeSignal>, String> {
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&symbol, 200, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    
    if historical.is_empty() {
        return Err("未找到历史数据".to_string());
    }
    
    let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let dates: Vec<String> = historical.iter().map(|h| h.date.format("%Y-%m-%d").to_string()).collect();
    
    let signals = multi_timeframe::generate_multi_timeframe_signals(&prices, &highs, &lows, &dates);
    
    Ok(signals)
}

/// 获取最新多周期信号
pub async fn get_latest_multi_timeframe_signal(symbol: String) -> Result<Option<multi_timeframe::MultiTimeframeSignal>, String> {
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&symbol, 60, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    
    if historical.is_empty() {
        return Err("未找到历史数据".to_string());
    }
    
    let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let date = historical.last().unwrap().date.format("%Y-%m-%d").to_string();
    
    let signal = multi_timeframe::get_latest_signal(&prices, &highs, &lows, &date);
    
    Ok(signal)
}

