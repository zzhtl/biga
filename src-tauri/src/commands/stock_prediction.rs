use crate::db::{prediction};
use crate::prediction::model;
use crate::stock_prediction::{TrainingRequest, PredictionRequest as CandlePredictionRequest, TrainingResult, ModelInfo, Prediction};
use chrono::{NaiveDate, Utc, Datelike};
use serde::{Deserialize, Serialize};
use tauri::{State};
use sqlx::{Pool, Sqlite};

// 简化版StockData结构体
#[derive(Debug, Clone)]
struct StockData {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

// 模型训练配置
#[derive(Debug, Deserialize)]
pub struct TrainModelRequest {
    pub stock_code: String,
    pub model_name: String,
    pub start_date: String,
    pub end_date: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: u32,
    pub model_type: String,
}

// 预测请求 - 重命名以避免与candle_prediction中的冲突
#[derive(Debug, Deserialize)]
pub struct LocalPredictionRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub prediction_days: u32,
}

// 模型元数据
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub stock_code: String,
    pub created_at: i64,
    pub model_type: String,
    pub features: Vec<String>,
    pub target: String,
    pub prediction_days: u32,
    pub accuracy: f64,
}

// 预测结果
#[derive(Debug, Serialize)]
pub struct PredictionResult {
    pub target_date: String,
    pub predicted_price: f64,
    pub predicted_change_percent: f64,
    pub confidence: f64,
}

// 列出所有股票预测模型
#[tauri::command]
pub async fn list_stock_prediction_models(symbol: String) -> Result<Vec<ModelInfo>, String> {
    // 使用Candle预测模块来获取模型列表
    Ok(crate::stock_prediction::list_models(&symbol))
}

// 删除股票预测模型
#[tauri::command]
pub async fn delete_stock_prediction_model(model_id: String) -> Result<(), String> {
    // 使用Candle预测模块删除模型
    crate::stock_prediction::delete_model(&model_id)
        .map_err(|e| format!("删除模型失败: {}", e))
}

// 使用Candle训练股票预测模型
#[tauri::command]
pub async fn train_candle_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    // 直接使用Candle模型进行训练
    crate::stock_prediction::train_candle_model(request).await
}

// 使用Candle进行股票价格预测
#[tauri::command]
pub async fn predict_with_candle(request: CandlePredictionRequest) -> Result<Vec<Prediction>, String> {
    // 使用Candle模型进行预测
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

// 训练股票预测模型 - 旧的API保留向后兼容
#[tauri::command]
pub async fn train_stock_prediction_model(
    state: State<'_, Pool<Sqlite>>, 
    request: TrainModelRequest
) -> Result<ModelMetadata, String> {
    // 从数据库获取历史数据
    let historical_data = sqlx::query_as::<_, crate::db::models::HistoricalData>(
        r#"SELECT * FROM historical_data
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#,
    )
    .bind(&request.stock_code)
    .bind(&request.start_date)
    .bind(&request.end_date)
    .fetch_all(&*state)
    .await
    .map_err(|e| format!("获取历史数据失败: {}", e))?;
    
    // 检查是否有足够的历史数据
    if historical_data.len() < 20 {
        return Err("历史数据不足，无法训练模型。需要至少20天的数据。".to_string());
    }
    
    // 构建模型配置
    let model_config = crate::db::models::PredictionModelConfig {
        model_name: request.model_name.clone(),
        model_type: request.model_type.clone(),
        parameters: serde_json::json!({
            "learning_rate": 0.01,
            "max_depth": 5,
            "c": 1.0
        }),
        features: request.features.clone(),
        lookback_days: 14, // 使用过去14天的数据
        train_test_split: 0.8, // 80%的数据用于训练
    };
    
    // 训练模型
    let result = model::train_model(
        &*state,
        &request.stock_code,
        &historical_data,
        &model_config
    )
    .await
    .map_err(|e| format!("模型训练失败: {}", e))?;
    
    // 构建响应
    let model_info = result.0;
    let model_metadata = ModelMetadata {
        id: model_info.id.to_string(),
        name: model_info.model_name,
        stock_code: model_info.symbol,
        created_at: model_info.created_at.timestamp(),
        model_type: model_info.model_type,
        features: request.features,
        target: request.target,
        prediction_days: request.prediction_days,
        accuracy: 0.8, // 使用一个默认值
    };
    
    Ok(model_metadata)
}

// 进行股票价格预测 - 旧的API保留向后兼容
#[tauri::command]
pub async fn predict_stock_price(
    state: State<'_, Pool<Sqlite>>, 
    request: LocalPredictionRequest
) -> Result<Vec<PredictionResult>, String> {
    // 获取最近的历史数据
    let end_date = Utc::now().naive_utc().date();
    let start_date = end_date - chrono::Duration::days(60); // 获取更多历史数据以便进行特征计算
    
    // 从数据库获取真实历史数据
    let historical_data = sqlx::query_as::<_, crate::db::models::HistoricalData>(
        r#"SELECT * FROM historical_data
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#,
    )
    .bind(&request.stock_code)
    .bind(start_date.format("%Y-%m-%d").to_string())
    .bind(end_date.format("%Y-%m-%d").to_string())
    .fetch_all(&*state)
    .await
    .map_err(|e| format!("获取历史数据失败: {}", e))?;
    
    if historical_data.is_empty() {
        return Err(format!("未找到股票 {} 的历史数据", request.stock_code));
    }
    
    // 如果提供了模型名称，使用特定模型进行预测
    if let Some(model_name) = &request.model_name {
        return predict_with_model(&state, &request.stock_code, model_name, &historical_data, request.prediction_days).await;
    }
    
    // 否则使用简单的时间序列预测
    let mut results = Vec::new();
    
    // 获取最近的收盘价作为基准
    let latest_price = historical_data.last()
        .map(|data| data.close)
        .unwrap_or(0.0) as f64;
    
    // 计算过去几天的平均变化率
    let avg_change_percent = if historical_data.len() >= 5 {
        let recent_data = &historical_data[historical_data.len() - 5..];
        let avg = recent_data.iter()
            .map(|d| d.change_percent)
            .sum::<f64>() / recent_data.len() as f64;
        avg
    } else {
        0.0 // 数据不足时默认为0
    };
    
    // 生成预测结果
    let mut current_date = end_date;
    let mut current_price = latest_price;
    
    for day in 1..=request.prediction_days {
        // 跳过周末
        current_date = next_trading_day(current_date);
        
        // 基于平均变化率预测价格
        let change_percent = avg_change_percent;
        let price_change = current_price * (change_percent / 100.0);
        current_price += price_change;
        
        results.push(PredictionResult {
            target_date: current_date.format("%Y-%m-%d").to_string(),
            predicted_price: current_price,
            predicted_change_percent: change_percent,
            confidence: 0.7, // 简单模型置信度较低
        });
    }
    
    Ok(results)
}

// 使用特定模型预测股票价格
async fn predict_with_model(
    pool: &Pool<Sqlite>,
    symbol: &str,
    model_name: &str,
    historical_data: &[crate::db::models::HistoricalData],
    prediction_days: u32
) -> Result<Vec<PredictionResult>, String> {
    // 获取模型
    let model = prediction::get_model_by_symbol_and_name(pool, symbol, model_name)
        .await
        .map_err(|e| format!("获取模型失败: {}", e))?;
    
    // 使用预测模块进行预测
    let prediction_results = model::predict_stock(
        pool,
        symbol,
        historical_data,
        Some(model_name.to_string()),
        prediction_days as i32
    )
    .await
    .map_err(|e| format!("预测失败: {}", e))?;
    
    // 转换为API响应格式
    let results = prediction_results.into_iter()
        .map(|result| PredictionResult {
            target_date: result.target_date.format("%Y-%m-%d").to_string(),
            predicted_price: result.predicted_price,
            predicted_change_percent: result.predicted_change_percent,
            confidence: result.confidence,
        })
        .collect();
    
    Ok(results)
}

// 获取下一个交易日（简单地跳过周末）
fn next_trading_day(date: NaiveDate) -> NaiveDate {
    let mut next_date = date + chrono::Duration::days(1);
    
    // 跳过周末 (6=周六, 7=周日)
    while next_date.weekday().number_from_monday() > 5 {
        next_date = next_date + chrono::Duration::days(1);
    }
    
    next_date
} 