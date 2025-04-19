use crate::db::{prediction};
use crate::prediction::model;
use chrono::{NaiveDate, Utc, Datelike};
use serde::{Deserialize, Serialize};
use std::fs;
use tauri::{AppHandle, Manager};
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

// 预测请求
#[derive(Debug, Deserialize)]
pub struct PredictionRequest {
    pub stock_code: String,
    pub model_id: String,
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
    pub date: String,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
}

// 从AppHandle获取数据库连接池的函数
fn get_pool(app_handle: &AppHandle) -> tauri::State<'_, Pool<Sqlite>> {
    app_handle.state::<Pool<Sqlite>>()
}

// 列出所有股票预测模型
#[tauri::command]
pub async fn list_stock_prediction_models(app_handle: AppHandle) -> Result<Vec<ModelMetadata>, String> {
    let pool = get_pool(&app_handle);
    
    // 获取模型列表
    let models = prediction::list_models_for_symbol(&*pool, "")
        .await
        .map_err(|e| format!("获取模型列表失败: {}", e))?;
    
    // 将数据库模型转换为API响应格式
    let result = models.into_iter()
        .map(|model| {
            let features: Vec<String> = serde_json::from_str(&model.parameters)
                .unwrap_or_default();
            
            ModelMetadata {
                id: model.id.to_string(),
                name: model.model_name,
                stock_code: model.symbol,
                created_at: model.created_at.timestamp(),
                model_type: model.model_type,
                features,
                target: "close".to_string(), // 默认使用收盘价
                prediction_days: 7, // 默认预测7天
                accuracy: 0.8, // 默认准确率
            }
        })
        .collect();
    
    Ok(result)
}

// 删除股票预测模型
#[tauri::command]
pub async fn delete_stock_prediction_model(
    app_handle: AppHandle, 
    model_id: String
) -> Result<(), String> {
    let pool = get_pool(&app_handle);
    
    // 删除模型
    prediction::delete_model(
        &*pool, 
        model_id.parse::<i64>().map_err(|e| format!("无效的模型ID: {}", e))?
    )
    .await
    .map_err(|e| format!("删除模型失败: {}", e))?;
    
    // 删除模型文件（如果有）
    // 在tauri 2.0中，路径处理可能需要使用tauri::PathResolver
    // 这里使用一个相对路径作为后备
    let app_data_dir = std::env::current_dir()
        .map_err(|e| format!("获取当前目录失败: {}", e))?
        .join("data");
    
    let model_path = app_data_dir.join("models").join(format!("{}.bin", model_id));
    if model_path.exists() {
        fs::remove_file(model_path)
            .map_err(|e| format!("删除模型文件失败: {}", e))?;
    }
    
    Ok(())
}

// 训练股票预测模型
#[tauri::command]
pub async fn train_stock_prediction_model(
    app_handle: AppHandle, 
    request: TrainModelRequest
) -> Result<ModelMetadata, String> {
    let pool = get_pool(&app_handle);

    // 从数据库获取历史数据
    let historical_data = sqlx::query_as::<_, crate::db::models::HistoricalData>(
        r#"SELECT * FROM historical_data
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#,
    )
    .bind(&request.stock_code)
    .bind(&request.start_date)
    .bind(&request.end_date)
    .fetch_all(&*pool)
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
        &*pool,
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

// 进行股票价格预测
#[tauri::command]
pub async fn predict_stock_price(
    app_handle: AppHandle, 
    request: PredictionRequest
) -> Result<Vec<PredictionResult>, String> {
    let pool = get_pool(&app_handle);
    
    // 获取最近的历史数据
    let end_date = Utc::now().naive_utc().date();
    let start_date = end_date - chrono::Duration::days(30);
    
    // 从数据库获取历史数据
    let historical_data = sqlx::query_as::<_, crate::db::models::HistoricalData>(
        r#"SELECT * FROM historical_data
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#,
    )
    .bind(&request.stock_code)
    .bind(start_date.to_string())
    .bind(end_date.to_string())
    .fetch_all(&*pool)
    .await
    .map_err(|e| format!("获取历史数据失败: {}", e))?;
    
    if historical_data.is_empty() {
        return Err("没有足够的历史数据用于预测".to_string());
    }
    
    // 获取模型ID
    let _model_id = request.model_id.parse::<i64>()
        .map_err(|e| format!("无效的模型ID: {}", e))?;
    
    // 进行预测
    let predictions = model::predict_stock(
        &*pool,
        &request.stock_code,
        &historical_data,
        None, // 使用默认模型
        request.prediction_days as i32
    )
    .await
    .map_err(|e| format!("预测失败: {}", e))?;
    
    // 转换预测结果为API格式
    let results = predictions.into_iter()
        .map(|p| {
            let confidence = p.confidence * p.predicted_price * 0.1; // 使用10%的置信区间
            PredictionResult {
                date: p.target_date.format("%Y-%m-%d").to_string(),
                predicted_value: p.predicted_price,
                confidence_interval: (p.predicted_price - confidence, p.predicted_price + confidence),
            }
        })
        .collect();
    
    Ok(results)
}

// 创建模拟历史数据（实际项目中应该从数据库获取）
fn get_mock_historical_data(symbol: &str, start_date: NaiveDate, end_date: NaiveDate) -> Vec<crate::db::models::HistoricalData> {
    let mut data = Vec::new();
    let mut current_date = start_date;
    let mut price = 100.0;
    
    while current_date <= end_date {
        // 跳过周末
        let weekday = current_date.weekday().num_days_from_monday();
        if weekday < 5 { // 0-4 是周一到周五 (Datelike trait中的方法返回0-6)
            // 模拟每天有少量变化的价格
            let change = (rand::random::<f64>() - 0.5) * 2.0; // -1.0 到 1.0 的随机变化
            price += change;
            
            let high = price + rand::random::<f64>() * 1.0;
            let low = price - rand::random::<f64>() * 1.0;
            let open = low + rand::random::<f64>() * (high - low);
            
            data.push(crate::db::models::HistoricalData {
                symbol: symbol.to_string(),
                date: current_date,
                open,
                close: price,
                high,
                low,
                volume: (1000000.0 * (1.0 + rand::random::<f64>() * 0.2)).round() as i64,
                amount: price * 1000000.0 * (1.0 + rand::random::<f64>() * 0.2),
                amplitude: ((high - low) / price * 100.0).round() / 100.0,
                turnover_rate: (rand::random::<f64>() * 5.0).round() / 100.0,
                change_percent: (change / (price - change) * 100.0).round() / 100.0,
                change,
            });
        }
        
        current_date = current_date.succ_opt().unwrap_or(current_date);
    }
    
    data
} 