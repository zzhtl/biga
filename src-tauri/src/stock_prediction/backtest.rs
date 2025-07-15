use crate::stock_prediction::types::{PredictionRequest, Prediction};
use crate::stock_prediction::database::get_historical_data_from_db;
use crate::stock_prediction::model_management::list_models;
use crate::stock_prediction::prediction::predict_with_candle;
use crate::stock_prediction::utils::{get_next_trading_day, calculate_direction_focused_accuracy};
use serde::{Deserialize, Serialize};
use chrono::{NaiveDate, Duration};
use std::collections::HashMap;

// 回测请求结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestRequest {
    pub stock_code: String,
    pub model_name: Option<String>,
    pub start_date: String,      // 回测开始日期
    pub end_date: String,        // 回测结束日期
    pub prediction_days: usize,  // 预测天数
    pub backtest_interval: usize, // 回测间隔天数（例如每7天进行一次预测）
}

// 单次回测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestEntry {
    pub prediction_date: String,        // 预测发起日期
    pub predictions: Vec<Prediction>,   // 预测结果
    pub actual_prices: Vec<f64>,       // 实际价格
    pub actual_changes: Vec<f64>,      // 实际涨跌幅
    pub price_accuracy: f64,           // 价格预测准确率
    pub direction_accuracy: f64,       // 方向预测准确率
    pub avg_prediction_error: f64,     // 平均预测误差
}

// 完整回测报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub stock_code: String,
    pub model_name: String,
    pub backtest_period: String,       // 回测期间
    pub total_predictions: usize,      // 总预测次数
    pub backtest_entries: Vec<BacktestEntry>,
    
    // 总体统计
    pub overall_price_accuracy: f64,      // 总体价格准确率
    pub overall_direction_accuracy: f64,   // 总体方向准确率
    pub average_prediction_error: f64,     // 平均预测误差
    pub accuracy_trend: Vec<f64>,          // 准确率趋势
    
    // 分时段统计
    pub daily_accuracy: Vec<DailyAccuracy>,
    
    // 详细统计
    pub price_error_distribution: Vec<f64>, // 价格误差分布
    pub direction_correct_rate: f64,        // 方向正确率
    pub volatility_vs_accuracy: Vec<(f64, f64)>, // 波动率与准确率的关系
}

// 日度准确率统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyAccuracy {
    pub date: String,
    pub price_accuracy: f64,
    pub direction_accuracy: f64,
    pub prediction_count: usize,
    pub market_volatility: f64,
}

// 执行回测
pub async fn run_backtest(request: BacktestRequest) -> Result<BacktestReport, String> {
    // 1. 验证参数
    let start_date = NaiveDate::parse_from_str(&request.start_date, "%Y-%m-%d")
        .map_err(|e| format!("开始日期解析失败: {}", e))?;
    let end_date = NaiveDate::parse_from_str(&request.end_date, "%Y-%m-%d")
        .map_err(|e| format!("结束日期解析失败: {}", e))?;
    
    if start_date >= end_date {
        return Err("开始日期必须早于结束日期".to_string());
    }
    
    // 2. 获取模型信息
    let model_list = list_models(&request.stock_code);
    if model_list.is_empty() {
        return Err("没有找到可用的模型".to_string());
    }
    
    let model_info = if let Some(model_name) = &request.model_name {
        model_list.iter()
            .find(|m| m.name == *model_name)
            .ok_or_else(|| format!("找不到名为 {} 的模型", model_name))?
            .clone()
    } else {
        model_list[0].clone()
    };
    
    println!("🔍 开始回测: {} 模型 '{}' 在 {} 到 {} 期间", 
             request.stock_code, model_info.name, request.start_date, request.end_date);
    
    // 3. 获取完整历史数据（包括回测期间+预测所需的额外数据）
    let data_start_date = start_date - Duration::days(60); // 额外获取60天数据用于预测
    let data_end_date = end_date + Duration::days(request.prediction_days as i64 + 10); // 额外获取预测天数+缓冲
    
    let historical_data = get_historical_data_from_db(
        &request.stock_code,
        &data_start_date.format("%Y-%m-%d").to_string(),
        &data_end_date.format("%Y-%m-%d").to_string()
    ).await.map_err(|e| format!("获取历史数据失败: {}", e))?;
    
    if historical_data.is_empty() {
        return Err("历史数据为空".to_string());
    }
    
    // 4. 按日期排序并建立索引
    let mut sorted_data = historical_data;
    sorted_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    let mut date_to_data: HashMap<String, &_> = HashMap::new();
    for data in &sorted_data {
        date_to_data.insert(data.date.clone(), data);
    }
    
    // 5. 执行回测
    let mut backtest_entries = Vec::new();
    let mut current_date = start_date;
    let interval_days = request.backtest_interval.max(1);
    
    while current_date <= end_date - Duration::days(request.prediction_days as i64) {
        // 确保是交易日
        let mut trading_date = current_date;
        while !date_to_data.contains_key(&trading_date.format("%Y-%m-%d").to_string()) {
            trading_date = get_next_trading_day(trading_date);
            if trading_date > end_date {
                break;
            }
        }
        
        if trading_date > end_date {
            break;
        }
        
        let prediction_date_str = trading_date.format("%Y-%m-%d").to_string();
        
        // 执行预测
        let prediction_request = PredictionRequest {
            stock_code: request.stock_code.clone(),
            model_name: Some(model_info.name.clone()),
            prediction_days: request.prediction_days,
            use_candle: true,
        };
        
        // 模拟在该日期进行预测（这里需要修改预测函数以支持历史日期预测）
        match simulate_historical_prediction(&prediction_request, &trading_date, &sorted_data).await {
            Ok(prediction_result) => {
                // 获取实际数据进行对比
                let actual_data = get_actual_data_for_prediction(
                    &trading_date, 
                    request.prediction_days, 
                    &date_to_data
                );
                
                if !actual_data.is_empty() {
                    let entry = create_backtest_entry(
                        prediction_date_str,
                        prediction_result.predictions,
                        actual_data,
                    );
                    backtest_entries.push(entry);
                    
                    println!("📊 完成 {} 的回测 - 价格准确率: {:.2}%, 方向准确率: {:.2}%",
                             trading_date.format("%Y-%m-%d"),
                             backtest_entries.last().unwrap().price_accuracy * 100.0,
                             backtest_entries.last().unwrap().direction_accuracy * 100.0);
                }
            }
            Err(e) => {
                println!("⚠️ {} 预测失败: {}", trading_date.format("%Y-%m-%d"), e);
            }
        }
        
        current_date = current_date + Duration::days(interval_days as i64);
    }
    
    if backtest_entries.is_empty() {
        return Err("回测期间没有成功的预测".to_string());
    }
    
    // 6. 计算总体统计
    let report = generate_backtest_report(
        request.stock_code,
        model_info.name,
        format!("{} 到 {}", request.start_date, request.end_date),
        backtest_entries,
    );
    
    println!("✅ 回测完成! 总体准确率 - 价格: {:.2}%, 方向: {:.2}%",
             report.overall_price_accuracy * 100.0,
             report.overall_direction_accuracy * 100.0);
    
    Ok(report)
}

// 模拟历史预测（简化版本，实际应该基于历史数据重新预测）
async fn simulate_historical_prediction(
    request: &PredictionRequest,
    _historical_date: &NaiveDate,
    _historical_data: &[crate::stock_prediction::types::HistoricalDataType],
) -> Result<crate::stock_prediction::types::PredictionResponse, String> {
    // 这里应该基于历史数据重新进行预测
    // 为了简化，现在使用现有的预测函数
    // 实际应用中需要修改预测函数以支持基于历史数据的预测
    predict_with_candle(request.clone()).await
}

// 获取预测期间的实际数据
fn get_actual_data_for_prediction(
    start_date: &NaiveDate,
    prediction_days: usize,
    date_to_data: &HashMap<String, &crate::stock_prediction::types::HistoricalDataType>,
) -> Vec<(f64, f64)> {
    let mut actual_data = Vec::new();
    let mut current_date = *start_date;
    
    for _ in 0..prediction_days {
        current_date = get_next_trading_day(current_date);
        let date_str = current_date.format("%Y-%m-%d").to_string();
        
        if let Some(data) = date_to_data.get(&date_str) {
            actual_data.push((data.close, data.change_percent));
        }
    }
    
    actual_data
}

// 创建回测条目
fn create_backtest_entry(
    prediction_date: String,
    predictions: Vec<Prediction>,
    actual_data: Vec<(f64, f64)>,
) -> BacktestEntry {
    let mut predicted_prices = Vec::new();
    let mut predicted_changes = Vec::new();
    let mut actual_prices = Vec::new();
    let mut actual_changes = Vec::new();
    
    let min_len = predictions.len().min(actual_data.len());
    
    for i in 0..min_len {
        predicted_prices.push(predictions[i].predicted_price);
        predicted_changes.push(predictions[i].predicted_change_percent);
        actual_prices.push(actual_data[i].0);
        actual_changes.push(actual_data[i].1);
    }
    
    // 计算准确率
    let (direction_accuracy, price_accuracy) = calculate_direction_focused_accuracy(
        &predicted_changes, 
        &actual_changes
    );
    
    // 计算平均误差
    let avg_prediction_error = if !predicted_prices.is_empty() {
        predicted_prices.iter()
            .zip(actual_prices.iter())
            .map(|(pred, actual)| ((pred - actual) / actual).abs())
            .sum::<f64>() / predicted_prices.len() as f64
    } else {
        0.0
    };
    
    BacktestEntry {
        prediction_date,
        predictions,
        actual_prices,
        actual_changes,
        price_accuracy,
        direction_accuracy,
        avg_prediction_error,
    }
}

// 生成回测报告
fn generate_backtest_report(
    stock_code: String,
    model_name: String,
    backtest_period: String,
    backtest_entries: Vec<BacktestEntry>,
) -> BacktestReport {
    let total_predictions = backtest_entries.len();
    
    // 计算总体准确率
    let overall_price_accuracy = backtest_entries.iter()
        .map(|entry| entry.price_accuracy)
        .sum::<f64>() / total_predictions as f64;
    
    let overall_direction_accuracy = backtest_entries.iter()
        .map(|entry| entry.direction_accuracy)
        .sum::<f64>() / total_predictions as f64;
    
    let average_prediction_error = backtest_entries.iter()
        .map(|entry| entry.avg_prediction_error)
        .sum::<f64>() / total_predictions as f64;
    
    // 计算准确率趋势
    let accuracy_trend = backtest_entries.iter()
        .map(|entry| entry.direction_accuracy)
        .collect();
    
    // 生成日度准确率统计
    let daily_accuracy: Vec<DailyAccuracy> = backtest_entries.iter()
        .map(|entry| {
            let volatility = entry.actual_changes.iter()
                .map(|change| change.abs())
                .sum::<f64>() / entry.actual_changes.len().max(1) as f64;
            
            DailyAccuracy {
                date: entry.prediction_date.clone(),
                price_accuracy: entry.price_accuracy,
                direction_accuracy: entry.direction_accuracy,
                prediction_count: entry.predictions.len(),
                market_volatility: volatility,
            }
        })
        .collect();
    
    // 计算误差分布
    let price_error_distribution = backtest_entries.iter()
        .map(|entry| entry.avg_prediction_error)
        .collect();
    
    // 计算方向正确率
    let direction_correct_rate = backtest_entries.iter()
        .map(|entry| {
            let correct_count = entry.predictions.iter()
                .zip(entry.actual_changes.iter())
                .filter(|(pred, actual)| {
                    (pred.predicted_change_percent > 0.0 && **actual > 0.0) ||
                    (pred.predicted_change_percent < 0.0 && **actual < 0.0)
                })
                .count();
            correct_count as f64 / entry.predictions.len().max(1) as f64
        })
        .sum::<f64>() / total_predictions as f64;
    
    // 波动率与准确率的关系
    let volatility_vs_accuracy = daily_accuracy.iter()
        .map(|daily| (daily.market_volatility, daily.direction_accuracy))
        .collect();
    
    BacktestReport {
        stock_code,
        model_name,
        backtest_period,
        total_predictions,
        backtest_entries,
        overall_price_accuracy,
        overall_direction_accuracy,
        average_prediction_error,
        accuracy_trend,
        daily_accuracy,
        price_error_distribution,
        direction_correct_rate,
        volatility_vs_accuracy,
    }
} 