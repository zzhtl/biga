use crate::stock_prediction::{TrainingRequest, PredictionRequest as CandlePredictionRequest, TrainingResult, ModelInfo, PredictionResponse, BacktestRequest, BacktestReport};
use crate::stock_prediction::feature_optimization::{analyze_feature_importance, FeatureOptimizationResult};
use crate::stock_prediction::hyperparameter_optimization::{HyperparameterOptimizer, HyperparameterConfig, OptimizationResult};
use crate::stock_prediction::multi_timeframe_analysis::MultiTimeframeSignal;

// 列出所有股票预测模型
#[tauri::command]
pub async fn list_stock_prediction_models(symbol: String) -> Result<Vec<ModelInfo>, String> {
    Ok(crate::stock_prediction::list_models(&symbol))
}

// 删除股票预测模型
#[tauri::command]
pub async fn delete_stock_prediction_model(model_id: String) -> Result<(), String> {
    crate::stock_prediction::delete_model(&model_id)
        .map_err(|e| format!("删除模型失败: {e}"))
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

// 预测功能（简化策略 - 专注方向准确性）
#[tauri::command]
pub async fn predict_candle_price_simple(request: CandlePredictionRequest) -> Result<PredictionResponse, String> {
    crate::stock_prediction::predict_with_simple_strategy(request).await
}

// 执行回测
#[tauri::command]
pub async fn run_model_backtest(request: BacktestRequest) -> Result<BacktestReport, String> {
    crate::stock_prediction::run_backtest(request).await
}

#[tauri::command]
pub async fn get_optimization_suggestions(
    stock_code: String,
    model_name: String,
    backtest_report: crate::stock_prediction::backtest::BacktestReport,
    current_features: Vec<String>,
    current_config: HyperparameterConfig,
) -> Result<OptimizationSuggestions, String> {
    // 1. 特征优化分析
    let feature_optimization = analyze_feature_importance(&backtest_report, &current_features)
        .map_err(|e| format!("特征分析失败: {e}"))?;
    
    // 2. 超参数优化建议
    let mut hyperparameter_optimizer = HyperparameterOptimizer::new(current_config);
    let hyperparameter_optimization = hyperparameter_optimizer.suggest_optimization(&backtest_report)
        .map_err(|e| format!("超参数优化失败: {e}"))?;
    
    // 3. 生成综合优化建议
    let suggestions = OptimizationSuggestions {
        stock_code,
        model_name,
        feature_optimization,
        hyperparameter_optimization,
        implementation_steps: generate_implementation_steps(&backtest_report),
        expected_overall_improvement: calculate_overall_improvement(&backtest_report),
    };
    
    Ok(suggestions)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationSuggestions {
    pub stock_code: String,
    pub model_name: String,
    pub feature_optimization: FeatureOptimizationResult,
    pub hyperparameter_optimization: OptimizationResult,
    pub implementation_steps: Vec<ImplementationStep>,
    pub expected_overall_improvement: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImplementationStep {
    pub step_number: i32,
    pub description: String,
    pub estimated_time: String,
    pub difficulty: String,
    pub expected_improvement: f64,
}

fn generate_implementation_steps(backtest_report: &crate::stock_prediction::backtest::BacktestReport) -> Vec<ImplementationStep> {
    let mut steps = Vec::new();
    
    // 基于回测结果生成具体的实施步骤
    if backtest_report.overall_price_accuracy < 0.6 {
        steps.push(ImplementationStep {
            step_number: 1,
            description: "增加技术指标特征：添加ATR、Williams%R、ROC等指标".to_string(),
            estimated_time: "2-3小时".to_string(),
            difficulty: "中等".to_string(),
            expected_improvement: 0.08,
        });
    }
    
    if backtest_report.overall_direction_accuracy < 0.7 {
        steps.push(ImplementationStep {
            step_number: 2,
            description: "优化趋势识别：实现多时间框架均线系统".to_string(),
            estimated_time: "3-4小时".to_string(),
            difficulty: "中等".to_string(),
            expected_improvement: 0.12,
        });
    }
    
    if backtest_report.average_prediction_error > 0.05 {
        steps.push(ImplementationStep {
            step_number: 3,
            description: "调整模型参数：优化学习率、批处理大小和网络结构".to_string(),
            estimated_time: "1-2小时".to_string(),
            difficulty: "简单".to_string(),
            expected_improvement: 0.06,
        });
    }
    
    steps.push(ImplementationStep {
        step_number: steps.len() as i32 + 1,
        description: "实施交叉验证：使用时间序列交叉验证验证改进效果".to_string(),
        estimated_time: "1小时".to_string(),
        difficulty: "简单".to_string(),
        expected_improvement: 0.03,
    });
    
    steps.push(ImplementationStep {
        step_number: steps.len() as i32 + 1,
        description: "部署A/B测试：对比新旧模型在实际预测中的表现".to_string(),
        estimated_time: "2小时".to_string(),
        difficulty: "中等".to_string(),
        expected_improvement: 0.05,
    });
    
    steps
}

fn calculate_overall_improvement(backtest_report: &crate::stock_prediction::backtest::BacktestReport) -> f64 {
    let mut total_improvement = 0.0;
    
    // 基于当前表现计算可能的改进空间
    if backtest_report.overall_price_accuracy < 0.6 {
        total_improvement += 0.15; // 特征工程改进
    } else if backtest_report.overall_price_accuracy < 0.75 {
        total_improvement += 0.08; // 参数调优改进
    }
    
    if backtest_report.overall_direction_accuracy < 0.7 {
        total_improvement += 0.12; // 趋势识别改进
    }
    
    if backtest_report.average_prediction_error > 0.05 {
        total_improvement += 0.06; // 模型优化改进
    }
    
    // 考虑递减效应
    total_improvement * 0.7 // 实际改进通常比理论值低30%
} 

#[tauri::command]
pub async fn get_multi_timeframe_signals(symbol: String) -> Result<Vec<MultiTimeframeSignal>, String> {
    use crate::stock_prediction::multi_timeframe_analysis::generate_multi_timeframe_signals;
    use crate::db::models::HistoricalData;
    use sqlx::Row;
    
    // 获取数据库连接
    let db_path = crate::stock_prediction::database::find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    let connection_string = format!("sqlite://{}", db_path.display());
    let db = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("数据库连接失败: {e}"))?;
    
    // 查询历史数据
    let rows = sqlx::query("SELECT symbol, date, open, high, low, close, volume, amount, amplitude, turnover_rate, change_percent, change 
         FROM historical_data 
         WHERE symbol = ? 
         ORDER BY date ASC")
    .bind(&symbol)
    .fetch_all(&db)
    .await
    .map_err(|e| format!("查询历史数据失败: {e}"))?;
    
    if rows.is_empty() {
        return Err("未找到该股票的历史数据".to_string());
    }
    
    // 转换为HistoricalData结构
    let mut historical_data = Vec::new();
    for row in rows {
        let date_str: String = row.get("date");
        let date = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
            .map_err(|e| format!("日期解析失败: {e}"))?;
        
        historical_data.push(HistoricalData {
            symbol: row.get("symbol"),
            date,
            open: row.get("open"),
            high: row.get("high"),
            low: row.get("low"),
            close: row.get("close"),
            volume: row.get("volume"),
            amount: row.get("amount"),
            amplitude: row.get("amplitude"),
            turnover_rate: row.get("turnover_rate"),
            change_percent: row.get("change_percent"),
            change: row.get("change"),
        });
    }
    
    // 生成多时间周期信号
    let signals = generate_multi_timeframe_signals(&symbol, &historical_data);
    
    Ok(signals)
}

#[tauri::command]
pub async fn get_latest_multi_timeframe_signal(symbol: String) -> Result<Option<MultiTimeframeSignal>, String> {
    use crate::stock_prediction::multi_timeframe_analysis::get_latest_multi_timeframe_signal;
    use crate::db::models::HistoricalData;
    use sqlx::Row;
    
    // 获取数据库连接
    let db_path = crate::stock_prediction::database::find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    let connection_string = format!("sqlite://{}", db_path.display());
    let db = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("数据库连接失败: {e}"))?;
    
    // 查询最近60天的历史数据（确保有足够数据计算技术指标）
    let rows = sqlx::query("SELECT symbol, date, open, high, low, close, volume, amount, amplitude, turnover_rate, change_percent, change 
         FROM historical_data 
         WHERE symbol = ? 
         ORDER BY date DESC 
         LIMIT 60")
    .bind(&symbol)
    .fetch_all(&db)
    .await
    .map_err(|e| format!("查询历史数据失败: {e}"))?;
    
    if rows.is_empty() {
        return Err("未找到该股票的历史数据".to_string());
    }
    
    // 转换为HistoricalData结构并反转顺序（从旧到新）
    let mut historical_data = Vec::new();
    for row in rows.iter().rev() {
        let date_str: String = row.get("date");
        let date = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
            .map_err(|e| format!("日期解析失败: {e}"))?;
        
        historical_data.push(HistoricalData {
            symbol: row.get("symbol"),
            date,
            open: row.get("open"),
            high: row.get("high"),
            low: row.get("low"),
            close: row.get("close"),
            volume: row.get("volume"),
            amount: row.get("amount"),
            amplitude: row.get("amplitude"),
            turnover_rate: row.get("turnover_rate"),
            change_percent: row.get("change_percent"),
            change: row.get("change"),
        });
    }
    
    // 获取最新的多时间周期信号
    let signal = get_latest_multi_timeframe_signal(&symbol, &historical_data);
    
    Ok(signal)
}

#[tauri::command]
pub async fn analyze_multi_timeframe_prediction_value(symbol: String) -> Result<std::collections::HashMap<String, f64>, String> {
    use crate::stock_prediction::multi_timeframe_analysis::{generate_multi_timeframe_signals, analyze_signal_prediction_value};
    use crate::db::models::HistoricalData;
    use sqlx::Row;
    
    // 获取数据库连接
    let db_path = crate::stock_prediction::database::find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    let connection_string = format!("sqlite://{}", db_path.display());
    let db = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("数据库连接失败: {e}"))?;
    
    // 查询历史数据
    let rows = sqlx::query("SELECT symbol, date, open, high, low, close, volume, amount, amplitude, turnover_rate, change_percent, change 
         FROM historical_data 
         WHERE symbol = ? 
         ORDER BY date ASC")
    .bind(&symbol)
    .fetch_all(&db)
    .await
    .map_err(|e| format!("查询历史数据失败: {e}"))?;
    
    if rows.is_empty() {
        return Err("未找到该股票的历史数据".to_string());
    }
    
    // 转换为HistoricalData结构
    let mut historical_data = Vec::new();
    for row in rows {
        let date_str: String = row.get("date");
        let date = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
            .map_err(|e| format!("日期解析失败: {e}"))?;
        
        historical_data.push(HistoricalData {
            symbol: row.get("symbol"),
            date,
            open: row.get("open"),
            high: row.get("high"),
            low: row.get("low"),
            close: row.get("close"),
            volume: row.get("volume"),
            amount: row.get("amount"),
            amplitude: row.get("amplitude"),
            turnover_rate: row.get("turnover_rate"),
            change_percent: row.get("change_percent"),
            change: row.get("change"),
        });
    }
    
    // 生成多时间周期信号
    let signals = generate_multi_timeframe_signals(&symbol, &historical_data);
    
    // 分析预测价值
    let analysis = analyze_signal_prediction_value(&signals);
    
    Ok(analysis)
} 