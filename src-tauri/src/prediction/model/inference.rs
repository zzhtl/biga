//! 模型推理模块

use crate::prediction::types::{
    PredictionRequest, PredictionResponse, Prediction, LastRealData,
    EvaluationResult, TechnicalIndicatorValues,
};
use crate::prediction::model::management::load_model_metadata;
use crate::prediction::indicators;
use crate::prediction::analysis::{trend, volume, pattern, support_resistance};
use crate::prediction::strategy::multi_factor;
use crate::utils::date::get_next_trading_day;
use crate::db::{connection::create_temp_pool, repository::get_recent_historical_data};

/// 使用模型进行预测
pub async fn predict(request: PredictionRequest) -> Result<PredictionResponse, String> {
    // 获取历史数据
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, 200, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    
    if historical.is_empty() {
        return Err("未找到历史数据".to_string());
    }
    
    // 提取数据
    let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let volumes: Vec<i64> = historical.iter().map(|h| h.volume).collect();
    let opens: Vec<f64> = historical.iter().map(|h| h.open).collect();
    
    let current_price = *prices.last().unwrap();
    let last_data = historical.last().unwrap();
    
    // 技术分析
    let trend_analysis = trend::analyze_trend(&prices, &highs, &lows);
    let volume_signal = volume::analyze_volume_price(&prices, &highs, &lows, &volumes);
    let patterns = pattern::recognize_patterns(&opens, &prices, &highs, &lows);
    let sr = support_resistance::calculate_support_resistance(&prices, &highs, &lows, current_price);
    let tech_indicators = indicators::calculate_all_indicators(&prices, &highs, &lows, &volumes);
    
    // 计算波动率
    let volatility = trend::calculate_historical_volatility(&prices, 20);
    
    // 多因子评分
    let multi_factor_score = multi_factor::calculate_multi_factor_score(
        &trend_analysis.overall_trend,
        &volume_signal,
        &tech_indicators,
        &patterns,
        &sr,
        volatility,
    );
    
    // 生成预测
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    
    for day in 1..=request.prediction_days {
        let target_date = get_next_trading_day(last_date);
        
        // 计算预测变化率
        let base_change = multi_factor::calculate_multi_factor_adjustment(&multi_factor_score);
        let trend_bias = trend_analysis.trend_strength * 0.01;
        let time_decay = 0.95_f64.powi(day as i32 - 1);
        
        let change_percent = (base_change + trend_bias) * time_decay * 100.0;
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        // 置信度随时间衰减
        let confidence = (multi_factor_score.signal_strength * time_decay).max(0.3);
        
        predictions.push(Prediction {
            target_date: target_date.format("%Y-%m-%d").to_string(),
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(multi_factor_score.signal.clone()),
            signal_strength: Some(multi_factor_score.signal_strength),
            technical_indicators: Some(convert_indicators(&tech_indicators)),
            prediction_reason: Some(format!("基于多因子综合评分 {:.1}", multi_factor_score.total_score)),
            key_factors: Some(volume_signal.key_factors.clone()),
        });
        
        last_date = target_date;
        last_price = predicted_price;
    }
    
    Ok(PredictionResponse {
        predictions,
        last_real_data: Some(LastRealData {
            date: last_data.date.format("%Y-%m-%d").to_string(),
            price: current_price,
            change_percent: last_data.change_percent,
        }),
    })
}

/// 简化策略预测
pub async fn predict_simple(request: PredictionRequest) -> Result<PredictionResponse, String> {
    predict(request).await
}

/// 评估模型
pub async fn evaluate_model(model_id: String) -> Result<EvaluationResult, String> {
    let metadata = load_model_metadata(&model_id)?;
    
    Ok(EvaluationResult {
        model_id,
        model_name: metadata.name,
        stock_code: metadata.stock_code,
        test_samples: 0,
        accuracy: metadata.accuracy,
        direction_accuracy: metadata.accuracy,
        mse: 0.0,
        mae: 0.0,
        rmse: 0.0,
        evaluation_date: chrono::Local::now().format("%Y-%m-%d").to_string(),
    })
}

fn convert_indicators(ind: &indicators::TechnicalIndicatorValues) -> TechnicalIndicatorValues {
    TechnicalIndicatorValues {
        rsi: ind.rsi,
        macd_histogram: ind.macd_histogram,
        kdj_j: ind.kdj_j,
        cci: ind.cci,
        obv_trend: ind.obv_trend,
        macd_dif: ind.macd_dif,
        macd_dea: ind.macd_dea,
        kdj_k: ind.kdj_k,
        kdj_d: ind.kdj_d,
        macd_golden_cross: ind.macd_golden_cross,
        macd_death_cross: ind.macd_death_cross,
        kdj_golden_cross: ind.kdj_golden_cross,
        kdj_death_cross: ind.kdj_death_cross,
        kdj_overbought: ind.kdj_overbought,
        kdj_oversold: ind.kdj_oversold,
    }
}

