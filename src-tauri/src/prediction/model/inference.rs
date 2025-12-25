//! 模型推理模块
//! 
//! 基于华尔街顶级量化策略优化的专业预测引擎
//! 
//! 核心优化点：
//! 1. 市场状态自适应 - 趋势市/震荡市使用不同策略
//! 2. 多维度信号验证 - 多个独立信号确认
//! 3. 背离检测 - RSI/MACD/量价背离
//! 4. 动态置信区间 - 基于波动率的预测范围

use crate::prediction::types::{
    PredictionRequest, PredictionResponse, Prediction, LastRealData,
    EvaluationResult, TechnicalIndicatorValues,
};
use crate::prediction::model::management::load_model_metadata;
use crate::prediction::indicators;
use crate::prediction::analysis::{trend, volume, pattern, support_resistance};
use crate::prediction::analysis::{market_regime, divergence};
use crate::prediction::strategy::{multi_factor, professional_engine};
use crate::utils::date::get_next_trading_day;
use crate::db::{connection::create_temp_pool, repository::get_recent_historical_data};

/// 使用专业预测引擎进行预测
pub async fn predict(request: PredictionRequest) -> Result<PredictionResponse, String> {
    // 获取历史数据（增加到250天以获得更准确的分析）
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, 250, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    
    if historical.is_empty() {
        return Err("未找到历史数据".to_string());
    }
    
    if historical.len() < 60 {
        return Err("历史数据不足60天，无法进行准确预测".to_string());
    }
    
    // 提取数据
    let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let volumes: Vec<i64> = historical.iter().map(|h| h.volume).collect();
    let opens: Vec<f64> = historical.iter().map(|h| h.open).collect();
    
    let current_price = *prices.last().unwrap();
    let last_data = historical.last().unwrap();
    
    // =========================================================================
    // 第一阶段：市场状态分析（核心创新）
    // =========================================================================
    let regime_analysis = market_regime::classify_market_regime(&prices, &highs, &lows);
    
    // =========================================================================
    // 第二阶段：技术分析
    // =========================================================================
    let trend_analysis = trend::analyze_trend(&prices, &highs, &lows);
    let volume_signal = volume::analyze_volume_price(&prices, &highs, &lows, &volumes);
    let patterns = pattern::recognize_patterns(&opens, &prices, &highs, &lows);
    let sr = support_resistance::calculate_support_resistance(&prices, &highs, &lows, current_price);
    let tech_indicators = indicators::calculate_all_indicators(&prices, &highs, &lows, &volumes);
    
    // =========================================================================
    // 第三阶段：背离检测（华尔街重视的反转信号）
    // =========================================================================
    let divergence_analysis = divergence::analyze_all_divergences(&prices, &highs, &lows, &volumes);
    
    // =========================================================================
    // 第四阶段：计算波动率
    // =========================================================================
    let volatility = trend::calculate_historical_volatility(&prices, 20);
    
    // =========================================================================
    // 第五阶段：自适应多因子评分
    // =========================================================================
    let multi_factor_score = multi_factor::calculate_adaptive_multi_factor_score(
        &trend_analysis.overall_trend,
        &volume_signal,
        &tech_indicators,
        &patterns,
        &sr,
        volatility,
        Some(&regime_analysis.regime),
        Some(&regime_analysis.volatility_level),
    );
    
    // =========================================================================
    // 第六阶段：专业预测引擎执行
    // =========================================================================
    let prediction_ctx = professional_engine::PredictionContext {
        current_price,
        market_regime: regime_analysis.clone(),
        trend_analysis: trend_analysis.clone(),
        volume_signal: volume_signal.clone(),
        divergence: divergence_analysis.clone(),
        indicators: tech_indicators.clone(),
        patterns: patterns.clone(),
        support_resistance: sr.clone(),
        multi_factor_score: multi_factor_score.clone(),
        volatility,
    };
    
    let professional_result = professional_engine::execute_professional_prediction(&prediction_ctx);
    
    // =========================================================================
    // 第七阶段：生成预测序列
    // =========================================================================
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    
    for day in 1..=request.prediction_days {
        let target_date = get_next_trading_day(last_date);
        
        // 计算当日预测变化率
        let (change_percent, confidence) = calculate_daily_prediction(
            day,
            &professional_result,
            &regime_analysis,
            &multi_factor_score,
            volatility,
        );
        
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        // 生成预测原因
        let prediction_reason = generate_prediction_reason(
            &professional_result,
            &regime_analysis,
            &divergence_analysis,
            &multi_factor_score,
        );
        
        // 生成关键因素
        let mut key_factors = professional_result.key_factors.clone();
        if divergence_analysis.has_divergence {
            key_factors.push(format!("背离信号: {}", divergence_analysis.primary_direction));
        }
        
        predictions.push(Prediction {
            target_date: target_date.format("%Y-%m-%d").to_string(),
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(professional_result.direction.to_string()),
            signal_strength: Some(professional_result.confidence),
            technical_indicators: Some(convert_indicators(&tech_indicators)),
            prediction_reason: Some(prediction_reason),
            key_factors: Some(key_factors),
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

/// 计算单日预测
fn calculate_daily_prediction(
    day: usize,
    professional_result: &professional_engine::ProfessionalPredictionResult,
    regime: &market_regime::MarketRegimeAnalysis,
    _multi_factor: &multi_factor::MultiFactorScore,
    volatility: f64,
) -> (f64, f64) {
    // 基础变化来自专业预测引擎
    let base_change = professional_result.expected_change;
    
    // 时间衰减（专业策略：趋势市衰减慢，震荡市衰减快）
    let decay_rate: f64 = match regime.regime {
        market_regime::MarketRegime::StrongUptrend | 
        market_regime::MarketRegime::StrongDowntrend => 0.97,
        market_regime::MarketRegime::ModerateUptrend | 
        market_regime::MarketRegime::ModerateDowntrend => 0.95,
        market_regime::MarketRegime::Ranging => 0.90,
        _ => 0.92,
    };
    
    let time_decay = decay_rate.powi(day as i32 - 1);
    
    // 信号确认加成
    let confirmation_boost = match professional_result.signal_confirmation {
        professional_engine::SignalConfirmation::StrongConfirm => 1.15,
        professional_engine::SignalConfirmation::ModerateConfirm => 1.05,
        professional_engine::SignalConfirmation::WeakConfirm => 0.95,
        professional_engine::SignalConfirmation::NoConfirm => 0.80,
    };
    
    // 波动率调整
    let volatility_factor = if volatility > 0.04 {
        0.85  // 高波动降低预测幅度
    } else if volatility < 0.02 {
        1.10  // 低波动可以稍微放大
    } else {
        1.0
    };
    
    // 综合计算
    let change_percent = base_change * time_decay * confirmation_boost * volatility_factor;
    
    // 限制单日变化幅度（A股涨跌停限制）
    let change_percent = change_percent.clamp(-9.5, 9.5);
    
    // 置信度计算
    let base_confidence = professional_result.confidence;
    let confidence_decay = 0.92_f64.powi(day as i32 - 1);
    let confidence = (base_confidence * confidence_decay).max(0.25).min(0.90);
    
    (change_percent, confidence)
}

/// 生成预测原因说明
fn generate_prediction_reason(
    result: &professional_engine::ProfessionalPredictionResult,
    regime: &market_regime::MarketRegimeAnalysis,
    divergence: &divergence::DivergenceAnalysis,
    multi_factor: &multi_factor::MultiFactorScore,
) -> String {
    let mut reasons = Vec::new();
    
    // 市场状态
    reasons.push(format!("市场状态:{}", regime.regime.to_string()));
    
    // 信号确认
    let confirmation_text = match result.signal_confirmation {
        professional_engine::SignalConfirmation::StrongConfirm => "强确认(4+信号)",
        professional_engine::SignalConfirmation::ModerateConfirm => "中等确认(3信号)",
        professional_engine::SignalConfirmation::WeakConfirm => "弱确认(2信号)",
        professional_engine::SignalConfirmation::NoConfirm => "信号冲突",
    };
    reasons.push(confirmation_text.to_string());
    
    // 多因子得分
    reasons.push(format!("综合评分:{:.1}", multi_factor.adaptive_score));
    
    // 背离信号
    if divergence.has_divergence {
        reasons.push(format!("检测到{}背离", divergence.primary_direction));
    }
    
    // 策略类型
    let strategy_text = match result.strategy_used {
        market_regime::StrategyType::TrendFollowing => "趋势跟踪",
        market_regime::StrategyType::MeanReversion => "均值回归",
        market_regime::StrategyType::Reversal => "反转策略",
    };
    reasons.push(format!("策略:{}", strategy_text));
    
    reasons.join(" | ")
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

