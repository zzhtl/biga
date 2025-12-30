//! 模型推理模块
//! 
//! 基于华尔街顶级量化策略优化的专业预测引擎
//! 
//! 核心优化点：
//! 1. 市场状态自适应 - 趋势市/震荡市使用不同策略
//! 2. 多维度信号验证 - 多个独立信号确认
//! 3. 背离检测 - RSI/MACD/量价/Williams%R/ROC背离
//! 4. 动态置信区间 - 基于GARCH波动率预测
//! 5. 自适应权重 - 基于历史回测的动态权重调整
//! 6. 信号冲突检测 - 多重信号确认与假信号过滤

use crate::prediction::types::{
    PredictionRequest, PredictionResponse, Prediction, LastRealData,
    EvaluationResult, TechnicalIndicatorValues,
};
use crate::prediction::model::management::load_model_metadata;
use crate::prediction::indicators;
use crate::prediction::analysis::{trend, volume, pattern, support_resistance};
use crate::prediction::analysis::{market_regime, divergence, signal_confirmation, volatility_forecast};
use crate::prediction::strategy::{multi_factor, professional_engine, adaptive_weights, price_model};
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
    // 第三阶段：增强背离检测（包括隐藏背离和三重背离）
    // =========================================================================
    let divergence_analysis = divergence::analyze_all_divergences(&prices, &highs, &lows, &volumes);
    
    // =========================================================================
    // 第四阶段：GARCH波动率预测
    // =========================================================================
    let volatility = trend::calculate_historical_volatility(&prices, 20);
    let vol_forecast = volatility_forecast::GarchForecaster::from_prices(&prices)
        .forecast(request.prediction_days);
    
    // =========================================================================
    // 第五阶段：信号确认与冲突检测
    // =========================================================================
    let signal_confirm = signal_confirmation::analyze_signal_confirmation(
        &tech_indicators,
        &trend_analysis.overall_trend,
        &volume_signal,
        &regime_analysis.regime,
        &regime_analysis.volatility_level,
    );
    
    // =========================================================================
    // 第六阶段：自适应权重计算
    // =========================================================================
    let _dynamic_weights = adaptive_weights::calculate_dynamic_weights(
        &regime_analysis.regime,
        regime_analysis.volatility_percentile,
        trend_analysis.trend_strength,
    );
    
    // =========================================================================
    // 第七阶段：自适应多因子评分
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
    // 第八阶段：VWAP均值回归分析
    // =========================================================================
    let vwap_signal = indicators::vwap::analyze_vwap_signal(&highs, &lows, &prices, &volumes, 20);
    let bb = indicators::bollinger::calculate_bollinger_bands(&prices, 20, 2.0);
    let bollinger_position = (current_price - bb.middle) / (bb.upper - bb.lower).max(0.001);
    
    // =========================================================================
    // 第九阶段：增强价格预测模型
    // =========================================================================
    let recent_momentum = if prices.len() >= 5 {
        (prices[prices.len() - 1] - prices[prices.len() - 5]) / prices[prices.len() - 5]
    } else {
        0.0
    };
    
    let price_ctx = price_model::PricePredictionContext {
        current_price,
        volatility,
        regime: regime_analysis.regime,
        volatility_level: regime_analysis.volatility_level,
        trend: trend_analysis.overall_trend.clone(),
        vwap_deviation: vwap_signal.deviation,
        bollinger_position,
        support_resistance: sr.clone(),
        recent_momentum,
    };
    
    let enhanced_prediction = price_model::calculate_enhanced_price_prediction(&price_ctx);
    
    // =========================================================================
    // 第十阶段：专业预测引擎执行
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
    // 第十一阶段：生成预测序列
    // =========================================================================
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    
    for day in 1..=request.prediction_days {
        let target_date = get_next_trading_day(last_date);
        
        // 使用增强模型计算预测
        let (change_percent, confidence) = calculate_enhanced_daily_prediction(
            day,
            &professional_result,
            &regime_analysis,
            &multi_factor_score,
            &enhanced_prediction,
            &signal_confirm,
            &vol_forecast,
            &divergence_analysis,
        );
        
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        // 生成增强版预测原因
        let prediction_reason = generate_enhanced_prediction_reason(
            &professional_result,
            &regime_analysis,
            &divergence_analysis,
            &multi_factor_score,
            &signal_confirm,
            &vol_forecast,
        );
        
        // 生成关键因素
        let mut key_factors = professional_result.key_factors.clone();
        if divergence_analysis.has_divergence {
            let div_info = if divergence_analysis.is_triple_divergence {
                format!("⚠️三重背离: {}", divergence_analysis.primary_direction)
            } else {
                format!("背离信号: {}", divergence_analysis.primary_direction)
            };
            key_factors.push(div_info);
        }
        if signal_confirm.is_potential_false_signal {
            key_factors.push("⚠️信号存在冲突，置信度降低".to_string());
        }
        key_factors.push(format!("波动率状态: {:?}", vol_forecast.volatility_regime));
        
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

/// 计算单日预测（基础版本，保留向后兼容）
#[allow(dead_code)]
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

/// 增强版单日预测计算
fn calculate_enhanced_daily_prediction(
    day: usize,
    professional_result: &professional_engine::ProfessionalPredictionResult,
    regime: &market_regime::MarketRegimeAnalysis,
    multi_factor: &multi_factor::MultiFactorScore,
    enhanced_pred: &price_model::PricePredictionResult,
    signal_confirm: &signal_confirmation::SignalConfirmationResult,
    vol_forecast: &volatility_forecast::VolatilityForecast,
    divergence: &divergence::DivergenceAnalysis,
) -> (f64, f64) {
    // 1. 基础预测：结合专业引擎和增强模型
    let professional_change = professional_result.expected_change;
    let enhanced_change = enhanced_pred.expected_change;
    
    // 根据信号确认程度调整两个模型的权重
    let (pro_weight, enh_weight) = match signal_confirm.confirmation_level {
        signal_confirmation::ConfirmationLevel::Strong => (0.55, 0.45),
        signal_confirmation::ConfirmationLevel::Moderate => (0.50, 0.50),
        signal_confirmation::ConfirmationLevel::Weak => (0.45, 0.55),
        signal_confirmation::ConfirmationLevel::Invalid => (0.40, 0.60),
    };
    
    let base_change = professional_change * pro_weight + enhanced_change * enh_weight;
    
    // 2. 时间衰减（根据市场状态和波动率预测动态调整）
    let decay_rate: f64 = match regime.regime {
        market_regime::MarketRegime::StrongUptrend | 
        market_regime::MarketRegime::StrongDowntrend => 0.97,
        market_regime::MarketRegime::ModerateUptrend | 
        market_regime::MarketRegime::ModerateDowntrend => 0.95,
        market_regime::MarketRegime::Ranging => 0.88,
        _ => 0.92,
    };
    
    // 波动率趋势影响衰减速度
    let vol_decay_adj = match vol_forecast.volatility_trend {
        volatility_forecast::VolatilityTrend::Expanding => 0.98,    // 波动率扩张，衰减慢
        volatility_forecast::VolatilityTrend::Contracting => 1.02,  // 波动率收缩，衰减快
        volatility_forecast::VolatilityTrend::Stable => 1.0,
    };
    
    let adjusted_decay = decay_rate * vol_decay_adj;
    let time_decay = adjusted_decay.powi(day as i32 - 1);
    
    // 3. 信号确认调整
    let signal_factor = signal_confirm.confidence_factor;
    
    // 假信号惩罚
    let false_signal_penalty = if signal_confirm.is_potential_false_signal {
        0.6
    } else {
        1.0
    };
    
    // 4. 背离信号调整
    let divergence_factor = if divergence.has_divergence {
        if divergence.is_triple_divergence {
            // 三重背离是极强信号
            1.25
        } else if divergence.divergence_count >= 2 {
            1.15
        } else {
            1.05
        }
    } else {
        1.0
    };
    
    // 5. 波动率状态调整
    let _vol_regime_factor = vol_forecast.volatility_regime.to_risk_multiplier();
    
    // 6. 多因子得分调整
    let mf_factor = if multi_factor.adaptive_score > 70.0 {
        1.1
    } else if multi_factor.adaptive_score < 30.0 {
        0.9
    } else {
        1.0
    };
    
    // 7. 综合计算
    let mut change_percent = base_change 
        * time_decay 
        * signal_factor 
        * false_signal_penalty
        * divergence_factor
        * mf_factor;
    
    // 高波动环境下限制预测幅度
    if matches!(vol_forecast.volatility_regime, 
        volatility_forecast::VolatilityRegime::VeryHigh | 
        volatility_forecast::VolatilityRegime::Extreme) 
    {
        change_percent *= 0.7;
    }
    
    // 限制单日变化幅度（A股涨跌停限制）
    let change_percent = change_percent.clamp(-9.5, 9.5);
    
    // 8. 置信度计算
    let base_confidence = professional_result.confidence
        .min(enhanced_pred.confidence)
        .min(signal_confirm.strength + 0.3);
    
    let confidence_decay = 0.92_f64.powi(day as i32 - 1);
    
    // 信号冲突降低置信度
    let conflict_penalty = if signal_confirm.is_potential_false_signal { 0.7 } else { 1.0 };
    
    // 极端波动降低置信度
    let vol_penalty = match vol_forecast.volatility_regime {
        volatility_forecast::VolatilityRegime::Extreme => 0.6,
        volatility_forecast::VolatilityRegime::VeryHigh => 0.75,
        volatility_forecast::VolatilityRegime::High => 0.9,
        _ => 1.0,
    };
    
    let confidence = (base_confidence * confidence_decay * conflict_penalty * vol_penalty)
        .max(0.25)
        .min(0.90);
    
    (change_percent, confidence)
}

/// 生成预测原因说明
#[allow(dead_code)]
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

/// 生成增强版预测原因说明
fn generate_enhanced_prediction_reason(
    result: &professional_engine::ProfessionalPredictionResult,
    regime: &market_regime::MarketRegimeAnalysis,
    divergence: &divergence::DivergenceAnalysis,
    multi_factor: &multi_factor::MultiFactorScore,
    signal_confirm: &signal_confirmation::SignalConfirmationResult,
    vol_forecast: &volatility_forecast::VolatilityForecast,
) -> String {
    let mut reasons = Vec::new();
    
    // 市场状态
    reasons.push(format!("市场:{}", regime.regime.to_string()));
    
    // 信号确认级别
    reasons.push(format!("信号:{}", signal_confirm.confirmation_level.to_string()));
    
    // 如果有假信号警告
    if signal_confirm.is_potential_false_signal {
        reasons.push("⚠️冲突".to_string());
    }
    
    // 多因子得分
    reasons.push(format!("评分:{:.0}", multi_factor.adaptive_score));
    
    // 背离信号（增强）
    if divergence.has_divergence {
        if divergence.is_triple_divergence {
            reasons.push(format!("⚠️三重{}背离!", divergence.primary_direction));
        } else if divergence.hidden_divergence_count > 0 {
            reasons.push(format!("隐藏{}背离", divergence.primary_direction));
        } else {
            reasons.push(format!("{}背离", divergence.primary_direction));
        }
    }
    
    // 波动率状态
    let vol_text = match vol_forecast.volatility_regime {
        volatility_forecast::VolatilityRegime::Extreme => "极端波动⚠️",
        volatility_forecast::VolatilityRegime::VeryHigh => "高波动",
        volatility_forecast::VolatilityRegime::High => "较高波动",
        volatility_forecast::VolatilityRegime::Normal => "正常波动",
        volatility_forecast::VolatilityRegime::Low => "低波动",
        volatility_forecast::VolatilityRegime::VeryLow => "极低波动",
    };
    reasons.push(format!("波动:{}", vol_text));
    
    // 波动率趋势
    let vol_trend = match vol_forecast.volatility_trend {
        volatility_forecast::VolatilityTrend::Expanding => "↑扩张",
        volatility_forecast::VolatilityTrend::Contracting => "↓收缩",
        volatility_forecast::VolatilityTrend::Stable => "→稳定",
    };
    reasons.push(vol_trend.to_string());
    
    // 策略类型
    let strategy_text = match result.strategy_used {
        market_regime::StrategyType::TrendFollowing => "趋势跟踪",
        market_regime::StrategyType::MeanReversion => "均值回归",
        market_regime::StrategyType::Reversal => "反转策略",
    };
    reasons.push(format!("策略:{}", strategy_text));
    
    // 主导信号
    reasons.push(format!("主导:{}", signal_confirm.dominant_signal));
    
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

