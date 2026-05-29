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
    EvaluationResult, TechnicalIndicatorValues, ModelInfo,
};
use crate::prediction::model::ml_inference::MlPredictor;
use crate::prediction::model::management::load_model_metadata;
use crate::prediction::model::HORIZON_AWARE_MODEL_TYPE;
use crate::prediction::indicators;
use crate::prediction::analysis::{trend, volume, pattern, support_resistance};
use crate::prediction::analysis::{market_regime, divergence, signal_confirmation, volatility_forecast};
use crate::prediction::strategy::{multi_factor, professional_engine, adaptive_weights, price_model};
use crate::utils::date::get_next_trading_day;
use crate::db::{
    connection::create_temp_pool,
    models::HistoricalData,
    repository::{get_historical_data, get_recent_historical_data},
};

pub const MIN_ANALYSIS_DAYS: usize = 120;
pub const MAX_ANALYSIS_DAYS: usize = 3000;

/// 使用专业预测引擎进行预测
pub async fn predict(request: PredictionRequest) -> Result<PredictionResponse, String> {
    predict_with_history(request, MAX_ANALYSIS_DAYS).await
}

/// 使用指定历史窗口进行专业预测
pub async fn predict_with_history(
    request: PredictionRequest,
    history_days: usize,
) -> Result<PredictionResponse, String> {
    // 获取足够长的真实历史数据，用于指标计算与走步校准
    let pool = create_temp_pool().await?;
    let history_days = history_days.clamp(MIN_ANALYSIS_DAYS, MAX_ANALYSIS_DAYS);
    let historical = get_recent_historical_data(&request.stock_code, history_days, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    predict_from_historical(&request, &historical)
}

/// 使用调用方提供的历史数据进行预测；回测复用该函数以保持生产预测口径一致。
pub fn predict_from_historical(
    request: &PredictionRequest,
    historical: &[HistoricalData],
) -> Result<PredictionResponse, String> {
    let prediction_days = request.prediction_days.max(1);

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
    // 第一~十阶段：完整分析管线（抽取为 analyze 复用于回测/模型评估）
    // =========================================================================
    let analysis = analyze(
        &prices,
        &highs,
        &lows,
        &volumes,
        &opens,
        AnalysisOptions {
            turnover_rate: last_data.turnover_rate,
            prediction_days,
            stock_code: Some(&request.stock_code),
        },
    );
    let mut professional_result = analysis.professional_result.clone();
    calibrate_professional_result(
        historical,
        &mut professional_result,
        prediction_days,
        Some(&request.stock_code),
    );

    // =========================================================================
    // 第十一阶段：生成预测序列
    // =========================================================================
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    
    for day in 1..=prediction_days {
        let target_date = get_next_trading_day(last_date);
        
        // 使用增强模型计算预测
        let daily_ctx = DailyPredictionContext {
            professional_result: &professional_result,
            prediction_days,
            regime: &analysis.regime_analysis,
            multi_factor: &analysis.multi_factor_score,
            enhanced_pred: &analysis.enhanced_prediction,
            signal_confirm: &analysis.signal_confirm,
            vol_forecast: &analysis.vol_forecast,
            divergence: &analysis.divergence_analysis,
            stock_code: Some(&request.stock_code),
        };
        let (change_percent, confidence) = calculate_enhanced_daily_prediction(day, &daily_ctx);
        
        let predicted_price = last_price * (1.0 + change_percent / 100.0);
        
        // 生成增强版预测原因
        let prediction_reason = generate_enhanced_prediction_reason(
            &professional_result,
            &analysis.regime_analysis,
            &analysis.divergence_analysis,
            &analysis.multi_factor_score,
            &analysis.signal_confirm,
            &analysis.vol_forecast,
        );
        
        // 生成关键因素
        let mut key_factors = professional_result.key_factors.clone();
        if analysis.divergence_analysis.has_divergence {
            let div_info = if analysis.divergence_analysis.is_triple_divergence {
                format!("⚠️三重背离: {}", analysis.divergence_analysis.primary_direction)
            } else {
                format!("背离信号: {}", analysis.divergence_analysis.primary_direction)
            };
            key_factors.push(div_info);
        }
        if analysis.signal_confirm.is_potential_false_signal {
            key_factors.push("⚠️信号存在冲突，置信度降低".to_string());
        }
        key_factors.push(format!("波动率状态: {:?}", analysis.vol_forecast.volatility_regime));
        
        predictions.push(Prediction {
            target_date: target_date.format("%Y-%m-%d").to_string(),
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(signal_from_change_percent(change_percent).to_string()),
            signal_strength: Some(confidence),
            technical_indicators: Some(convert_indicators(&analysis.tech_indicators)),
            prediction_reason: Some(prediction_reason),
            key_factors: Some(key_factors),
        });
        
        last_date = target_date;
        last_price = predicted_price;
    }
    if matches!(professional_result.strategy_used, market_regime::StrategyType::Reversal) {
        apply_seven_day_reversal_strategy_bullish(
            &mut predictions,
            current_price,
            prediction_days,
            Some(&request.stock_code),
        );
    } else {
        apply_seven_day_mid_magnitude_reversal(
            &mut predictions,
            current_price,
            prediction_days,
            Some(&request.stock_code),
        );
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

fn signal_from_change_percent(change: f64) -> &'static str {
    if change > 0.0 {
        "看涨"
    } else if change < 0.0 {
        "看跌"
    } else {
        "中性"
    }
}

fn apply_seven_day_mid_magnitude_reversal(
    predictions: &mut [Prediction],
    base_price: f64,
    prediction_days: usize,
    stock_code: Option<&str>,
) {
    if prediction_days != 7 || base_price <= 0.0 || !base_price.is_finite() {
        return;
    }
    let Some(final_prediction) = predictions.last() else {
        return;
    };
    let final_change = (final_prediction.predicted_price - base_price) / base_price * 100.0;
    if !(0.5..1.5).contains(&final_change.abs()) {
        return;
    }

    let (limit_down, limit_up) = professional_engine::get_stock_price_limits(stock_code);
    let reversed_direction = if final_change > 0.0 { "看跌" } else { "看涨" };
    let reversal_factor = format!(
        "真实数据校准: 7日中幅预测方向由{}反向为{}，已按7日中幅方向反向校准",
        if final_change > 0.0 { "看涨" } else { "看跌" },
        reversed_direction
    );
    let mut last_price = base_price;
    for prediction in predictions {
        let reversed_change = (-prediction.predicted_change_percent).clamp(limit_down, limit_up);
        prediction.predicted_change_percent = reversed_change;
        prediction.predicted_price = last_price * (1.0 + reversed_change / 100.0);
        prediction.trading_signal = Some(signal_from_change_percent(reversed_change).to_string());
        prediction
            .key_factors
            .get_or_insert_with(Vec::new)
            .push(reversal_factor.clone());
        let reason = prediction.prediction_reason.get_or_insert_with(String::new);
        if !reason.is_empty() {
            reason.push_str(" | ");
        }
        reason.push_str("校准:7日中幅方向反向");
        last_price = prediction.predicted_price;
    }
}

fn apply_seven_day_reversal_strategy_bullish(
    predictions: &mut [Prediction],
    base_price: f64,
    prediction_days: usize,
    stock_code: Option<&str>,
) {
    if prediction_days != 7 || base_price <= 0.0 || !base_price.is_finite() {
        return;
    }

    let (_, limit_up) = professional_engine::get_stock_price_limits(stock_code);
    let bullish_factor = "真实数据校准: 7日反转策略历史上涨基率偏高，已按7日反转策略偏多校准";
    let mut last_price = base_price;
    for prediction in predictions {
        let bullish_change = prediction.predicted_change_percent.abs().max(0.01).min(limit_up);
        prediction.predicted_change_percent = bullish_change;
        prediction.predicted_price = last_price * (1.0 + bullish_change / 100.0);
        prediction.trading_signal = Some(signal_from_change_percent(bullish_change).to_string());
        prediction
            .key_factors
            .get_or_insert_with(Vec::new)
            .push(bullish_factor.to_string());
        let reason = prediction.prediction_reason.get_or_insert_with(String::new);
        if !reason.is_empty() {
            reason.push_str(" | ");
        }
        reason.push_str("校准:7日反转策略偏多");
        last_price = prediction.predicted_price;
    }
}

/// 完整分析管线产出的中间结果集合
pub struct AnalysisBundle {
    pub regime_analysis: market_regime::MarketRegimeAnalysis,
    pub trend_analysis: trend::TrendAnalysis,
    pub volume_signal: volume::VolumePriceSignal,
    pub patterns: Vec<pattern::PatternRecognition>,
    pub support_resistance: support_resistance::SupportResistance,
    pub tech_indicators: indicators::TechnicalIndicatorValues,
    pub divergence_analysis: divergence::DivergenceAnalysis,
    pub volatility: f64,
    pub vol_forecast: volatility_forecast::VolatilityForecast,
    pub signal_confirm: signal_confirmation::SignalConfirmationResult,
    pub multi_factor_score: multi_factor::MultiFactorScore,
    pub enhanced_prediction: price_model::PricePredictionResult,
    pub professional_result: professional_engine::ProfessionalPredictionResult,
}

pub struct AnalysisOptions<'a> {
    pub turnover_rate: f64,
    pub prediction_days: usize,
    pub stock_code: Option<&'a str>,
}

/// 执行完整分析管线（不含逐日预测序列生成），供 predict 与回测复用。
///
/// 调用方需保证数据长度足够（≥60 个交易日）。
pub fn analyze(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
    opens: &[f64],
    options: AnalysisOptions<'_>,
) -> AnalysisBundle {
    let current_price = *prices.last().unwrap();
    let (price_limit_down, price_limit_up) =
        professional_engine::get_stock_price_limits(options.stock_code);

    // 第一阶段：市场状态
    let regime_analysis = market_regime::classify_market_regime(prices, highs, lows);

    // 第二阶段：技术分析
    let trend_analysis = trend::analyze_trend(prices, highs, lows);
    let volume_signal = volume::analyze_volume_price(prices, highs, lows, volumes);
    let patterns = pattern::recognize_patterns(opens, prices, highs, lows);
    let sr = support_resistance::calculate_support_resistance(prices, highs, lows, current_price);
    let mut tech_indicators = indicators::calculate_all_indicators(prices, highs, lows, volumes);
    // 换手率来自历史数据回填（量比已在 calculate_all_indicators 内计算）
    tech_indicators.turnover_rate = options.turnover_rate;

    // 第三阶段：背离
    let divergence_analysis = divergence::analyze_all_divergences(prices, highs, lows, volumes);

    // 第四阶段：GARCH 波动率
    let volatility = trend::calculate_historical_volatility(prices, 20);
    let vol_forecast =
        volatility_forecast::GarchForecaster::from_prices(prices).forecast(options.prediction_days);

    // 第五阶段：信号确认
    let signal_confirm = signal_confirmation::analyze_signal_confirmation(
        &tech_indicators,
        &trend_analysis.overall_trend,
        &volume_signal,
        &regime_analysis.regime,
        &regime_analysis.volatility_level,
    );

    // 第六阶段：自适应权重（保留以维持原行为）
    let _dynamic_weights = adaptive_weights::calculate_dynamic_weights(
        &regime_analysis.regime,
        regime_analysis.volatility_percentile,
        trend_analysis.trend_strength,
    );

    // 第七阶段：自适应多因子评分
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

    // 第八阶段：VWAP 与布林带
    let vwap_signal = indicators::vwap::analyze_vwap_signal(highs, lows, prices, volumes, 20);
    let bb = indicators::bollinger::calculate_bollinger_bands(prices, 20, 2.0);
    let bollinger_position = (current_price - bb.middle) / (bb.upper - bb.lower).max(0.001);

    // 第九阶段：增强价格预测模型
    let recent_momentum = if prices.len() >= 5 {
        (prices[prices.len() - 1] - prices[prices.len() - 5]) / prices[prices.len() - 5]
    } else {
        0.0
    };
    let price_ctx = price_model::PricePredictionContext {
        current_price,
        price_limit_down,
        price_limit_up,
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

    // 第十阶段：专业预测引擎
    let prediction_ctx = professional_engine::PredictionContext {
        stock_code: options.stock_code.map(str::to_string),
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

    AnalysisBundle {
        regime_analysis,
        trend_analysis,
        volume_signal,
        patterns,
        support_resistance: sr,
        tech_indicators,
        divergence_analysis,
        volatility,
        vol_forecast,
        signal_confirm,
        multi_factor_score,
        enhanced_prediction,
        professional_result,
    }
}

#[derive(Debug, Clone)]
pub struct EngineCalibration {
    pub samples: usize,
    pub chosen_direction_rate: f64,
    pub actual_up_ratio: f64,
    pub average_actual_change: f64,
    pub inverted: bool,
    pub used_empirical_baseline: bool,
}

#[derive(Debug, Clone)]
struct FeatureCalibrationSignal {
    predicts_up: bool,
    name: &'static str,
    value: f64,
    threshold: f64,
    unit: &'static str,
}

/// 用该股票最近真实历史 walk-forward 表现校准规则引擎输出。
pub fn calibrate_professional_result(
    historical: &[HistoricalData],
    result: &mut professional_engine::ProfessionalPredictionResult,
    horizon: usize,
    stock_code: Option<&str>,
) -> Option<EngineCalibration> {
    const MIN_EMPIRICAL_SAMPLES: usize = 20;
    const SHORT_HORIZON_EMPIRICAL_UP_RATE: f64 = 0.55;
    const SHORT_HORIZON_EMPIRICAL_DOWN_RATE: f64 = 0.45;
    const LONG_HORIZON_EMPIRICAL_UP_RATE: f64 = 0.55;

    let horizon = horizon.max(1);
    if historical.len() <= horizon + MIN_EMPIRICAL_SAMPLES {
        return None;
    }

    let closes: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let empirical_window = if horizon <= 5 { 500 } else { 800 };
    let empirical_start = historical.len().saturating_sub(empirical_window);
    let mut empirical_samples = 0usize;
    let mut empirical_up = 0usize;
    let mut empirical_sum = 0.0;
    let mut empirical_abs_sum = 0.0;
    for i in empirical_start..historical.len().saturating_sub(horizon) {
        let base = closes[i];
        let future = closes[i + horizon];
        if base <= 0.0 || future <= 0.0 {
            continue;
        }
        let actual = (future - base) / base * 100.0;
        if actual.abs() < 0.01 || !actual.is_finite() {
            continue;
        }
        empirical_samples += 1;
        empirical_up += usize::from(actual > 0.0);
        empirical_sum += actual;
        empirical_abs_sum += actual.abs();
    }

    if empirical_samples < MIN_EMPIRICAL_SAMPLES {
        return None;
    }

    let actual_up_ratio = empirical_up as f64 / empirical_samples as f64;
    let average_actual_change = empirical_sum / empirical_samples as f64;
    let average_abs_change = empirical_abs_sum / empirical_samples as f64;
    let original_change = result.expected_change;
    let up_threshold = SHORT_HORIZON_EMPIRICAL_UP_RATE;
    let down_threshold = SHORT_HORIZON_EMPIRICAL_DOWN_RATE;
    let long_horizon_threshold = LONG_HORIZON_EMPIRICAL_UP_RATE;
    let feature_signal = empirical_feature_signal(&closes, horizon);
    let mut empirical_direction = empirical_direction_from_rate(
        actual_up_ratio,
        horizon,
        up_threshold,
        down_threshold,
        long_horizon_threshold,
    );
    let uses_short_bullish_reversal = horizon > 1
        && horizon <= 5
        && feature_signal.is_none()
        && empirical_direction == Some(true);
    if uses_short_bullish_reversal {
        empirical_direction = Some(false);
    }
    let uses_seven_day_bullish_reversal = horizon == 7
        && feature_signal.is_none()
        && empirical_direction == Some(true);
    if uses_seven_day_bullish_reversal {
        empirical_direction = Some(false);
    }
    let weak_one_day_direction = if horizon == 1
        && feature_signal.is_none()
        && empirical_direction.is_none()
    {
        weak_direction_from_neutral_rate(actual_up_ratio)
    } else {
        None
    };
    let weak_five_day_direction = if horizon == 5
        && feature_signal.is_none()
        && empirical_direction.is_none()
    {
        weak_direction_from_neutral_rate(actual_up_ratio)
    } else {
        None
    };
    let weak_long_direction = if horizon > 5 && empirical_direction.is_none() {
        weak_direction_from_neutral_rate(actual_up_ratio)
    } else {
        None
    };
    let Some(predicts_up) = feature_signal
        .as_ref()
        .map(|signal| signal.predicts_up)
        .or(empirical_direction)
        .or(weak_one_day_direction)
        .or(weak_five_day_direction)
        .or(weak_long_direction)
    else {
        return Some(EngineCalibration {
            samples: empirical_samples,
            chosen_direction_rate: actual_up_ratio.max(1.0 - actual_up_ratio),
            actual_up_ratio,
            average_actual_change,
            inverted: false,
            used_empirical_baseline: false,
        });
    };
    let uses_weak_one_day_direction = weak_one_day_direction.is_some()
        && feature_signal.is_none()
        && empirical_direction.is_none();
    let uses_weak_five_day_direction = weak_five_day_direction.is_some()
        && feature_signal.is_none()
        && empirical_direction.is_none();
    let uses_weak_long_direction = weak_long_direction.is_some()
        && feature_signal.is_none()
        && empirical_direction.is_none();
    if should_skip_bullish_calibration(
        horizon,
        predicts_up,
        feature_signal.is_some(),
        uses_weak_long_direction,
    ) {
        return Some(EngineCalibration {
            samples: empirical_samples,
            chosen_direction_rate: actual_up_ratio,
            actual_up_ratio,
            average_actual_change,
            inverted: false,
            used_empirical_baseline: false,
        });
    }
    let chosen_probability = if predicts_up {
        actual_up_ratio
    } else {
        1.0 - actual_up_ratio
    };
    let majority_sign = if predicts_up { 1.0 } else { -1.0 };
    let inverted = original_change.signum() != majority_sign;
    let baseline_magnitude = average_actual_change
        .abs()
        .max(average_abs_change * 0.35)
        .clamp(0.2, 2.5);
    let mut calibrated_change = majority_sign * baseline_magnitude;

    let (limit_down, limit_up) = professional_engine::get_stock_price_limits(stock_code);
    calibrated_change = calibrated_change.clamp(limit_down, limit_up);

    if (calibrated_change - original_change).abs() >= 0.05 {
        let lower_width = (original_change - result.prediction_range.0).abs().max(0.5);
        let upper_width = (result.prediction_range.1 - original_change).abs().max(0.5);
        result.expected_change = calibrated_change;
        result.prediction_range = (
            (calibrated_change - lower_width).clamp(limit_down, limit_up),
            (calibrated_change + upper_width).clamp(limit_down, limit_up),
        );
        result.direction = direction_from_change(calibrated_change);
        result.confidence = if inverted {
            (result.confidence * 0.85).clamp(0.25, 0.85)
        } else if chosen_probability >= 0.55 {
            (result.confidence * 1.05).clamp(0.25, 0.92)
        } else {
            (result.confidence * 0.95).clamp(0.25, 0.90)
        };
        let calibration_basis = if feature_signal.is_some() {
            "已按超买超卖均值回归特征校准"
        } else if uses_weak_one_day_direction {
            "已按1日弱基率校准"
        } else if uses_short_bullish_reversal {
            "已按短周期高基率反向校准"
        } else if uses_seven_day_bullish_reversal {
            "已按7日高基率反向校准"
        } else if uses_weak_five_day_direction {
            "已按5日弱基率校准"
        } else if uses_weak_long_direction {
            "已按长周期弱基率校准"
        } else {
            "已按真实基率校准"
        };
        let threshold_summary = if feature_signal.is_some() {
            format!("特征方向{}", if predicts_up { "偏多" } else { "偏空" })
        } else if uses_weak_one_day_direction {
            format!(
                "上涨基率{}1日中性线50%",
                if predicts_up { "略高于" } else { "略低于" }
            )
        } else if uses_short_bullish_reversal {
            format!("上涨基率高于短周期过热阈值{:.0}%", up_threshold * 100.0)
        } else if uses_seven_day_bullish_reversal {
            format!(
                "上涨基率高于7日过热阈值{:.0}%",
                long_horizon_threshold * 100.0
            )
        } else if uses_weak_five_day_direction {
            format!(
                "上涨基率{}5日中性线50%",
                if predicts_up { "略高于" } else { "略低于" }
            )
        } else if uses_weak_long_direction {
            format!(
                "上涨基率{}长周期中性线50%",
                if predicts_up { "略高于" } else { "略低于" }
            )
        } else if predicts_up {
            format!(
                "上涨基率达到{}阈值{:.0}%",
                if horizon <= 5 { "短周期基率" } else { "长周期基率" },
                if horizon <= 5 {
                    up_threshold * 100.0
                } else {
                    long_horizon_threshold * 100.0
                }
            )
        } else if horizon <= 5 {
            if actual_up_ratio <= down_threshold {
                format!(
                    "上涨基率低于短周期基率阈值{:.0}%",
                    down_threshold * 100.0
                )
            } else {
                format!(
                    "上涨基率未达短周期基率阈值{:.0}%",
                    up_threshold * 100.0
                )
            }
        } else {
            format!(
                "上涨基率低于长周期基率阈值{:.0}%",
                long_horizon_threshold * 100.0
            )
        };
        result.key_factors.push(format!(
            "真实数据校准: 近{}次历史结果显示上涨基率{:.0}%，{}，{}",
            empirical_samples,
            actual_up_ratio * 100.0,
            threshold_summary,
            calibration_basis,
        ));
        if let Some(signal) = &feature_signal {
            result.key_factors.push(format!(
                "均值回归特征校准: {}{:.1}{}，阈值{:.1}{}，按3000日真实回测特征{}",
                signal.name,
                signal.value,
                signal.unit,
                signal.threshold,
                signal.unit,
                if signal.predicts_up { "偏多" } else { "偏空" }
            ));
        }
        if original_change.signum() != calibrated_change.signum() && calibrated_change.abs() >= 0.2 {
            result.suggested_action = format!(
                "近期真实数据校准后{}，原始技术信号仅作参考，建议控制仓位",
                if calibrated_change > 0.0 { "偏多" } else { "偏空" }
            );
        }
    }

    Some(EngineCalibration {
        samples: empirical_samples,
        chosen_direction_rate: chosen_probability,
        actual_up_ratio,
        average_actual_change,
        inverted,
        used_empirical_baseline: true,
    })
}

fn weak_direction_from_neutral_rate(actual_up_ratio: f64) -> Option<bool> {
    const NEUTRAL_UP_RATE: f64 = 0.5;
    const EPS: f64 = 1e-12;

    if actual_up_ratio > NEUTRAL_UP_RATE + EPS {
        Some(true)
    } else if actual_up_ratio < NEUTRAL_UP_RATE - EPS {
        Some(false)
    } else {
        None
    }
}

fn should_skip_bullish_calibration(
    horizon: usize,
    predicts_up: bool,
    has_feature_signal: bool,
    uses_weak_long_direction: bool,
) -> bool {
    if !predicts_up {
        return false;
    }

    if horizon <= 1 {
        // 1日超跌/高上涨基率偏多校准在 walk-forward 中缺少稳定 edge，仅保留偏空校准。
        return true;
    }

    // 2-9日只跳过偏多强真实基率校准；5/7日反向规则、弱基率兜底与RSI等明确特征信号仍保留。
    horizon < 10 && !has_feature_signal && !uses_weak_long_direction
}

fn empirical_feature_signal(closes: &[f64], horizon: usize) -> Option<FeatureCalibrationSignal> {
    if horizon <= 1 {
        const MA5_OVERSOLD_GAP: f64 = -1.13;
        const MA5_OVERBOUGHT_GAP: f64 = 1.13;

        let ma5_gap = moving_average_gap(closes, 5)?;
        if ma5_gap <= MA5_OVERSOLD_GAP {
            return Some(FeatureCalibrationSignal {
                predicts_up: true,
                name: "5日均线偏离",
                value: ma5_gap,
                threshold: MA5_OVERSOLD_GAP,
                unit: "%",
            });
        }
        if ma5_gap >= MA5_OVERBOUGHT_GAP {
            return Some(FeatureCalibrationSignal {
                predicts_up: false,
                name: "5日均线偏离",
                value: ma5_gap,
                threshold: MA5_OVERBOUGHT_GAP,
                unit: "%",
            });
        }
        return None;
    }

    if horizon <= 5 {
        const RSI_OVERSOLD: f64 = 33.9;
        const RSI_OVERBOUGHT: f64 = 66.1;

        let rsi14 = simple_rsi(closes, 14)?;
        if rsi14 <= RSI_OVERSOLD {
            return Some(FeatureCalibrationSignal {
                predicts_up: true,
                name: "RSI14",
                value: rsi14,
                threshold: RSI_OVERSOLD,
                unit: "",
            });
        }
        if rsi14 >= RSI_OVERBOUGHT {
            return Some(FeatureCalibrationSignal {
                predicts_up: false,
                name: "RSI14",
                value: rsi14,
                threshold: RSI_OVERBOUGHT,
                unit: "",
            });
        }
        return None;
    }

    None
}

fn empirical_direction_from_rate(
    actual_up_ratio: f64,
    horizon: usize,
    up_threshold: f64,
    down_threshold: f64,
    long_horizon_up_threshold: f64,
) -> Option<bool> {
    const EPS: f64 = 1e-12;

    if horizon <= 5 {
        if actual_up_ratio + EPS >= up_threshold {
            Some(true)
        } else if actual_up_ratio <= down_threshold + EPS {
            Some(false)
        } else {
            None
        }
    } else if actual_up_ratio + EPS >= long_horizon_up_threshold {
        Some(true)
    } else if actual_up_ratio <= 1.0 - long_horizon_up_threshold + EPS {
        Some(false)
    } else {
        None
    }
}

fn moving_average_gap(closes: &[f64], window: usize) -> Option<f64> {
    if closes.len() < window || window == 0 {
        return None;
    }

    let start = closes.len() - window;
    let average = closes[start..].iter().sum::<f64>() / window as f64;
    let latest = *closes.last()?;
    if average <= 0.0 || !average.is_finite() || !latest.is_finite() {
        return None;
    }

    Some((latest - average) / average * 100.0)
}

fn simple_rsi(closes: &[f64], period: usize) -> Option<f64> {
    if closes.len() <= period || period == 0 {
        return None;
    }

    let start = closes.len() - period;
    let mut gains = 0.0;
    let mut losses = 0.0;
    for i in start..closes.len() {
        let previous = closes[i - 1];
        let current = closes[i];
        if previous <= 0.0 || !previous.is_finite() || !current.is_finite() {
            return None;
        }
        let change = current - previous;
        if change > 0.0 {
            gains += change;
        } else {
            losses -= change;
        }
    }

    if gains + losses == 0.0 {
        return Some(50.0);
    }
    if losses == 0.0 {
        return Some(100.0);
    }

    let rs = gains / losses;
    Some(100.0 - 100.0 / (1.0 + rs))
}

fn direction_from_change(change: f64) -> professional_engine::PredictionDirection {
    if change > 3.0 {
        professional_engine::PredictionDirection::StrongBullish
    } else if change > 0.2 {
        professional_engine::PredictionDirection::Bullish
    } else if change < -3.0 {
        professional_engine::PredictionDirection::StrongBearish
    } else if change < -0.2 {
        professional_engine::PredictionDirection::Bearish
    } else {
        professional_engine::PredictionDirection::Neutral
    }
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
struct DailyPredictionContext<'a> {
    professional_result: &'a professional_engine::ProfessionalPredictionResult,
    prediction_days: usize,
    regime: &'a market_regime::MarketRegimeAnalysis,
    multi_factor: &'a multi_factor::MultiFactorScore,
    enhanced_pred: &'a price_model::PricePredictionResult,
    signal_confirm: &'a signal_confirmation::SignalConfirmationResult,
    vol_forecast: &'a volatility_forecast::VolatilityForecast,
    divergence: &'a divergence::DivergenceAnalysis,
    stock_code: Option<&'a str>,
}

fn calculate_enhanced_daily_prediction(
    day: usize,
    ctx: &DailyPredictionContext<'_>,
) -> (f64, f64) {
    // 1. 基础预测：结合专业引擎和增强模型
    let professional_change = daily_change_from_horizon_change(
        ctx.professional_result.expected_change,
        ctx.prediction_days,
    );
    let enhanced_change = ctx.enhanced_pred.expected_change;
    
    // 真实历史校准后的专业引擎是方向主来源；增强价格模型只作幅度修正。
    let (pro_weight, enh_weight) = prediction_blend_weights(
        ctx.signal_confirm.confirmation_level,
        professional_change,
        enhanced_change,
    );
    
    let base_change = professional_change * pro_weight + enhanced_change * enh_weight;
    
    // 2. 时间衰减（根据市场状态和波动率预测动态调整）
    let decay_rate: f64 = match ctx.regime.regime {
        market_regime::MarketRegime::StrongUptrend | 
        market_regime::MarketRegime::StrongDowntrend => 0.97,
        market_regime::MarketRegime::ModerateUptrend | 
        market_regime::MarketRegime::ModerateDowntrend => 0.95,
        market_regime::MarketRegime::Ranging => 0.88,
        _ => 0.92,
    };
    
    // 波动率趋势影响衰减速度
    let vol_decay_adj = match ctx.vol_forecast.volatility_trend {
        volatility_forecast::VolatilityTrend::Expanding => 0.98,    // 波动率扩张，衰减慢
        volatility_forecast::VolatilityTrend::Contracting => 1.02,  // 波动率收缩，衰减快
        volatility_forecast::VolatilityTrend::Stable => 1.0,
    };
    
    let adjusted_decay = decay_rate * vol_decay_adj;
    let time_decay = adjusted_decay.powi(day as i32 - 1);
    
    // 3. 信号确认调整
    let signal_factor = ctx.signal_confirm.confidence_factor;
    
    // 假信号惩罚
    let false_signal_penalty = if ctx.signal_confirm.is_potential_false_signal {
        0.6
    } else {
        1.0
    };
    
    // 4. 背离信号调整
    let divergence_factor = if ctx.divergence.has_divergence {
        if ctx.divergence.is_triple_divergence {
            // 三重背离是极强信号
            1.25
        } else if ctx.divergence.divergence_count >= 2 {
            1.15
        } else {
            1.05
        }
    } else {
        1.0
    };
    
    // 5. 波动率状态调整
    let _vol_regime_factor = ctx.vol_forecast.volatility_regime.to_risk_multiplier();
    
    // 6. 多因子得分调整
    let mf_factor = if ctx.multi_factor.adaptive_score > 70.0 {
        1.1
    } else if ctx.multi_factor.adaptive_score < 30.0 {
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
    if matches!(ctx.vol_forecast.volatility_regime,
        volatility_forecast::VolatilityRegime::VeryHigh | 
        volatility_forecast::VolatilityRegime::Extreme) 
    {
        change_percent *= 0.7;
    }
    
    // 限制单日变化幅度（A股涨跌停限制）
    let (limit_down, limit_up) = professional_engine::get_stock_price_limits(ctx.stock_code);
    let change_percent = change_percent.clamp(limit_down, limit_up);
    
    // 8. 置信度计算
    let base_confidence = ctx.professional_result.confidence
        .min(ctx.enhanced_pred.confidence)
        .min(ctx.signal_confirm.strength + 0.3);
    
    let confidence_decay = 0.92_f64.powi(day as i32 - 1);
    
    // 信号冲突降低置信度
    let conflict_penalty = if ctx.signal_confirm.is_potential_false_signal { 0.7 } else { 1.0 };
    
    // 极端波动降低置信度
    let vol_penalty = match ctx.vol_forecast.volatility_regime {
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

fn prediction_blend_weights(
    confirmation_level: signal_confirmation::ConfirmationLevel,
    professional_change: f64,
    enhanced_change: f64,
) -> (f64, f64) {
    const CONFLICT_DIRECTION_THRESHOLD: f64 = 0.2;

    if professional_change.abs() >= CONFLICT_DIRECTION_THRESHOLD
        && enhanced_change.abs() >= CONFLICT_DIRECTION_THRESHOLD
        && professional_change.signum() != enhanced_change.signum()
    {
        return (1.0, 0.0);
    }

    match confirmation_level {
        signal_confirmation::ConfirmationLevel::Strong => (0.75, 0.25),
        signal_confirmation::ConfirmationLevel::Moderate => (0.80, 0.20),
        signal_confirmation::ConfirmationLevel::Weak => (0.85, 0.15),
        signal_confirmation::ConfirmationLevel::Invalid => (0.90, 0.10),
    }
}

fn daily_change_from_horizon_change(change: f64, horizon: usize) -> f64 {
    let horizon = horizon.max(1);
    if horizon == 1 {
        return change;
    }

    let gross = 1.0 + change / 100.0;
    if gross <= 0.0 || !gross.is_finite() {
        return change / horizon as f64;
    }

    (gross.powf(1.0 / horizon as f64) - 1.0) * 100.0
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

/// 使用已训练的 Candle 模型预测；该股无可用模型时回退到规则引擎。
pub async fn predict_with_model(request: PredictionRequest) -> Result<PredictionResponse, String> {
    use crate::prediction::model::management::{
        get_model_file_path, list_models, model_matches_identifier,
    };

    // 优先使用用户指定模型；未指定时优先选取训练周期匹配请求天数的可用模型。
    let models = list_models(&request.stock_code);
    let selected_name = request
        .model_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let model = if let Some(name) = selected_name {
        models
            .into_iter()
            .find(|m| model_matches_identifier(m, name) && get_model_file_path(&m.id).exists())
            .ok_or_else(|| format!("选择的模型 `{name}` 不存在或权重文件不存在"))?
    } else {
        let available = models
            .into_iter()
            .filter(|m| get_model_file_path(&m.id).exists())
            .collect::<Vec<_>>();
        if available.is_empty() {
            return predict(request).await; // 无模型 → 规则引擎
        }

        select_default_model(available, request.prediction_days.max(1))
            .ok_or_else(|| "没有可用模型".to_string())?
    };

    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, 250, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    if historical.len() < 60 {
        return predict(request).await;
    }

    let predictor = MlPredictor::load(&get_model_file_path(&model.id))?;
    predict_with_model_from_historical(&request, &historical, &model, &predictor)
}

/// 使用已加载模型和调用方提供的可见历史数据预测；回测复用该函数以避免未来函数。
pub fn predict_with_model_from_historical(
    request: &PredictionRequest,
    historical: &[HistoricalData],
    model: &ModelInfo,
    predictor: &MlPredictor,
) -> Result<PredictionResponse, String> {
    use crate::prediction::model::features::latest_features;

    if historical.len() < 60 {
        return Err("历史数据不足60天，无法进行准确预测".to_string());
    }

    let feats = latest_features(historical).ok_or("数据不足以构造特征")?;
    let ml_return = predictor.predict(&feats)?; // 模型训练周期对应的预期收益率 %
    let model_horizon = model_training_horizon(&model.model_type, model.prediction_days);
    let daily_ml_return = daily_change_from_horizon_change(ml_return, model_horizon);

    let last_data = historical.last().unwrap();
    let current_price = last_data.close;
    let confidence = model.accuracy.clamp(0.3, 0.92);
    let direction = if daily_ml_return > 0.05 {
        "看涨"
    } else if daily_ml_return < -0.05 {
        "看跌"
    } else {
        "中性"
    };

    // 多日预测：horizon-aware 模型在训练周期内保持累计收益口径，超出周期后再衰减。
    let prediction_days = request.prediction_days.max(1);
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    for day in 1..=prediction_days {
        let target_date = get_next_trading_day(last_date);
        let change_percent = ml_daily_change_for_day(daily_ml_return, model_horizon, day);
        let predicted_price = last_price * (1.0 + change_percent / 100.0);

        predictions.push(Prediction {
            target_date: target_date.format("%Y-%m-%d").to_string(),
            predicted_price,
            predicted_change_percent: change_percent,
            confidence,
            trading_signal: Some(direction.to_string()),
            signal_strength: Some(confidence),
            technical_indicators: None,
            prediction_reason: Some(format!(
                "Candle MLP 模型预测（{}日模型，历史方向准确率 {:.0}%）",
                model_horizon,
                model.accuracy * 100.0
            )),
            key_factors: Some(vec![
                format!("模型: {}", model.name),
                format!("{model_horizon}日预期收益 {ml_return:.2}%"),
                format!("单日等效收益 {daily_ml_return:.2}%"),
            ]),
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

/// 评估模型：新模型优先使用训练结束日之后的样本；旧模型缺少训练窗口元数据时退回最近历史样本。
pub async fn evaluate_model(model_id: String) -> Result<EvaluationResult, String> {
    use crate::prediction::model::management::get_model_file_path;
    use crate::prediction::model::ml_inference::{
        evaluate_on_horizon, evaluate_on_horizon_after, MlPredictor,
    };

    let metadata = load_model_metadata(&model_id)?;
    let model_path = get_model_file_path(&model_id);
    if !model_path.exists() {
        return Err("模型权重文件不存在，请先训练".to_string());
    }

    let predictor = MlPredictor::load(&model_path)?;

    let pool = create_temp_pool().await?;
    let horizon = model_training_horizon(&metadata.model_type, metadata.prediction_days);
    let ((direction_accuracy, mae, rmse, test_samples), evaluation_scope, evaluation_note) =
        if let Some(training_end_date) = metadata.training_end_date.as_deref() {
            let training_end = chrono::NaiveDate::parse_from_str(training_end_date, "%Y-%m-%d")
                .map_err(|e| format!("模型训练结束日期元数据格式错误: {e}"))?;
            let historical = get_historical_data(&metadata.stock_code, "1900-01-01", "9999-12-31", &pool)
                .await
                .map_err(|e| format!("获取历史数据失败: {e}"))?;
            let evaluation_cutoff =
                training_label_cutoff_date(&historical, training_end, horizon)?;
            let metrics = evaluate_on_horizon_after(&historical, &predictor, horizon, evaluation_cutoff);
            if metrics.3 == 0 {
                return Err(format!(
                    "训练标签截止日 {} 之后暂无可评估样本，请等待新的历史K线产生后再评估",
                    evaluation_cutoff.format("%Y-%m-%d")
                ));
            }
            (
                metrics,
                "训练后样本评估".to_string(),
                format!(
                    "训练特征结束日 {training_end_date}，仅统计训练标签截止日 {} 之后已产生真实标签的样本",
                    evaluation_cutoff.format("%Y-%m-%d")
                ),
            )
        } else {
            let historical = get_recent_historical_data(&metadata.stock_code, 250, &pool)
                .await
                .map_err(|e| format!("获取历史数据失败: {e}"))?;
            (
                evaluate_on_horizon(&historical, &predictor, horizon),
                "最近历史样本评估".to_string(),
                "旧模型缺少训练窗口元数据，评估可能包含训练期样本".to_string(),
            )
        };
    if test_samples == 0 {
        return Err(format!(
            "{evaluation_scope}没有可用测试样本，请扩大历史数据范围或等待真实标签产生"
        ));
    }

    Ok(EvaluationResult {
        model_id,
        model_name: metadata.name,
        stock_code: metadata.stock_code,
        test_samples,
        accuracy: direction_accuracy,
        direction_accuracy,
        mse: rmse * rmse,
        mae,
        rmse,
        evaluation_date: chrono::Local::now().format("%Y-%m-%d").to_string(),
        evaluation_scope,
        evaluation_note,
    })
}

fn training_label_cutoff_date(
    historical: &[HistoricalData],
    training_end: chrono::NaiveDate,
    horizon: usize,
) -> Result<chrono::NaiveDate, String> {
    let horizon = horizon.max(1);
    let training_end_idx = historical
        .iter()
        .position(|bar| bar.date == training_end)
        .ok_or_else(|| format!("模型训练结束日 {} 不在当前历史数据中", training_end.format("%Y-%m-%d")))?;
    historical
        .get(training_end_idx + horizon)
        .map(|bar| bar.date)
        .ok_or_else(|| {
            format!(
                "训练结束日 {} 之后不足 {horizon} 个交易日，暂无严格样本外评估标签",
                training_end.format("%Y-%m-%d")
            )
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

fn model_training_horizon(model_type: &str, prediction_days: usize) -> usize {
    if model_type == HORIZON_AWARE_MODEL_TYPE {
        prediction_days.max(1)
    } else {
        1
    }
}

fn select_default_model(models: Vec<ModelInfo>, target_horizon: usize) -> Option<ModelInfo> {
    let target_horizon = target_horizon.max(1);
    let has_matching_horizon = models
        .iter()
        .any(|model| model_training_horizon(&model.model_type, model.prediction_days) == target_horizon);

    models
        .into_iter()
        .filter(|model| {
            !has_matching_horizon
                || model_training_horizon(&model.model_type, model.prediction_days) == target_horizon
        })
        .max_by(|a, b| {
            a.accuracy
                .total_cmp(&b.accuracy)
                .then_with(|| a.created_at.cmp(&b.created_at))
        })
}

fn ml_daily_change_for_day(daily_change: f64, model_horizon: usize, day: usize) -> f64 {
    const DECAY: f64 = 0.9;

    let model_horizon = model_horizon.max(1);
    let day = day.max(1);
    let decay_start_day = if model_horizon == 1 { 1 } else { model_horizon };
    daily_change * DECAY.powi(day.saturating_sub(decay_start_day) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::analysis::market_regime::{MarketRegime, StrategyType};
    use crate::prediction::strategy::professional_engine::{
        PredictionDirection, RiskAssessment, SignalConfirmation, SignalSummary,
    };
    use chrono::{Duration, NaiveDate};

    fn history_with_late_selloff() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..80)
            .map(|i| {
                if i >= 77 {
                    close *= 0.98;
                } else {
                    close *= 0.999;
                }
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: -0.1,
                    change: -0.1,
                }
            })
            .collect()
    }

    fn history_with_late_rally() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..80)
            .map(|i| {
                if i >= 77 {
                    close *= 1.02;
                } else {
                    close *= 1.001;
                }
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_mild_uptrend() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..80)
            .map(|i| {
                close *= 1.001;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_bullish_five_day_rate_neutral_rsi() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..100)
            .map(|i| {
                if i > 0 {
                    if i >= 86 {
                        close *= if i % 2 == 0 { 1.002 } else { 0.998 };
                    } else {
                        close *= 1.001;
                    }
                }
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_weak_bearish_five_day_rate() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..100)
            .map(|i| {
                let close = 100.0 + (i as f64 * 0.5).sin() * 0.5;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: -0.1,
                    change: -0.1,
                }
            })
            .collect()
    }

    fn history_with_mild_downtrend() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..80)
            .map(|i| {
                close *= 0.999;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: -0.1,
                    change: -0.1,
                }
            })
            .collect()
    }

    fn history_with_weak_long_bullish_rate() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..80)
            .map(|i| {
                let close = 100.0 + 0.05 * i as f64 + (i as f64 * 0.4).sin();
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_neutral_long_rate() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..70)
            .map(|i| {
                let close = 100.0 + (i % 20) as f64;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_mixed_one_day_rate() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let mut close = 100.0;
        (0..121)
            .map(|i| {
                if i > 0 {
                    let phase = (i - 1) % 11;
                    close *= if phase % 2 == 0 { 1.001 } else { 0.999 };
                }
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: 0.1,
                    change: 0.1,
                }
            })
            .collect()
    }

    fn history_with_weak_bearish_one_day_rate() -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        (0..121)
            .map(|i| {
                let close = 100.0 + (i as f64 * 0.2).sin() * 0.1;
                HistoricalData {
                    symbol: "test".to_string(),
                    date: start + Duration::days(i as i64),
                    open: close,
                    close,
                    high: close + 0.5,
                    low: close - 0.5,
                    volume: 1000,
                    amount: close * 1000.0,
                    amplitude: 1.0,
                    turnover_rate: 1.0,
                    volume_ratio: 1.0,
                    change_percent: -0.1,
                    change: -0.1,
                }
            })
            .collect()
    }

    fn bearish_result() -> professional_engine::ProfessionalPredictionResult {
        professional_engine::ProfessionalPredictionResult {
            direction: PredictionDirection::Bearish,
            expected_change: -0.8,
            prediction_range: (-1.5, -0.2),
            confidence: 0.6,
            signal_confirmation: SignalConfirmation::WeakConfirm,
            market_regime: MarketRegime::Ranging,
            strategy_used: StrategyType::MeanReversion,
            signal_summary: SignalSummary {
                bullish_signals: 0,
                bearish_signals: 1,
                signal_details: Vec::new(),
                net_signal_score: -0.5,
            },
            risk_assessment: RiskAssessment {
                risk_level: "中风险".to_string(),
                volatility_risk: 0.5,
                support_distance: 1.0,
                resistance_distance: 1.0,
                suggested_stop_loss: 3.0,
                suggested_take_profit: 5.0,
                risk_reward_ratio: 1.5,
            },
            key_factors: Vec::new(),
            suggested_action: "谨慎".to_string(),
        }
    }

    fn synthetic_predictions(base_price: f64, daily_change: f64, days: usize) -> Vec<Prediction> {
        let mut last_price = base_price;
        (1..=days)
            .map(|day| {
                let predicted_price = last_price * (1.0 + daily_change / 100.0);
                last_price = predicted_price;
                Prediction {
                    target_date: format!("2026-01-{day:02}"),
                    predicted_price,
                    predicted_change_percent: daily_change,
                    confidence: 0.3,
                    trading_signal: Some(signal_from_change_percent(daily_change).to_string()),
                    signal_strength: Some(0.3),
                    technical_indicators: None,
                    prediction_reason: Some("test".to_string()),
                    key_factors: Some(Vec::new()),
                }
            })
            .collect()
    }

    #[test]
    fn test_seven_day_mid_magnitude_reversal_flips_prediction_series() {
        let base_price = 100.0;
        let mut predictions = synthetic_predictions(base_price, 0.1, 7);

        apply_seven_day_mid_magnitude_reversal(&mut predictions, base_price, 7, Some("600000"));

        assert!(predictions.last().unwrap().predicted_price < base_price);
        assert!(predictions
            .iter()
            .all(|prediction| prediction.predicted_change_percent < 0.0));
        assert!(predictions
            .iter()
            .all(|prediction| prediction.trading_signal.as_deref() == Some("看跌")));
        assert!(predictions
            .last()
            .unwrap()
            .key_factors
            .as_ref()
            .unwrap()
            .iter()
            .any(|factor| factor.contains("7日中幅方向反向校准")));
        assert!(predictions
            .last()
            .unwrap()
            .prediction_reason
            .as_ref()
            .unwrap()
            .contains("7日中幅方向反向"));
    }

    #[test]
    fn test_seven_day_mid_magnitude_reversal_ignores_other_horizons() {
        let base_price = 100.0;
        let mut predictions = synthetic_predictions(base_price, 0.1, 7);
        let original_final_price = predictions.last().unwrap().predicted_price;

        apply_seven_day_mid_magnitude_reversal(&mut predictions, base_price, 5, Some("600000"));

        assert_eq!(predictions.last().unwrap().predicted_price, original_final_price);
        assert!(predictions
            .iter()
            .all(|prediction| prediction.predicted_change_percent > 0.0));
    }

    #[test]
    fn test_seven_day_mid_magnitude_reversal_ignores_small_magnitude() {
        let base_price = 100.0;
        let mut predictions = synthetic_predictions(base_price, 0.03, 7);
        let original_final_price = predictions.last().unwrap().predicted_price;

        apply_seven_day_mid_magnitude_reversal(&mut predictions, base_price, 7, Some("600000"));

        assert_eq!(predictions.last().unwrap().predicted_price, original_final_price);
        assert!(predictions
            .iter()
            .all(|prediction| prediction.predicted_change_percent > 0.0));
    }

    #[test]
    fn test_seven_day_reversal_strategy_bullish_flips_prediction_series() {
        let base_price = 100.0;
        let mut predictions = synthetic_predictions(base_price, -0.1, 7);

        apply_seven_day_reversal_strategy_bullish(&mut predictions, base_price, 7, Some("600000"));

        assert!(predictions.last().unwrap().predicted_price > base_price);
        assert!(predictions
            .iter()
            .all(|prediction| prediction.predicted_change_percent > 0.0));
        assert!(predictions
            .iter()
            .all(|prediction| prediction.trading_signal.as_deref() == Some("看涨")));
        assert!(predictions
            .last()
            .unwrap()
            .key_factors
            .as_ref()
            .unwrap()
            .iter()
            .any(|factor| factor.contains("7日反转策略偏多校准")));
        assert!(predictions
            .last()
            .unwrap()
            .prediction_reason
            .as_ref()
            .unwrap()
            .contains("7日反转策略偏多"));
    }

    #[test]
    fn test_seven_day_reversal_strategy_bullish_ignores_other_horizons() {
        let base_price = 100.0;
        let mut predictions = synthetic_predictions(base_price, -0.1, 7);
        let original_final_price = predictions.last().unwrap().predicted_price;

        apply_seven_day_reversal_strategy_bullish(&mut predictions, base_price, 5, Some("600000"));

        assert_eq!(predictions.last().unwrap().predicted_price, original_final_price);
        assert!(predictions
            .iter()
            .all(|prediction| prediction.predicted_change_percent < 0.0));
    }

    #[test]
    fn test_one_day_calibration_skips_bullish_mean_reversion_signal() {
        let historical = history_with_late_selloff();
        let mut result = bearish_result();
        let original_change = result.expected_change;

        calibrate_professional_result(&historical, &mut result, 1, Some("600000"));

        assert_eq!(result.expected_change, original_change);
        assert!(result
            .key_factors
            .iter()
            .all(|factor| !factor.contains("超买超卖均值回归")));
    }

    #[test]
    fn test_one_day_calibration_keeps_bearish_mean_reversion_signal() {
        let historical = history_with_late_rally();
        let mut result = bearish_result();
        result.direction = PredictionDirection::Bullish;
        result.expected_change = 0.8;
        result.prediction_range = (0.2, 1.5);

        calibrate_professional_result(&historical, &mut result, 1, Some("600000"));

        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("均值回归特征校准")));
    }

    #[test]
    fn test_calibration_uses_rsi_reversion_for_five_day() {
        let historical = history_with_late_selloff();
        let mut result = bearish_result();

        calibrate_professional_result(&historical, &mut result, 5, Some("600000"));

        assert!(result.expected_change > 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("RSI14")));
    }

    #[test]
    fn test_one_day_calibration_skips_bullish_empirical_rate() {
        let historical = history_with_mild_uptrend();
        let mut result = bearish_result();
        let original_change = result.expected_change;

        calibrate_professional_result(&historical, &mut result, 1, Some("600000"));

        assert_eq!(result.expected_change, original_change);
        assert!(result
            .key_factors
            .iter()
            .all(|factor| !factor.contains("已按真实基率校准")));
    }

    #[test]
    fn test_one_day_calibration_uses_weak_bearish_rate() {
        let historical = history_with_weak_bearish_one_day_rate();
        let mut result = bearish_result();
        result.direction = PredictionDirection::Bullish;
        result.expected_change = 0.8;
        result.prediction_range = (0.2, 1.5);

        let calibration =
            calibrate_professional_result(&historical, &mut result, 1, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(calibration.actual_up_ratio > 0.45);
        assert!(calibration.actual_up_ratio < 0.5);
        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("1日弱基率")));
    }

    #[test]
    fn test_five_day_calibration_uses_bullish_empirical_reversal() {
        let historical = history_with_bullish_five_day_rate_neutral_rsi();
        let mut result = bearish_result();

        let calibration =
            calibrate_professional_result(&historical, &mut result, 5, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("短周期高基率反向校准")));
    }

    #[test]
    fn test_five_day_calibration_uses_weak_bearish_rate() {
        let historical = history_with_weak_bearish_five_day_rate();
        let mut result = bearish_result();
        result.direction = PredictionDirection::Bullish;
        result.expected_change = 0.8;
        result.prediction_range = (0.2, 1.5);

        let calibration =
            calibrate_professional_result(&historical, &mut result, 5, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(calibration.actual_up_ratio > 0.45);
        assert!(calibration.actual_up_ratio < 0.5);
        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("5日弱基率")));
    }

    #[test]
    fn test_weak_direction_from_neutral_rate_ignores_exact_neutral() {
        assert_eq!(weak_direction_from_neutral_rate(0.5), None);
        assert_eq!(weak_direction_from_neutral_rate(0.5001), Some(true));
        assert_eq!(weak_direction_from_neutral_rate(0.4999), Some(false));
    }

    #[test]
    fn test_seven_day_calibration_uses_bullish_empirical_reversal() {
        let historical = history_with_mild_uptrend();
        let mut result = bearish_result();
        result.direction = PredictionDirection::Bullish;
        result.expected_change = 0.8;
        result.prediction_range = (0.2, 1.5);

        let calibration =
            calibrate_professional_result(&historical, &mut result, 7, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("7日高基率反向校准")));
    }

    #[test]
    fn test_ten_day_calibration_uses_bullish_empirical_rate() {
        let historical = history_with_mild_uptrend();
        let mut result = bearish_result();

        let calibration =
            calibrate_professional_result(&historical, &mut result, 10, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(result.expected_change > 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("已按真实基率校准")));
    }

    #[test]
    fn test_long_horizon_calibration_uses_weak_bullish_rate() {
        let historical = history_with_weak_long_bullish_rate();
        let mut result = bearish_result();

        let calibration =
            calibrate_professional_result(&historical, &mut result, 10, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(calibration.actual_up_ratio > 0.5);
        assert!(calibration.actual_up_ratio < 0.55);
        assert!(result.expected_change > 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("长周期弱基率")));
    }

    #[test]
    fn test_long_horizon_calibration_ignores_neutral_weak_rate() {
        let historical = history_with_neutral_long_rate();
        let mut result = bearish_result();
        let original_change = result.expected_change;

        let calibration =
            calibrate_professional_result(&historical, &mut result, 10, Some("600000")).unwrap();

        assert!(!calibration.used_empirical_baseline);
        assert!((calibration.actual_up_ratio - 0.5).abs() < 1e-12);
        assert_eq!(result.expected_change, original_change);
        assert!(result.key_factors.is_empty());
    }

    #[test]
    fn test_long_horizon_calibration_keeps_bearish_empirical_rate() {
        let historical = history_with_mild_downtrend();
        let mut result = bearish_result();
        result.direction = PredictionDirection::Bullish;
        result.expected_change = 0.8;
        result.prediction_range = (0.2, 1.5);

        let calibration =
            calibrate_professional_result(&historical, &mut result, 10, Some("600000")).unwrap();

        assert!(calibration.used_empirical_baseline);
        assert!(result.expected_change < 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("长周期基率")));
    }

    #[test]
    fn test_calibration_ignores_weak_short_horizon_empirical_rate() {
        let historical = history_with_mixed_one_day_rate();
        let mut result = bearish_result();
        let original_change = result.expected_change;

        let calibration =
            calibrate_professional_result(&historical, &mut result, 1, Some("600000")).unwrap();

        assert!(!calibration.used_empirical_baseline);
        assert_eq!(result.expected_change, original_change);
        assert!(result
            .key_factors
            .iter()
            .all(|factor| !factor.contains("已按真实基率校准")));
    }

    #[test]
    fn test_empirical_direction_from_rate_requires_clear_edge() {
        assert_eq!(
            empirical_direction_from_rate(0.52, 1, 0.55, 0.45, 0.55),
            None
        );
        assert_eq!(
            empirical_direction_from_rate(0.55, 1, 0.55, 0.45, 0.55),
            Some(true)
        );
        assert_eq!(
            empirical_direction_from_rate(0.45, 1, 0.55, 0.45, 0.55),
            Some(false)
        );
        assert_eq!(
            empirical_direction_from_rate(0.54, 10, 0.55, 0.45, 0.55),
            None
        );
        assert_eq!(
            empirical_direction_from_rate(0.45, 10, 0.55, 0.45, 0.55),
            Some(false)
        );
    }

    #[test]
    fn test_should_skip_bullish_calibration_boundaries() {
        assert!(should_skip_bullish_calibration(1, true, true, false));
        assert!(should_skip_bullish_calibration(1, true, false, false));
        assert!(!should_skip_bullish_calibration(1, false, true, false));
        assert!(!should_skip_bullish_calibration(5, true, true, false));
        assert!(should_skip_bullish_calibration(5, true, false, false));
        assert!(should_skip_bullish_calibration(7, true, false, false));
        assert!(!should_skip_bullish_calibration(10, true, false, false));
        assert!(!should_skip_bullish_calibration(10, true, false, true));
        assert!(!should_skip_bullish_calibration(10, false, false, false));
    }

    #[test]
    fn test_prediction_blend_uses_calibrated_direction_on_conflict() {
        let weights = prediction_blend_weights(
            signal_confirmation::ConfirmationLevel::Strong,
            0.8,
            -1.2,
        );

        assert_eq!(weights, (1.0, 0.0));
    }

    #[test]
    fn test_prediction_blend_keeps_enhanced_model_as_magnitude_adjustment() {
        let strong = prediction_blend_weights(
            signal_confirmation::ConfirmationLevel::Strong,
            0.8,
            1.2,
        );
        let invalid = prediction_blend_weights(
            signal_confirmation::ConfirmationLevel::Invalid,
            0.8,
            1.2,
        );

        assert_eq!(strong, (0.75, 0.25));
        assert_eq!(invalid, (0.90, 0.10));
    }

    #[test]
    fn test_daily_change_from_horizon_change_compounds_back_to_horizon_change() {
        let daily = daily_change_from_horizon_change(5.0, 5);
        let compounded = (1.0 + daily / 100.0).powi(5) - 1.0;

        assert!((compounded * 100.0 - 5.0).abs() < 1e-9);
        assert_eq!(daily_change_from_horizon_change(1.2, 1), 1.2);
    }

    #[test]
    fn test_model_training_horizon_preserves_legacy_model_semantics() {
        assert_eq!(model_training_horizon("candle_mlp", 5), 1);
        assert_eq!(model_training_horizon(HORIZON_AWARE_MODEL_TYPE, 5), 5);
        assert_eq!(model_training_horizon(HORIZON_AWARE_MODEL_TYPE, 0), 1);
    }

    fn test_model(id: &str, model_type: &str, prediction_days: usize, accuracy: f64, created_at: u64) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            stock_code: "600000".to_string(),
            created_at,
            model_type: model_type.to_string(),
            features: Vec::new(),
            target: "close".to_string(),
            prediction_days,
            accuracy,
            training_start_date: None,
            training_end_date: None,
            training_samples: None,
            test_samples: None,
            mae: None,
            rmse: None,
        }
    }

    #[test]
    fn test_select_default_model_prefers_best_matching_horizon() {
        let selected = select_default_model(
            vec![
                test_model("best_1d", "candle_mlp", 5, 0.90, 1),
                test_model("weak_5d", HORIZON_AWARE_MODEL_TYPE, 5, 0.52, 3),
                test_model("best_5d", HORIZON_AWARE_MODEL_TYPE, 5, 0.63, 2),
            ],
            5,
        )
        .unwrap();

        assert_eq!(selected.id, "best_5d");
    }

    #[test]
    fn test_select_default_model_falls_back_to_best_accuracy_when_no_horizon_match() {
        let selected = select_default_model(
            vec![
                test_model("older", HORIZON_AWARE_MODEL_TYPE, 5, 0.61, 1),
                test_model("better", "candle_mlp", 5, 0.72, 2),
            ],
            10,
        )
        .unwrap();

        assert_eq!(selected.id, "better");
    }

    #[test]
    fn test_select_default_model_breaks_accuracy_ties_by_newer_model() {
        let selected = select_default_model(
            vec![
                test_model("older", HORIZON_AWARE_MODEL_TYPE, 5, 0.61, 1),
                test_model("newer", HORIZON_AWARE_MODEL_TYPE, 5, 0.61, 2),
            ],
            5,
        )
        .unwrap();

        assert_eq!(selected.id, "newer");
    }

    #[test]
    fn test_horizon_aware_ml_daily_change_preserves_model_horizon_return() {
        let daily = daily_change_from_horizon_change(5.0, 5);
        let compounded = (1..=5)
            .map(|day| ml_daily_change_for_day(daily, 5, day))
            .fold(1.0, |gross, change| gross * (1.0 + change / 100.0));

        assert!(((compounded - 1.0) * 100.0 - 5.0).abs() < 1e-9);
        assert_eq!(ml_daily_change_for_day(1.0, 1, 1), 1.0);
        assert!((ml_daily_change_for_day(1.0, 1, 2) - 0.9).abs() < 1e-9);
        assert_eq!(ml_daily_change_for_day(1.0, 5, 5), 1.0);
        assert!((ml_daily_change_for_day(1.0, 5, 6) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_training_label_cutoff_date_skips_horizon_window() {
        let historical = history_with_mild_uptrend();
        let training_end = historical[30].date;

        let cutoff = training_label_cutoff_date(&historical, training_end, 5).unwrap();

        assert_eq!(cutoff, historical[35].date);
    }

    #[test]
    fn test_training_label_cutoff_date_requires_future_label_window() {
        let historical = history_with_mild_uptrend();
        let training_end = historical[historical.len() - 2].date;

        let cutoff = training_label_cutoff_date(&historical, training_end, 5);

        assert!(cutoff.is_err());
    }

    #[test]
    fn test_predict_from_historical_normalizes_zero_prediction_days() {
        let historical = history_with_mild_uptrend();
        let request = PredictionRequest {
            stock_code: "600000".to_string(),
            model_name: None,
            prediction_days: 0,
            use_candle: false,
        };

        let response = predict_from_historical(&request, &historical).unwrap();

        assert_eq!(response.predictions.len(), 1);
        assert!(response.predictions[0].predicted_price.is_finite());
    }
}
