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
use crate::db::{connection::create_temp_pool, models::HistoricalData, repository::get_recent_historical_data};

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
            prediction_days: request.prediction_days,
            stock_code: Some(&request.stock_code),
        },
    );
    let mut professional_result = analysis.professional_result.clone();
    // 预测列表逐行展示的是单个交易日增量；多日 horizon 回测会单独使用 horizon 口径校准。
    calibrate_professional_result(
        &historical,
        &mut professional_result,
        1,
        Some(&request.stock_code),
    );

    // =========================================================================
    // 第十一阶段：生成预测序列
    // =========================================================================
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    
    for day in 1..=request.prediction_days {
        let target_date = get_next_trading_day(last_date);
        
        // 使用增强模型计算预测
        let daily_ctx = DailyPredictionContext {
            professional_result: &professional_result,
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
            trading_signal: Some(professional_result.direction.to_string()),
            signal_strength: Some(professional_result.confidence),
            technical_indicators: Some(convert_indicators(&analysis.tech_indicators)),
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
    let up_threshold = if horizon <= 1 { 0.55 } else { 0.50 };
    let feature_signal = empirical_feature_signal(&closes, horizon);
    let predicts_up = feature_signal
        .as_ref()
        .map(|signal| signal.predicts_up)
        .unwrap_or(actual_up_ratio >= up_threshold);
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
            "已按超跌均值回归特征校准"
        } else {
            "已按真实基率校准"
        };
        result.key_factors.push(format!(
            "真实数据校准: 近{}次历史结果显示上涨基率{:.0}%，{}阈值{:.0}%，{}",
            empirical_samples,
            actual_up_ratio * 100.0,
            if predicts_up {
                "达到"
            } else {
                "未达"
            },
            up_threshold * 100.0,
            calibration_basis,
        ));
        if let Some(signal) = &feature_signal {
            result.key_factors.push(format!(
                "超跌均值回归校准: {}{:.1}{}，阈值{:.1}{}，按3000日真实回测特征{}",
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

fn empirical_feature_signal(closes: &[f64], horizon: usize) -> Option<FeatureCalibrationSignal> {
    if horizon <= 1 {
        let ma5_gap = moving_average_gap(closes, 5)?;
        return Some(FeatureCalibrationSignal {
            predicts_up: ma5_gap <= -1.13,
            name: "5日均线偏离",
            value: ma5_gap,
            threshold: -1.13,
            unit: "%",
        });
    }

    if horizon <= 5 {
        let rsi14 = simple_rsi(closes, 14)?;
        return Some(FeatureCalibrationSignal {
            predicts_up: rsi14 <= 33.9,
            name: "RSI14",
            value: rsi14,
            threshold: 33.9,
            unit: "",
        });
    }

    None
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
    let professional_change = ctx.professional_result.expected_change;
    let enhanced_change = ctx.enhanced_pred.expected_change;
    
    // 根据信号确认程度调整两个模型的权重
    let (pro_weight, enh_weight) = match ctx.signal_confirm.confirmation_level {
        signal_confirmation::ConfirmationLevel::Strong => (0.55, 0.45),
        signal_confirmation::ConfirmationLevel::Moderate => (0.50, 0.50),
        signal_confirmation::ConfirmationLevel::Weak => (0.45, 0.55),
        signal_confirmation::ConfirmationLevel::Invalid => (0.40, 0.60),
    };
    
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
    use crate::prediction::model::features::latest_features;
    use crate::prediction::model::management::{get_model_file_path, list_models};
    use crate::prediction::model::ml_inference::MlPredictor;

    // 优先使用用户指定模型；未指定时选取该股票最新且权重文件存在的模型。
    let models = list_models(&request.stock_code);
    let selected_name = request
        .model_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let model = if let Some(name) = selected_name {
        models
            .into_iter()
            .find(|m| m.name == name && get_model_file_path(&m.id).exists())
            .ok_or_else(|| format!("选择的模型 `{name}` 不存在或权重文件不存在"))?
    } else {
        match models
            .into_iter()
            .find(|m| get_model_file_path(&m.id).exists())
        {
            Some(m) => m,
            None => return predict(request).await, // 无模型 → 规则引擎
        }
    };

    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, 250, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    if historical.len() < 60 {
        return predict(request).await;
    }

    let predictor = MlPredictor::load(&get_model_file_path(&model.id))?;
    let feats = latest_features(&historical).ok_or("数据不足以构造特征")?;
    let ml_return = predictor.predict(&feats)?; // 次日预期收益率 %

    let last_data = historical.last().unwrap();
    let current_price = last_data.close;
    let confidence = model.accuracy.clamp(0.3, 0.92);
    let direction = if ml_return > 0.05 {
        "看涨"
    } else if ml_return < -0.05 {
        "看跌"
    } else {
        "中性"
    };

    // 多日预测：以模型次日收益为基准，按日衰减
    let decay = 0.9_f64;
    let mut predictions = Vec::new();
    let mut last_date = last_data.date;
    let mut last_price = current_price;
    for day in 1..=request.prediction_days {
        let target_date = get_next_trading_day(last_date);
        let change_percent = ml_return * decay.powi(day as i32 - 1);
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
                "Candle MLP 模型预测（历史方向准确率 {:.0}%）",
                model.accuracy * 100.0
            )),
            key_factors: Some(vec![
                format!("模型: {}", model.name),
                format!("次日预期收益 {ml_return:.2}%"),
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

/// 评估模型：加载已训练权重，在最近历史数据上计算真实指标
pub async fn evaluate_model(model_id: String) -> Result<EvaluationResult, String> {
    use crate::prediction::model::management::get_model_file_path;
    use crate::prediction::model::ml_inference::{evaluate_on, MlPredictor};

    let metadata = load_model_metadata(&model_id)?;
    let model_path = get_model_file_path(&model_id);
    if !model_path.exists() {
        return Err("模型权重文件不存在，请先训练".to_string());
    }

    let predictor = MlPredictor::load(&model_path)?;

    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&metadata.stock_code, 250, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    let (direction_accuracy, mae, rmse, test_samples) = evaluate_on(&historical, &predictor);

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

    #[test]
    fn test_calibration_uses_mean_reversion_signal_for_one_day() {
        let historical = history_with_late_selloff();
        let mut result = bearish_result();

        calibrate_professional_result(&historical, &mut result, 1, Some("600000"));

        assert!(result.expected_change > 0.0);
        assert!(result
            .key_factors
            .iter()
            .any(|factor| factor.contains("超跌均值回归校准")));
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
}
