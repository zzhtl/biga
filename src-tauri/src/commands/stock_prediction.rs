//! 股票预测命令模块
//! 
//! 提供前端调用的预测相关命令

use crate::prediction::{
    types::*,
    model::{training, inference, management},
    strategy::multi_timeframe::{self, MultiTimeframeSignal},
    analysis::*,
};
use crate::db::{connection::create_temp_pool, repository::{get_historical_data, get_recent_historical_data, get_recent_historical_data_for_symbols, get_symbols_with_min_bars}};
use crate::services;
use chrono::NaiveDate;
use sqlx::sqlite::SqlitePool;

// =============================================================================
// 模型管理命令
// =============================================================================

/// 列出所有股票预测模型
#[tauri::command]
pub async fn list_stock_prediction_models(symbol: String) -> Result<Vec<ModelInfo>, String> {
    Ok(management::list_available_models(&symbol))
}

/// 删除股票预测模型
#[tauri::command]
pub async fn delete_stock_prediction_model(model_id: String) -> Result<(), String> {
    management::delete_model(&model_id)
}

// =============================================================================
// 训练命令
// =============================================================================

/// 训练股票预测模型
#[tauri::command]
pub async fn train_stock_prediction_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    training::train_model(request).await
}

/// 使用 Candle 训练模型
#[tauri::command]
pub async fn train_candle_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    training::train_model(request).await
}

/// 重新训练模型
#[tauri::command]
pub async fn retrain_candle_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    training::retrain_model(model_id, epochs, batch_size, learning_rate).await
}

// =============================================================================
// 预测命令
// =============================================================================

/// 股票价格预测
#[tauri::command]
pub async fn predict_stock_price(request: PredictionRequest) -> Result<PredictionResponse, String> {
    inference::predict(request).await
}

/// 使用 Candle 进行预测（有已训练模型时走 ML，否则回退规则引擎）
#[tauri::command]
pub async fn predict_with_candle(request: PredictionRequest) -> Result<PredictionResponse, String> {
    inference::predict_with_model(request).await
}

/// 简化策略预测
#[tauri::command]
pub async fn predict_candle_price_simple(request: PredictionRequest) -> Result<PredictionResponse, String> {
    inference::predict_simple(request).await
}

// =============================================================================
// 评估与回测命令
// =============================================================================

/// 评估模型
#[tauri::command]
pub async fn evaluate_candle_model(model_id: String) -> Result<EvaluationResult, String> {
    inference::evaluate_model(model_id).await
}

/// 执行回测（真实 walk-forward：逐日仅用历史数据预测并与未来真实涨跌对比）
#[tauri::command]
pub async fn run_model_backtest(request: BacktestRequest) -> Result<BacktestReport, String> {
    use crate::prediction::backtest::{
        run_backtest_window, run_backtest_window_with_predictor, MIN_LOOKBACK,
    };
    use crate::prediction::model::ml_inference::MlPredictor;

    let pool = create_temp_pool().await?;
    let start_date = NaiveDate::parse_from_str(&request.start_date, "%Y-%m-%d")
        .map_err(|e| format!("回测开始日期格式错误: {e}"))?;
    let end_date = NaiveDate::parse_from_str(&request.end_date, "%Y-%m-%d")
        .map_err(|e| format!("回测结束日期格式错误: {e}"))?;
    if end_date < start_date {
        return Err("回测结束日期不能早于开始日期".to_string());
    }

    // 结束日期限制的是预测发起日；仍需查询之后的真实K线来评估 horizon 收益。
    let historical = get_historical_data(&request.stock_code, "1900-01-01", "9999-12-31", &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    if historical.is_empty() {
        return Err("未找到历史数据".to_string());
    }

    let selected_model_name = request
        .model_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    let loaded_model = if let Some(name) = selected_model_name {
        let model = management::list_models(&request.stock_code)
            .into_iter()
            .find(|model| {
                management::model_matches_identifier(model, name)
                    && management::get_model_file_path(&model.id).exists()
            })
            .ok_or_else(|| format!("选择的模型 `{name}` 不存在或权重文件不存在"))?;
        let predictor = MlPredictor::load(&management::get_model_file_path(&model.id))?;
        Some((model, predictor))
    } else {
        None
    };

    let horizon = request.prediction_days.max(1);
    let report = if let Some((model, predictor)) = loaded_model.as_ref() {
        run_backtest_window_with_predictor(
            &request.stock_code,
            &historical,
            MIN_LOOKBACK,
            horizon,
            request.backtest_interval,
            Some(start_date),
            Some(end_date),
            |prediction_request, visible_history| {
                inference::predict_with_model_from_historical(
                    prediction_request,
                    visible_history,
                    model,
                    predictor,
                )
            },
        )?
    } else {
        run_backtest_window(
            &request.stock_code,
            &historical,
            MIN_LOOKBACK,
            horizon,
            request.backtest_interval,
            Some(start_date),
            Some(end_date),
        )?
    };
    let m = &report.metrics;
    if m.total == 0 {
        return Err("指定日期范围内没有可回测样本，请扩大区间或确认历史数据覆盖范围".to_string());
    }

    // 价格准确率：由平均绝对误差换算的有界评分（误差 10 个百分点对应 0 分）
    let price_accuracy = (1.0 - m.mean_abs_error / 10.0).clamp(0.0, 1.0);
    let report_model_name = loaded_model
        .as_ref()
        .map(|(model, _)| {
            let training_days = if model.model_type == crate::prediction::model::HORIZON_AWARE_MODEL_TYPE {
                model.prediction_days.max(1)
            } else {
                1
            };
            format!("{}（{}日Candle模型）", model.name, training_days)
        })
        .unwrap_or_else(|| "规则引擎+真实数据校准".to_string());
    let prediction_reason = if loaded_model.is_some() {
        "固定权重模型走步回测：每次预测输入仅使用预测日前历史数据"
    } else {
        "规则引擎走步回测：仅使用预测日前历史数据"
    };
    let backtest_entries: Vec<BacktestEntry> = report
        .observations
        .iter()
        .map(|observation| {
            backtest_entry_from_observation(observation, prediction_reason, &report_model_name)
        })
        .collect();
    let accuracy_trend = backtest_entries
        .iter()
        .map(|entry| entry.direction_accuracy)
        .collect();
    let daily_accuracy = backtest_entries
        .iter()
        .map(|entry| DailyAccuracy {
            date: entry.prediction_date.clone(),
            price_accuracy: entry.price_accuracy,
            direction_accuracy: entry.direction_accuracy,
            prediction_count: entry.predictions.len(),
            market_volatility: entry.actual_changes.first().copied().unwrap_or(0.0).abs(),
        })
        .collect();
    let price_error_distribution = backtest_entries
        .iter()
        .map(|entry| entry.avg_prediction_error)
        .collect();
    let volatility_vs_accuracy = backtest_entries
        .iter()
        .map(|entry| {
            (
                entry.actual_changes.first().copied().unwrap_or(0.0).abs(),
                entry.direction_accuracy,
            )
        })
        .collect();
    Ok(BacktestReport {
        stock_code: request.stock_code,
        model_name: report_model_name,
        backtest_period: format!("{} 至 {}", request.start_date, request.end_date),
        total_predictions: m.total,
        backtest_entries,
        overall_price_accuracy: price_accuracy,
        overall_direction_accuracy: m.direction_accuracy,
        average_prediction_error: m.mean_abs_error,
        accuracy_trend,
        daily_accuracy,
        price_error_distribution,
        direction_correct_rate: m.direction_accuracy,
        volatility_vs_accuracy,
    })
}

fn backtest_entry_from_observation(
    observation: &crate::prediction::backtest::BacktestObservation,
    prediction_reason: &str,
    model_name: &str,
) -> BacktestEntry {
    let error_percent = (observation.predicted_change - observation.actual_change).abs();
    let price_accuracy = (1.0 - error_percent / 10.0).clamp(0.0, 1.0);
    let direction_accuracy = if same_direction(observation.predicted_change, observation.actual_change) {
        1.0
    } else {
        0.0
    };

    BacktestEntry {
        prediction_date: observation.prediction_date.format("%Y-%m-%d").to_string(),
        predictions: vec![Prediction {
            target_date: observation.target_date.format("%Y-%m-%d").to_string(),
            predicted_price: observation.predicted_price,
            predicted_change_percent: observation.predicted_change,
            confidence: observation.confidence,
            trading_signal: Some(signal_from_change(observation.predicted_change).to_string()),
            signal_strength: Some(observation.confidence),
            technical_indicators: None,
            prediction_reason: Some(prediction_reason.to_string()),
            key_factors: Some(vec![
                format!("回测对象: {model_name}"),
                format!("预测发起日: {}", observation.prediction_date.format("%Y-%m-%d")),
                format!("基准价格: {:.2}", observation.base_price),
                format!("实际涨跌幅: {:+.2}%", observation.actual_change),
            ]),
            interval: None,
        }],
        actual_prices: vec![observation.actual_price],
        actual_changes: vec![observation.actual_change],
        price_accuracy,
        direction_accuracy,
        avg_prediction_error: error_percent,
    }
}

fn same_direction(predicted_change: f64, actual_change: f64) -> bool {
    (predicted_change > 0.0 && actual_change > 0.0)
        || (predicted_change < 0.0 && actual_change < 0.0)
}

fn signal_from_change(change: f64) -> &'static str {
    if change > 0.0 {
        "看涨"
    } else if change < 0.0 {
        "看跌"
    } else {
        "中性"
    }
}

// =============================================================================
// 截面相对强弱排名（市场中性多因子）
// =============================================================================

/// 滚动截面多因子相对强弱排名（限可投资的流动大中盘域）。
///
/// ⚠️ 这是**弱相对强弱描述指标**，不是经证实的收益预测器：无前视走步评估显示该技术
/// 截面信号样本外不稳定（t<2、对票池构成敏感、在小盘上反向），详见 .claude/CLAUDE.md。
/// 仅限流通市值 ≥ 200 亿的大中盘排名——小盘上信号反向且不可交易。
#[tauri::command]
pub async fn cross_sectional_ranking() -> Result<Vec<crate::prediction::cross_section::RankedStock>, String> {
    use crate::db::repository::get_symbols_with_min_bars_and_cap;
    use crate::prediction::cross_section::rank_latest;

    let pool = create_temp_pool().await?;
    // 历史 ≥300 根（FACTOR_LOOKBACK+horizon+window≈285，留余量）且流通市值 ≥200 亿。
    let symbols = get_symbols_with_min_bars_and_cap(300, 200.0e8, &pool)
        .await
        .map_err(|e| format!("获取股票列表失败: {e}"))?;
    if symbols.len() < 5 {
        return Err("满足流动域门槛（≥300根且市值≥200亿）的股票不足 5 只，无法做截面排名".to_string());
    }

    let stocks = get_recent_historical_data_for_symbols(&symbols, 800, &pool)
        .await
        .map_err(|e| format!("获取截面历史数据失败: {e}"))?
        .into_iter()
        .filter(|(_, hist)| hist.len() >= 300)
        .collect::<Vec<_>>();

    // 持有期 15 日 + IC 估计窗口 250 日（降换手、稳权重；非收益保证）。
    let ranking = rank_latest(&stocks, 15, 250);
    if ranking.is_empty() {
        return Err("数据不足以生成截面排名".to_string());
    }
    Ok(ranking)
}

// =============================================================================
// 估值上下文命令（PE/PB + 最新基本面，供预测页参考展示）
// =============================================================================

/// 单只股票的估值/质量/成长画像——**仅作参考展示，非收益预测**。
/// PE/PB/市值来自 stock_capital（ssjy），基本面来自 stock_fundamentals 最新报告期（cwzb）。
/// 未刷新或无数据的字段以 `None` 返回，前端显示占位符而非 0。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValuationContext {
    pub symbol: String,
    pub pe: Option<f64>,
    pub pb: Option<f64>,
    /// 流通市值（亿元）
    pub circulating_market_cap_yi: Option<f64>,
    pub report_date: Option<String>,
    pub roe: Option<f64>,
    pub eps: Option<f64>,
    pub bps: Option<f64>,
    pub revenue_growth: Option<f64>,
    pub profit_growth: Option<f64>,
}

/// 获取单只股票估值上下文（PE/PB + 最新基本面）。数据随"刷新"按钮统一更新。
#[tauri::command]
pub async fn get_valuation_context(symbol: String) -> Result<ValuationContext, String> {
    use crate::db::repository::{get_stock_capital, get_stock_fundamentals};

    let pool = create_temp_pool().await?;
    let cap = get_stock_capital(&symbol, &pool)
        .await
        .map_err(|e| format!("获取股本估值失败: {e}"))?;
    let funds = get_stock_fundamentals(&symbol, &pool)
        .await
        .map_err(|e| format!("获取基本面失败: {e}"))?;
    // 列表按报告期升序，最后一个为最新一期
    let latest = funds.last();

    // 0 / 非有限值视为"未刷新/无数据"，返回 None
    let pos = |v: f64| (v.is_finite() && v != 0.0).then_some(v);

    Ok(ValuationContext {
        symbol,
        pe: cap.as_ref().and_then(|c| pos(c.pe)),
        pb: cap.as_ref().and_then(|c| pos(c.pb)),
        circulating_market_cap_yi: cap
            .as_ref()
            .and_then(|c| pos(c.circulating_market_cap).map(|v| v / 1.0e8)),
        report_date: latest.map(|f| f.report_date.clone()),
        roe: latest.and_then(|f| f.roe),
        eps: latest.and_then(|f| f.eps),
        bps: latest.and_then(|f| f.bps),
        revenue_growth: latest.and_then(|f| f.revenue_growth),
        profit_growth: latest.and_then(|f| f.profit_growth),
    })
}

// =============================================================================
// 优化建议命令
// =============================================================================

/// 优化建议结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationSuggestions {
    pub stock_code: String,
    pub model_name: String,
    pub suggestions: Vec<String>,
    pub expected_improvement: f64,
}

/// 获取优化建议
#[tauri::command]
pub async fn get_optimization_suggestions(
    stock_code: String,
    model_name: String,
    backtest_report: BacktestReport,
) -> Result<OptimizationSuggestions, String> {
    let mut suggestions = Vec::new();
    let mut expected_improvement = 0.0;
    
    if backtest_report.overall_price_accuracy < 0.6 {
        suggestions.push("增加技术指标特征：添加ATR、Williams%R、ROC等指标".to_string());
        expected_improvement += 0.08;
    }
    
    if backtest_report.overall_direction_accuracy < 0.7 {
        suggestions.push("优化趋势识别：实现多时间框架均线系统".to_string());
        expected_improvement += 0.12;
    }
    
    if suggestions.is_empty() {
        suggestions.push("当前模型表现良好，建议继续观察".to_string());
    }
    
    Ok(OptimizationSuggestions {
        stock_code,
        model_name,
        suggestions,
        expected_improvement,
    })
}

// =============================================================================
// 多周期分析命令
// =============================================================================

/// 获取多周期信号
#[tauri::command]
pub async fn get_multi_timeframe_signals(symbol: String) -> Result<Vec<MultiTimeframeSignal>, String> {
    services::prediction::get_multi_timeframe_signals(symbol).await
}

/// 获取最新多周期信号
#[tauri::command]
pub async fn get_latest_multi_timeframe_signal(symbol: String) -> Result<Option<MultiTimeframeSignal>, String> {
    services::prediction::get_latest_multi_timeframe_signal(symbol).await
}

/// 分析多周期预测价值
#[tauri::command]
pub async fn analyze_multi_timeframe_prediction_value(symbol: String) -> Result<std::collections::HashMap<String, f64>, String> {
    let signals = services::prediction::get_multi_timeframe_signals(symbol).await?;
    
    let mut analysis = std::collections::HashMap::new();
    
    // 计算信号统计
    let total = signals.len() as f64;
    let buy_count = signals.iter().filter(|s| s.buy_signal).count() as f64;
    let sell_count = signals.iter().filter(|s| s.sell_signal).count() as f64;
    let avg_quality = signals.iter().map(|s| s.signal_quality).sum::<f64>() / total.max(1.0);
    
    analysis.insert("total_signals".to_string(), total);
    analysis.insert("buy_signals".to_string(), buy_count);
    analysis.insert("sell_signals".to_string(), sell_count);
    analysis.insert("buy_ratio".to_string(), buy_count / total.max(1.0));
    analysis.insert("avg_quality".to_string(), avg_quality);
    
    Ok(analysis)
}

// =============================================================================
// 专业预测命令
// =============================================================================

/// 专业策略预测
#[tauri::command]
pub async fn predict_with_professional_strategy(request: PredictionRequest) -> Result<ProfessionalPredictionResponse, String> {
    predict_with_professional_strategy_inner(request, None).await
}

async fn predict_with_professional_strategy_inner(
    request: PredictionRequest,
    history_days: Option<usize>,
) -> Result<ProfessionalPredictionResponse, String> {
    let analysis_days = history_days
        .unwrap_or(inference::MAX_ANALYSIS_DAYS)
        .clamp(inference::MIN_ANALYSIS_DAYS, inference::MAX_ANALYSIS_DAYS);

    let mut predictions = if request.use_candle {
        inference::predict_with_model(request.clone()).await?
    } else {
        inference::predict_with_history(request.clone(), analysis_days).await?
    };

    // 获取历史数据进行专业分析
    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&request.stock_code, analysis_days, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;
    
    if historical.len() < 60 {
        return Err("历史数据不足60天，无法进行准确预测".to_string());
    }
    
    let prices: Vec<f64> = historical.iter().map(|h| h.close).collect();
    let highs: Vec<f64> = historical.iter().map(|h| h.high).collect();
    let lows: Vec<f64> = historical.iter().map(|h| h.low).collect();
    let volumes: Vec<i64> = historical.iter().map(|h| h.volume).collect();
    let opens: Vec<f64> = historical.iter().map(|h| h.open).collect();
    
    let current_price = *prices.last().unwrap();
    let last_data = historical.last().unwrap();
    
    let prediction_days = request.prediction_days.max(1);
    let analysis = inference::analyze(
        &prices,
        &highs,
        &lows,
        &volumes,
        &opens,
        inference::AnalysisOptions {
            turnover_rate: last_data.turnover_rate,
            prediction_days,
            stock_code: Some(&request.stock_code),
        },
    );
    let mut professional_result = analysis.professional_result.clone();
    inference::calibrate_professional_result(
        &historical,
        &mut professional_result,
        prediction_days,
        Some(&request.stock_code),
    );
    if let Some(adjustment) =
        latest_cross_section_adjustment(&request.stock_code, prediction_days, &pool).await?
    {
        // 截面排名已降级为"相对强弱描述指标"（见 .claude/CLAUDE.md 结论2，非收益保证），
        // 仅作描述性提示附到 key_factors，不再注入点预测涨跌幅。
        append_prediction_factor(&mut predictions, &adjustment.summary);
        professional_result.key_factors.push(adjustment.summary);
    }
    let risk = &professional_result.risk_assessment;
    
    // 生成买卖点
    let mut buy_points = Vec::new();
    let mut sell_points = Vec::new();
    
    // 根据分析结果生成买点
    if professional_result.direction.to_bias() > 0.0 || analysis.patterns.iter().any(|p| p.is_bullish) {
        let price_level = analysis
            .support_resistance
            .support_levels
            .first()
            .copied()
            .unwrap_or(current_price);
        let stop_loss = price_level * (1.0 - risk.suggested_stop_loss / 100.0);
        let take_profit = vec![
            price_level * (1.0 + risk.suggested_take_profit / 200.0),
            price_level * (1.0 + risk.suggested_take_profit / 100.0),
        ];
        
        buy_points.push(BuySellPoint {
            point_type: "买入".to_string(),
            signal_strength: professional_result.confidence,
            price_level,
            stop_loss,
            take_profit,
            risk_reward_ratio: risk.risk_reward_ratio,
            reasons: vec![
                format!("专业方向: {}", professional_result.direction.to_string()),
                format!("量价信号: {}", analysis.volume_signal.signal),
                format!("策略建议: {}", professional_result.suggested_action),
            ],
            confidence: professional_result.confidence,
        });
    }
    
    // 根据分析结果生成卖点
    if professional_result.direction.to_bias() < 0.0 || analysis.patterns.iter().any(|p| !p.is_bullish) {
        let price_level = analysis
            .support_resistance
            .resistance_levels
            .first()
            .copied()
            .unwrap_or(current_price);
        let stop_loss = price_level * (1.0 + risk.suggested_stop_loss / 100.0);
        let take_profit = vec![
            price_level * (1.0 - risk.suggested_take_profit / 200.0),
            price_level * (1.0 - risk.suggested_take_profit / 100.0),
        ];
        
        sell_points.push(BuySellPoint {
            point_type: "卖出".to_string(),
            signal_strength: professional_result.confidence,
            price_level,
            stop_loss,
            take_profit,
            risk_reward_ratio: risk.risk_reward_ratio,
            reasons: vec![
                format!("专业方向: {}", professional_result.direction.to_string()),
                format!("量价信号: {}", analysis.volume_signal.signal),
                format!("策略建议: {}", professional_result.suggested_action),
            ],
            confidence: professional_result.confidence,
        });
    }

    let date = last_data.date.format("%Y-%m-%d").to_string();
    let multi_timeframe = multi_timeframe::get_latest_signal(&prices, &highs, &lows, &date)
        .unwrap_or_else(|| neutral_multi_timeframe_signal(&date));
    
    let professional_analysis = ProfessionalPrediction {
        buy_points,
        sell_points,
        support_resistance: analysis.support_resistance,
        multi_timeframe,
        divergence: summarize_divergence(&analysis.divergence_analysis),
        current_advice: professional_result.suggested_action.clone(),
        risk_level: risk.risk_level.clone(),
        candle_patterns: analysis.patterns,
        volume_analysis: summarize_volume(&analysis.volume_signal, analysis.tech_indicators.obv_trend),
        multi_factor_score: analysis.multi_factor_score,
    };
    
    Ok(ProfessionalPredictionResponse {
        predictions,
        professional_analysis,
    })
}

/// 纯技术分析预测
#[tauri::command]
pub async fn predict_with_technical_only(request: TechnicalOnlyRequest) -> Result<ProfessionalPredictionResponse, String> {
    let pred_request = PredictionRequest {
        stock_code: request.stock_code.clone(),
        model_name: None,
        prediction_days: request.prediction_days,
        use_candle: false,
    };
    
    predict_with_professional_strategy_inner(pred_request, request.history_days).await
}

struct CrossSectionAdjustment {
    summary: String,
}

async fn latest_cross_section_adjustment(
    stock_code: &str,
    prediction_days: usize,
    pool: &SqlitePool,
) -> Result<Option<CrossSectionAdjustment>, String> {
    use crate::prediction::cross_section::{daily_bias_from_rank, rank_latest};

    let horizon = prediction_days.max(1);
    if horizon != 5 {
        return Ok(None);
    }

    let symbols = get_symbols_with_min_bars(150, pool)
        .await
        .map_err(|e| format!("获取截面股票池失败: {e}"))?;
    if symbols.len() < 20 {
        return Ok(None);
    }

    let stocks = get_recent_historical_data_for_symbols(&symbols, 800, pool)
        .await
        .map_err(|e| format!("获取截面历史数据失败: {e}"))?
        .into_iter()
        .filter(|(_, hist)| hist.len() >= 150)
        .collect::<Vec<_>>();
    let ranking = rank_latest(&stocks, horizon, 120);
    if ranking.len() < 20 {
        return Ok(None);
    }

    let target_digits = symbol_digits(stock_code);
    let ranked = ranking.iter().find(|ranked| {
        ranked.symbol.eq_ignore_ascii_case(stock_code)
            || (!target_digits.is_empty() && symbol_digits(&ranked.symbol) == target_digits)
    });
    let Some(ranked) = ranked else {
        return Ok(None);
    };

    let total = ranking.len();
    let percentile = if total > 1 {
        (ranked.rank - 1) as f64 / (total - 1) as f64
    } else {
        0.5
    };
    let daily_bias = daily_bias_from_rank(ranked.rank, total);
    if daily_bias == 0.0 {
        return Ok(None);
    }

    let relative_strength = (1.0 - percentile) * 100.0;
    Ok(Some(CrossSectionAdjustment {
        summary: format!(
            "截面强弱参考: 全市场第{}/{}名，相对强度{:.0}%（仅相对强弱描述，非收益预测）",
            ranked.rank, total, relative_strength
        ),
    }))
}

fn append_prediction_factor(predictions: &mut PredictionResponse, summary: &str) {
    for prediction in predictions.predictions.iter_mut() {
        prediction
            .key_factors
            .get_or_insert_with(Vec::new)
            .push(summary.to_string());
    }
}

fn symbol_digits(symbol: &str) -> String {
    symbol.chars().filter(|c| c.is_ascii_digit()).collect()
}

fn neutral_multi_timeframe_signal(date: &str) -> MultiTimeframeSignal {
    MultiTimeframeSignal {
        date: date.to_string(),
        daily_trend: "中性".to_string(),
        weekly_trend: "中性".to_string(),
        monthly_trend: "中性".to_string(),
        resonance_level: 0,
        resonance_direction: "中性".to_string(),
        signal_quality: 30.0,
        buy_signal: false,
        sell_signal: false,
    }
}

fn summarize_divergence(analysis: &DivergenceAnalysis) -> VolumePriceDivergence {
    let signals = [
        &analysis.rsi_divergence,
        &analysis.macd_divergence,
        &analysis.obv_divergence,
        &analysis.williams_divergence,
        &analysis.roc_divergence,
    ];
    let has_bullish_divergence = signals
        .iter()
        .filter_map(|signal| signal.as_ref())
        .any(|signal| signal.divergence_type.is_bullish());
    let has_bearish_divergence = signals
        .iter()
        .filter_map(|signal| signal.as_ref())
        .any(|signal| !signal.divergence_type.is_bullish());

    VolumePriceDivergence {
        has_bullish_divergence,
        has_bearish_divergence,
        divergence_strength: analysis.overall_confidence,
        warning_message: analysis.suggested_action.clone(),
    }
}

fn summarize_volume(signal: &VolumePriceSignal, obv_trend: f64) -> VolumeAnalysisInfo {
    let volume_price_sync = matches!(signal.direction.as_str(), "上涨" | "下跌")
        && signal.volume_trend.contains("放量");
    let accumulation_signal = match signal.direction.as_str() {
        "上涨" => signal.confidence,
        "下跌" => -signal.confidence,
        _ => 0.0,
    };
    let obv_trend = if obv_trend > 0.05 {
        "上升"
    } else if obv_trend < -0.05 {
        "下降"
    } else {
        "平稳"
    };

    VolumeAnalysisInfo {
        volume_trend: signal.volume_trend.clone(),
        volume_price_sync,
        accumulation_signal,
        obv_trend: obv_trend.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_digits_normalizes_market_suffix() {
        assert_eq!(symbol_digits("000001.SZ"), "000001");
        assert_eq!(symbol_digits("sh600000"), "600000");
    }

    #[test]
    fn test_append_prediction_factor_adds_context() {
        let mut response = PredictionResponse {
            predictions: vec![
                Prediction {
                    target_date: "2026-01-02".to_string(),
                    predicted_price: 10.0,
                    predicted_change_percent: 1.0,
                    confidence: 0.6,
                    trading_signal: Some("看涨".to_string()),
                    signal_strength: Some(0.6),
                    technical_indicators: None,
                    prediction_reason: None,
                    key_factors: None,
                    interval: None,
                },
                Prediction {
                    target_date: "2026-01-05".to_string(),
                    predicted_price: 10.0,
                    predicted_change_percent: 1.0,
                    confidence: 0.6,
                    trading_signal: Some("看涨".to_string()),
                    signal_strength: Some(0.6),
                    technical_indicators: None,
                    prediction_reason: None,
                    key_factors: None,
                    interval: None,
                },
            ],
            last_real_data: Some(LastRealData {
                date: "2026-01-01".to_string(),
                price: 10.0,
                change_percent: 0.0,
            }),
        };

        append_prediction_factor(&mut response, "截面测试");

        assert!((response.predictions[0].predicted_change_percent - 1.0).abs() < 1e-9);
        assert!((response.predictions[0].predicted_price - 10.0).abs() < 1e-9);
        assert_eq!(
            response.predictions[0].key_factors.as_ref().unwrap()[0],
            "截面测试"
        );
    }

}
