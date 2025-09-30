use serde::{Deserialize, Serialize};
use crate::stock_prediction::backtest::BacktestReport;
use crate::stock_prediction::multi_timeframe_analysis::generate_multi_timeframe_signals;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub feature_name: String,
    pub importance_score: f64,
    pub correlation_with_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: String,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureOptimizationResult {
    pub current_features: Vec<String>,
    pub suggested_features: Vec<String>,
    pub feature_importance: Vec<FeatureImportance>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

/// 基于回测结果分析特征重要性
pub fn analyze_feature_importance(
    backtest_report: &BacktestReport,
    current_features: &[String],
) -> Result<FeatureOptimizationResult, Box<dyn std::error::Error>> {
    let mut feature_importance = Vec::new();
    let mut suggestions = Vec::new();
    
    // 分析当前特征的表现
    for feature in current_features {
        let importance = calculate_feature_importance(feature, backtest_report)?;
        feature_importance.push(importance);
    }
    
    // 根据回测结果生成优化建议
    suggestions.extend(generate_feature_suggestions(backtest_report)?);
    suggestions.extend(generate_parameter_suggestions(backtest_report)?);
    
    // 建议新特征
    let suggested_features = suggest_new_features(backtest_report, current_features)?;
    
    Ok(FeatureOptimizationResult {
        current_features: current_features.to_vec(),
        suggested_features,
        feature_importance,
        optimization_suggestions: suggestions,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeFeatureAnalysis {
    pub monthly_macd_golden_accuracy: f64,
    pub monthly_macd_death_accuracy: f64,
    pub weekly_macd_golden_accuracy: f64,
    pub weekly_macd_death_accuracy: f64,
    pub daily_macd_golden_accuracy: f64,
    pub daily_macd_death_accuracy: f64,
    pub monthly_kdj_golden_accuracy: f64,
    pub monthly_kdj_death_accuracy: f64,
    pub weekly_kdj_golden_accuracy: f64,
    pub weekly_kdj_death_accuracy: f64,
    pub daily_kdj_golden_accuracy: f64,
    pub daily_kdj_death_accuracy: f64,
    pub combined_signal_accuracy: f64,
    pub feature_weights: std::collections::HashMap<String, f64>,
    pub optimization_suggestions: Vec<FeatureOptimizationSuggestion>,
}

/// 生成多时间周期特征的预测权重
pub fn generate_multi_timeframe_weights(analysis: &MultiTimeframeFeatureAnalysis) -> MultiTimeframeWeights {
    let mut weights = MultiTimeframeWeights::default();
    
    // 基于准确率设置权重
    weights.monthly_macd_weight = (analysis.monthly_macd_golden_accuracy + analysis.monthly_macd_death_accuracy) / 2.0 * 0.3;
    weights.weekly_macd_weight = (analysis.weekly_macd_golden_accuracy + analysis.weekly_macd_death_accuracy) / 2.0 * 0.2;
    weights.daily_macd_weight = (analysis.daily_macd_golden_accuracy + analysis.daily_macd_death_accuracy) / 2.0 * 0.1;
    
    weights.monthly_kdj_weight = (analysis.monthly_kdj_golden_accuracy + analysis.monthly_kdj_death_accuracy) / 2.0 * 0.3;
    weights.weekly_kdj_weight = (analysis.weekly_kdj_golden_accuracy + analysis.weekly_kdj_death_accuracy) / 2.0 * 0.2;
    weights.daily_kdj_weight = (analysis.daily_kdj_golden_accuracy + analysis.daily_kdj_death_accuracy) / 2.0 * 0.1;
    
    // 综合信号权重
    weights.combined_signal_weight = analysis.combined_signal_accuracy * 0.4;
    
    // 归一化权重
    let total_weight = weights.monthly_macd_weight + weights.weekly_macd_weight + weights.daily_macd_weight
        + weights.monthly_kdj_weight + weights.weekly_kdj_weight + weights.daily_kdj_weight
        + weights.combined_signal_weight;
    
    if total_weight > 0.0 {
        weights.monthly_macd_weight /= total_weight;
        weights.weekly_macd_weight /= total_weight;
        weights.daily_macd_weight /= total_weight;
        weights.monthly_kdj_weight /= total_weight;
        weights.weekly_kdj_weight /= total_weight;
        weights.daily_kdj_weight /= total_weight;
        weights.combined_signal_weight /= total_weight;
    }
    
    weights
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeWeights {
    pub monthly_macd_weight: f64,
    pub weekly_macd_weight: f64,
    pub daily_macd_weight: f64,
    pub monthly_kdj_weight: f64,
    pub weekly_kdj_weight: f64,
    pub daily_kdj_weight: f64,
    pub combined_signal_weight: f64,
}

impl Default for MultiTimeframeWeights {
    fn default() -> Self {
        Self {
            monthly_macd_weight: 0.2,
            weekly_macd_weight: 0.15,
            daily_macd_weight: 0.1,
            monthly_kdj_weight: 0.2,
            weekly_kdj_weight: 0.15,
            daily_kdj_weight: 0.1,
            combined_signal_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureOptimizationSuggestion {
    pub feature_name: String,
    pub current_performance: f64,
    pub suggested_weight: f64,
    pub optimization_type: String,
    pub expected_improvement: f64,
    pub implementation_difficulty: String,
    pub estimated_time: String,
}

/// 分析多时间周期技术指标特征的重要性
pub fn analyze_multi_timeframe_features(
    backtest_entries: &[crate::stock_prediction::backtest::BacktestEntry],
    historical_data: &[crate::db::models::HistoricalData],
    symbol: &str,
) -> MultiTimeframeFeatureAnalysis {
    // 生成多时间周期信号
    let multi_signals = generate_multi_timeframe_signals(symbol, historical_data);
    
    let mut monthly_macd_golden_accuracy = 0.0;
    let mut monthly_macd_death_accuracy = 0.0;
    let mut weekly_macd_golden_accuracy = 0.0;
    let mut weekly_macd_death_accuracy = 0.0;
    let mut daily_macd_golden_accuracy = 0.0;
    let mut daily_macd_death_accuracy = 0.0;
    
    let mut monthly_kdj_golden_accuracy = 0.0;
    let mut monthly_kdj_death_accuracy = 0.0;
    let mut weekly_kdj_golden_accuracy = 0.0;
    let mut weekly_kdj_death_accuracy = 0.0;
    let mut daily_kdj_golden_accuracy = 0.0;
    let mut daily_kdj_death_accuracy = 0.0;
    
    let mut monthly_macd_golden_count = 0;
    let mut monthly_macd_death_count = 0;
    let mut weekly_macd_golden_count = 0;
    let mut weekly_macd_death_count = 0;
    let mut daily_macd_golden_count = 0;
    let mut daily_macd_death_count = 0;
    
    let mut monthly_kdj_golden_count = 0;
    let mut monthly_kdj_death_count = 0;
    let mut weekly_kdj_golden_count = 0;
    let mut weekly_kdj_death_count = 0;
    let mut daily_kdj_golden_count = 0;
    let mut daily_kdj_death_count = 0;
    
    // 分析每个信号与回测结果的关联性
    for signal in &multi_signals {
        // 查找对应日期的回测结果
        if let Some(backtest_entry) = backtest_entries.iter().find(|e| e.prediction_date == signal.date) {
            let actual_accuracy = backtest_entry.direction_accuracy;
            
            // 分析MACD信号
            if signal.monthly_macd_signal.is_golden_cross {
                monthly_macd_golden_accuracy += actual_accuracy;
                monthly_macd_golden_count += 1;
            }
            if signal.monthly_macd_signal.is_death_cross {
                monthly_macd_death_accuracy += actual_accuracy;
                monthly_macd_death_count += 1;
            }
            
            if signal.weekly_macd_signal.is_golden_cross {
                weekly_macd_golden_accuracy += actual_accuracy;
                weekly_macd_golden_count += 1;
            }
            if signal.weekly_macd_signal.is_death_cross {
                weekly_macd_death_accuracy += actual_accuracy;
                weekly_macd_death_count += 1;
            }
            
            if signal.daily_macd_signal.is_golden_cross {
                daily_macd_golden_accuracy += actual_accuracy;
                daily_macd_golden_count += 1;
            }
            if signal.daily_macd_signal.is_death_cross {
                daily_macd_death_accuracy += actual_accuracy;
                daily_macd_death_count += 1;
            }
            
            // 分析KDJ信号
            if signal.monthly_kdj_signal.is_golden_cross {
                monthly_kdj_golden_accuracy += actual_accuracy;
                monthly_kdj_golden_count += 1;
            }
            if signal.monthly_kdj_signal.is_death_cross {
                monthly_kdj_death_accuracy += actual_accuracy;
                monthly_kdj_death_count += 1;
            }
            
            if signal.weekly_kdj_signal.is_golden_cross {
                weekly_kdj_golden_accuracy += actual_accuracy;
                weekly_kdj_golden_count += 1;
            }
            if signal.weekly_kdj_signal.is_death_cross {
                weekly_kdj_death_accuracy += actual_accuracy;
                weekly_kdj_death_count += 1;
            }
            
            if signal.daily_kdj_signal.is_golden_cross {
                daily_kdj_golden_accuracy += actual_accuracy;
                daily_kdj_golden_count += 1;
            }
            if signal.daily_kdj_signal.is_death_cross {
                daily_kdj_death_accuracy += actual_accuracy;
                daily_kdj_death_count += 1;
            }
        }
    }
    
    // 计算平均准确率
    let monthly_macd_golden_avg = if monthly_macd_golden_count > 0 {
        monthly_macd_golden_accuracy / monthly_macd_golden_count as f64
    } else { 0.0 };
    
    let monthly_macd_death_avg = if monthly_macd_death_count > 0 {
        monthly_macd_death_accuracy / monthly_macd_death_count as f64
    } else { 0.0 };
    
    let weekly_macd_golden_avg = if weekly_macd_golden_count > 0 {
        weekly_macd_golden_accuracy / weekly_macd_golden_count as f64
    } else { 0.0 };
    
    let weekly_macd_death_avg = if weekly_macd_death_count > 0 {
        weekly_macd_death_accuracy / weekly_macd_death_count as f64
    } else { 0.0 };
    
    let daily_macd_golden_avg = if daily_macd_golden_count > 0 {
        daily_macd_golden_accuracy / daily_macd_golden_count as f64
    } else { 0.0 };
    
    let daily_macd_death_avg = if daily_macd_death_count > 0 {
        daily_macd_death_accuracy / daily_macd_death_count as f64
    } else { 0.0 };
    
    let monthly_kdj_golden_avg = if monthly_kdj_golden_count > 0 {
        monthly_kdj_golden_accuracy / monthly_kdj_golden_count as f64
    } else { 0.0 };
    
    let monthly_kdj_death_avg = if monthly_kdj_death_count > 0 {
        monthly_kdj_death_accuracy / monthly_kdj_death_count as f64
    } else { 0.0 };
    
    let weekly_kdj_golden_avg = if weekly_kdj_golden_count > 0 {
        weekly_kdj_golden_accuracy / weekly_kdj_golden_count as f64
    } else { 0.0 };
    
    let weekly_kdj_death_avg = if weekly_kdj_death_count > 0 {
        weekly_kdj_death_accuracy / weekly_kdj_death_count as f64
    } else { 0.0 };
    
    let daily_kdj_golden_avg = if daily_kdj_golden_count > 0 {
        daily_kdj_golden_accuracy / daily_kdj_golden_count as f64
    } else { 0.0 };
    
    let daily_kdj_death_avg = if daily_kdj_death_count > 0 {
        daily_kdj_death_accuracy / daily_kdj_death_count as f64
    } else { 0.0 };
    
    // 计算特征权重（基于准确率和信号频率）
    let mut feature_weights = std::collections::HashMap::new();
    
    // MACD特征权重
    feature_weights.insert("monthly_macd_golden".to_string(), monthly_macd_golden_avg * 0.3);
    feature_weights.insert("monthly_macd_death".to_string(), monthly_macd_death_avg * 0.3);
    feature_weights.insert("weekly_macd_golden".to_string(), weekly_macd_golden_avg * 0.2);
    feature_weights.insert("weekly_macd_death".to_string(), weekly_macd_death_avg * 0.2);
    feature_weights.insert("daily_macd_golden".to_string(), daily_macd_golden_avg * 0.1);
    feature_weights.insert("daily_macd_death".to_string(), daily_macd_death_avg * 0.1);
    
    // KDJ特征权重
    feature_weights.insert("monthly_kdj_golden".to_string(), monthly_kdj_golden_avg * 0.3);
    feature_weights.insert("monthly_kdj_death".to_string(), monthly_kdj_death_avg * 0.3);
    feature_weights.insert("weekly_kdj_golden".to_string(), weekly_kdj_golden_avg * 0.2);
    feature_weights.insert("weekly_kdj_death".to_string(), weekly_kdj_death_avg * 0.2);
    feature_weights.insert("daily_kdj_golden".to_string(), daily_kdj_golden_avg * 0.1);
    feature_weights.insert("daily_kdj_death".to_string(), daily_kdj_death_avg * 0.1);
    
    // 生成优化建议
    let mut suggestions = Vec::new();
    
    // 月线MACD优化建议
    if monthly_macd_golden_avg > 0.7 {
        suggestions.push(FeatureOptimizationSuggestion {
            feature_name: "月线MACD金叉".to_string(),
            current_performance: monthly_macd_golden_avg,
            suggested_weight: 0.3,
            optimization_type: "增加权重".to_string(),
            expected_improvement: 0.08,
            implementation_difficulty: "简单".to_string(),
            estimated_time: "1小时".to_string(),
        });
    }
    
    if monthly_macd_death_avg > 0.7 {
        suggestions.push(FeatureOptimizationSuggestion {
            feature_name: "月线MACD死叉".to_string(),
            current_performance: monthly_macd_death_avg,
            suggested_weight: 0.3,
            optimization_type: "增加权重".to_string(),
            expected_improvement: 0.08,
            implementation_difficulty: "简单".to_string(),
            estimated_time: "1小时".to_string(),
        });
    }
    
    // 周线KDJ优化建议
    if weekly_kdj_golden_avg > 0.65 {
        suggestions.push(FeatureOptimizationSuggestion {
            feature_name: "周线KDJ金叉".to_string(),
            current_performance: weekly_kdj_golden_avg,
            suggested_weight: 0.2,
            optimization_type: "增加权重".to_string(),
            expected_improvement: 0.05,
            implementation_difficulty: "简单".to_string(),
            estimated_time: "1小时".to_string(),
        });
    }
    
    if weekly_kdj_death_avg > 0.65 {
        suggestions.push(FeatureOptimizationSuggestion {
            feature_name: "周线KDJ死叉".to_string(),
            current_performance: weekly_kdj_death_avg,
            suggested_weight: 0.2,
            optimization_type: "增加权重".to_string(),
            expected_improvement: 0.05,
            implementation_difficulty: "简单".to_string(),
            estimated_time: "1小时".to_string(),
        });
    }
    
    // 综合信号优化建议
    let combined_signal_accuracy = multi_signals.iter()
        .map(|s| s.prediction_confidence)
        .sum::<f64>() / multi_signals.len() as f64;
    
    if combined_signal_accuracy > 0.75 {
        suggestions.push(FeatureOptimizationSuggestion {
            feature_name: "多时间周期综合信号".to_string(),
            current_performance: combined_signal_accuracy,
            suggested_weight: 0.4,
            optimization_type: "新增特征".to_string(),
            expected_improvement: 0.12,
            implementation_difficulty: "中等".to_string(),
            estimated_time: "3小时".to_string(),
        });
    }
    
    MultiTimeframeFeatureAnalysis {
        monthly_macd_golden_accuracy: monthly_macd_golden_avg,
        monthly_macd_death_accuracy: monthly_macd_death_avg,
        weekly_macd_golden_accuracy: weekly_macd_golden_avg,
        weekly_macd_death_accuracy: weekly_macd_death_avg,
        daily_macd_golden_accuracy: daily_macd_golden_avg,
        daily_macd_death_accuracy: daily_macd_death_avg,
        monthly_kdj_golden_accuracy: monthly_kdj_golden_avg,
        monthly_kdj_death_accuracy: monthly_kdj_death_avg,
        weekly_kdj_golden_accuracy: weekly_kdj_golden_avg,
        weekly_kdj_death_accuracy: weekly_kdj_death_avg,
        daily_kdj_golden_accuracy: daily_kdj_golden_avg,
        daily_kdj_death_accuracy: daily_kdj_death_avg,
        combined_signal_accuracy,
        feature_weights,
        optimization_suggestions: suggestions,
    }
}

fn calculate_feature_importance(
    feature: &str,
    backtest_report: &BacktestReport,
) -> Result<FeatureImportance, Box<dyn std::error::Error>> {
    // 基于回测准确率计算特征重要性
    let accuracy_correlation = match feature {
        "rsi" => analyze_rsi_correlation(backtest_report),
        "macd" => analyze_macd_correlation(backtest_report),
        "kdj" => analyze_kdj_correlation(backtest_report),
        "volume" => analyze_volume_correlation(backtest_report),
        "ma_trend" => analyze_ma_trend_correlation(backtest_report),
        _ => 0.5, // 默认中等重要性
    };
    
    let importance_score = calculate_importance_score(feature, backtest_report);
    
    Ok(FeatureImportance {
        feature_name: feature.to_string(),
        importance_score,
        correlation_with_accuracy: accuracy_correlation,
    })
}

fn analyze_rsi_correlation(backtest_report: &BacktestReport) -> f64 {
    // 分析RSI与预测准确率的相关性
    let mut high_accuracy_count = 0;
    let mut total_count = 0;
    
    for entry in &backtest_report.backtest_entries {
        for prediction in &entry.predictions {
            if let Some(tech_indicators) = &prediction.technical_indicators {
                total_count += 1;
                // RSI在30-70区间时准确率通常更高
                if tech_indicators.rsi > 30.0 && tech_indicators.rsi < 70.0
                    && entry.price_accuracy > 0.7 {
                        high_accuracy_count += 1;
                    }
            }
        }
    }
    
    if total_count > 0 {
        high_accuracy_count as f64 / total_count as f64
    } else {
        0.5
    }
}

fn analyze_macd_correlation(backtest_report: &BacktestReport) -> f64 {
    let mut correlation_sum = 0.0;
    let mut count = 0;
    
    for entry in &backtest_report.backtest_entries {
        for prediction in &entry.predictions {
            if let Some(tech_indicators) = &prediction.technical_indicators {
                count += 1;
                // MACD金叉死叉信号与准确率的关系
                let macd_signal_strength = if tech_indicators.macd_golden_cross {
                    0.8
                } else if tech_indicators.macd_death_cross {
                    0.3
                } else {
                    0.5
                };
                
                correlation_sum += macd_signal_strength * entry.price_accuracy;
            }
        }
    }
    
    if count > 0 {
        correlation_sum / count as f64
    } else {
        0.5
    }
}

fn analyze_kdj_correlation(backtest_report: &BacktestReport) -> f64 {
    let mut high_accuracy_with_kdj = 0;
    let mut total_kdj_signals = 0;
    
    for entry in &backtest_report.backtest_entries {
        for prediction in &entry.predictions {
            if let Some(tech_indicators) = &prediction.technical_indicators {
                if tech_indicators.kdj_golden_cross || tech_indicators.kdj_death_cross {
                    total_kdj_signals += 1;
                    if entry.price_accuracy > 0.65 {
                        high_accuracy_with_kdj += 1;
                    }
                }
            }
        }
    }
    
    if total_kdj_signals > 0 {
        high_accuracy_with_kdj as f64 / total_kdj_signals as f64
    } else {
        0.5
    }
}

fn analyze_volume_correlation(backtest_report: &BacktestReport) -> f64 {
    // 分析成交量与预测准确率的关系
    let mut high_volume_accuracy = Vec::new();
    let mut low_volume_accuracy = Vec::new();
    
    for entry in &backtest_report.backtest_entries {
        for prediction in &entry.predictions {
            // 假设通过某种方式获取成交量数据
            // 这里简化处理，实际应该从历史数据中获取
            if prediction.confidence > 0.7 {
                high_volume_accuracy.push(entry.price_accuracy);
            } else {
                low_volume_accuracy.push(entry.price_accuracy);
            }
        }
    }
    
    let high_vol_avg = if !high_volume_accuracy.is_empty() {
        high_volume_accuracy.iter().sum::<f64>() / high_volume_accuracy.len() as f64
    } else {
        0.5
    };
    
    let low_vol_avg = if !low_volume_accuracy.is_empty() {
        low_volume_accuracy.iter().sum::<f64>() / low_volume_accuracy.len() as f64
    } else {
        0.5
    };
    
    (high_vol_avg - low_vol_avg).abs()
}

fn analyze_ma_trend_correlation(backtest_report: &BacktestReport) -> f64 {
    // 分析均线趋势与预测准确率的关系
    let mut trend_accuracy_sum = 0.0;
    let mut count = 0;
    
    for entry in &backtest_report.backtest_entries {
        count += 1;
        // 简化处理：假设方向准确率高说明趋势特征有效
        trend_accuracy_sum += entry.direction_accuracy;
    }
    
    if count > 0 {
        trend_accuracy_sum / count as f64
    } else {
        0.5
    }
}

fn calculate_importance_score(feature: &str, backtest_report: &BacktestReport) -> f64 {
    // 基于整体准确率和特征类型计算重要性分数
    let base_accuracy = backtest_report.overall_price_accuracy;
    
    let base_score = match feature {
        "close" => 0.9,  // 收盘价是最重要的基础特征
        "volume" => 0.8, // 成交量是重要的确认指标
        "ma_trend" => 0.85, // 趋势是核心特征
        "rsi" => 0.7,    // RSI是重要的超买超卖指标
        "macd" => 0.75,  // MACD是重要的趋势确认指标
        "kdj" => 0.65,   // KDJ是有用的辅助指标
        "bollinger" => 0.6, // 布林带提供波动性信息
        "cci" => 0.55,   // CCI是补充指标
        "obv" => 0.5,    // OBV是成交量确认指标
        _ => 0.4,        // 其他特征默认较低重要性
    };
    
    base_score * (1.0 + base_accuracy) // 根据整体准确率调整
}

fn generate_feature_suggestions(
    backtest_report: &BacktestReport,
) -> Result<Vec<OptimizationSuggestion>, Box<dyn std::error::Error>> {
    let mut suggestions = Vec::new();
    
    // 基于整体准确率给出建议
    if backtest_report.overall_price_accuracy < 0.6 {
        suggestions.push(OptimizationSuggestion {
            suggestion_type: "增加技术指标".to_string(),
            description: "当前价格预测准确率较低，建议增加更多技术指标如DMI、ADX等".to_string(),
            expected_improvement: 0.1,
            implementation_priority: 1,
        });
    }
    
    if backtest_report.overall_direction_accuracy < 0.7 {
        suggestions.push(OptimizationSuggestion {
            suggestion_type: "强化趋势特征".to_string(),
            description: "方向预测准确率不足，建议增加多时间框架的均线系统".to_string(),
            expected_improvement: 0.15,
            implementation_priority: 2,
        });
    }
    
    // 分析误差分布
    if backtest_report.average_prediction_error > 0.05 {
        suggestions.push(OptimizationSuggestion {
            suggestion_type: "优化模型参数".to_string(),
            description: "预测误差较大，建议调整学习率、增加训练轮数或使用更复杂的模型".to_string(),
            expected_improvement: 0.08,
            implementation_priority: 3,
        });
    }
    
    Ok(suggestions)
}

fn generate_parameter_suggestions(
    backtest_report: &BacktestReport,
) -> Result<Vec<OptimizationSuggestion>, Box<dyn std::error::Error>> {
    let mut suggestions = Vec::new();
    
    // 基于准确率趋势分析
    let trend_declining = is_accuracy_declining(&backtest_report.accuracy_trend);
    if trend_declining {
        suggestions.push(OptimizationSuggestion {
            suggestion_type: "防止过拟合".to_string(),
            description: "准确率呈下降趋势，建议增加Dropout率或使用正则化".to_string(),
            expected_improvement: 0.12,
            implementation_priority: 1,
        });
    }
    
    // 基于市场波动性分析
    let high_volatility_periods = count_high_volatility_periods(&backtest_report.daily_accuracy);
    if high_volatility_periods > backtest_report.daily_accuracy.len() / 3 {
        suggestions.push(OptimizationSuggestion {
            suggestion_type: "适应性训练".to_string(),
            description: "市场波动较大，建议使用滑动窗口训练或在线学习".to_string(),
            expected_improvement: 0.1,
            implementation_priority: 2,
        });
    }
    
    Ok(suggestions)
}

fn suggest_new_features(
    backtest_report: &BacktestReport,
    current_features: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut suggested = Vec::new();
    
    // 基于当前特征和表现建议新特征
    if !current_features.contains(&"atr".to_string()) && 
       backtest_report.average_prediction_error > 0.04 {
        suggested.push("atr".to_string()); // 平均真实波动率
    }
    
    if !current_features.contains(&"williams_r".to_string()) &&
       backtest_report.overall_direction_accuracy < 0.65 {
        suggested.push("williams_r".to_string()); // 威廉指标
    }
    
    if !current_features.contains(&"momentum_roc".to_string()) {
        suggested.push("momentum_roc".to_string()); // 变动率指标
    }
    
    // 如果准确率在某些时间段表现不佳，建议时间特征
    if has_temporal_bias(&backtest_report.daily_accuracy) {
        suggested.push("day_of_week".to_string());
        suggested.push("month_of_year".to_string());
        suggested.push("quarter".to_string());
    }
    
    Ok(suggested)
}

fn is_accuracy_declining(accuracy_trend: &[f64]) -> bool {
    if accuracy_trend.len() < 3 {
        return false;
    }
    
    let recent_avg = accuracy_trend[accuracy_trend.len()-3..].iter().sum::<f64>() / 3.0;
    let early_avg = accuracy_trend[..3].iter().sum::<f64>() / 3.0;
    
    recent_avg < early_avg - 0.05
}

fn count_high_volatility_periods(daily_accuracy: &[crate::stock_prediction::backtest::DailyAccuracy]) -> usize {
    daily_accuracy.iter()
        .filter(|day| day.market_volatility > 0.03)
        .count()
}

fn has_temporal_bias(daily_accuracy: &[crate::stock_prediction::backtest::DailyAccuracy]) -> bool {
    // 简化判断：如果某些日期的准确率明显偏低
    let avg_accuracy = daily_accuracy.iter()
        .map(|day| day.price_accuracy)
        .sum::<f64>() / daily_accuracy.len() as f64;
    
    daily_accuracy.iter()
        .any(|day| day.price_accuracy < avg_accuracy - 0.15)
} 