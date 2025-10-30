//! 股票预测模块快速入门示例
//!
//! 本示例展示如何使用优化后的股票预测模块进行高精度预测
//!
//! 运行方式：
//! ```bash
//! cargo run --example stock_prediction_quick_start
//! ```

use std::collections::HashMap;

// 示例：使用集成学习进行预测
async fn example_ensemble_prediction() -> Result<(), String> {
    println!("=== 集成学习预测示例 ===\n");

    // 模拟多个模型的预测结果
    let predictions = vec![
        ModelPrediction {
            model_name: "深度神经网络模型".to_string(),
            model_type: "深度学习".to_string(),
            predicted_direction: 1,
            predicted_change: 0.025,
            confidence: 0.82,
            weight: 1.0,
            features_used: vec!["RSI".to_string(), "MACD".to_string(), "KDJ".to_string()],
        },
        ModelPrediction {
            model_name: "技术分析模型".to_string(),
            model_type: "技术指标".to_string(),
            predicted_direction: 1,
            predicted_change: 0.018,
            confidence: 0.75,
            weight: 0.8,
            features_used: vec!["MA5".to_string(), "MA10".to_string(), "MA20".to_string()],
        },
        ModelPrediction {
            model_name: "统计模型".to_string(),
            model_type: "时间序列".to_string(),
            predicted_direction: 1,
            predicted_change: 0.022,
            confidence: 0.78,
            weight: 0.9,
            features_used: vec!["ARIMA".to_string()],
        },
        ModelPrediction {
            model_name: "动量模型".to_string(),
            model_type: "技术分析".to_string(),
            predicted_direction: 0,
            predicted_change: 0.003,
            confidence: 0.65,
            weight: 0.7,
            features_used: vec!["ROC".to_string(), "Momentum".to_string()],
        },
    ];

    // 配置集成策略
    let config = EnsembleConfig {
        strategy: EnsembleStrategy::Hybrid, // 使用混合策略
        min_models: 3,
        confidence_threshold: 0.6,
        adaptive_weights: true,
        outlier_removal: true,
    };

    // 执行集成预测
    let ensemble_result = ensemble_predictions(predictions.clone(), &config);

    // 输出结果
    println!("📊 集成预测结果：");
    println!("  预测方向: {}",
        match ensemble_result.final_direction {
            1 => "上涨 ↗",
            -1 => "下跌 ↘",
            _ => "横盘 →",
        }
    );
    println!("  预测变化率: {:.2}%", ensemble_result.final_change * 100.0);
    println!("  集成置信度: {:.1}%", ensemble_result.ensemble_confidence * 100.0);
    println!("  模型一致性: {:.1}%", ensemble_result.consensus_score * 100.0);
    println!("  策略类型: {}", ensemble_result.strategy_used);
    println!("\n🔍 风险评估：");
    println!("  风险等级: {}", ensemble_result.risk_assessment.risk_level);
    println!("  不确定性: {:.1}%", ensemble_result.risk_assessment.uncertainty_score * 100.0);
    println!("  模型分歧度: {:.3}", ensemble_result.risk_assessment.model_disagreement);
    println!("  市场波动率: {:.3}", ensemble_result.risk_assessment.market_volatility);
    println!("  操作建议: {}", ensemble_result.risk_assessment.recommendation);

    println!("\n📋 各模型预测详情：");
    for (i, pred) in predictions.iter().enumerate() {
        println!("  {}. {} ({})",
            i + 1,
            pred.model_name,
            pred.model_type
        );
        println!("     方向: {}, 变化: {:.2}%, 置信度: {:.1}%, 权重: {:.2}",
            match pred.predicted_direction {
                1 => "↗",
                -1 => "↘",
                _ => "→",
            },
            pred.predicted_change * 100.0,
            pred.confidence * 100.0,
            pred.weight
        );
    }

    // 决策逻辑示例
    println!("\n💡 决策建议：");
    if ensemble_result.ensemble_confidence > 0.75
        && ensemble_result.consensus_score > 0.75
        && ensemble_result.risk_assessment.risk_level != "极高" {
        println!("  ✅ 信号强烈，可以考虑{}操作",
            if ensemble_result.final_direction > 0 { "买入" }
            else if ensemble_result.final_direction < 0 { "卖出" }
            else { "观望" }
        );
        println!("  建议仓位: {}",
            if ensemble_result.ensemble_confidence > 0.85 { "70-80%" }
            else if ensemble_result.ensemble_confidence > 0.75 { "50-60%" }
            else { "30-40%" }
        );
    } else if ensemble_result.ensemble_confidence > 0.6 {
        println!("  ⚠️  信号中等，建议谨慎操作或轻仓试探");
        println!("  建议仓位: 20-30%");
    } else {
        println!("  ❌ 信号不明确，建议观望");
    }

    Ok(())
}

// 示例：计算和使用高级特征
fn example_advanced_features() -> Result<(), String> {
    println!("\n\n=== 高级特征工程示例 ===\n");

    // 模拟历史数据
    let prices = vec![
        100.0, 101.5, 103.0, 102.5, 104.0, 105.5, 107.0, 106.0, 108.0, 109.5,
        111.0, 110.5, 112.0, 113.5, 115.0, 114.0, 116.0, 117.5, 119.0, 118.5,
        120.0, 121.5, 123.0, 122.0, 124.0, 125.5, 127.0, 126.0, 128.0, 129.5,
    ];
    let volumes = vec![
        1000000, 1100000, 1200000, 900000, 1300000, 1400000, 1500000, 1100000,
        1600000, 1700000, 1800000, 1200000, 1900000, 2000000, 2100000, 1500000,
        2200000, 2300000, 2400000, 1800000, 2500000, 2600000, 2700000, 2000000,
        2800000, 2900000, 3000000, 2200000, 3100000, 3200000,
    ];
    let highs = prices.iter().map(|&p| p * 1.02).collect::<Vec<_>>();
    let lows = prices.iter().map(|&p| p * 0.98).collect::<Vec<_>>();
    let opens = prices.iter().map(|&p| p * 0.995).collect::<Vec<_>>();

    // 计算高级特征
    let advanced_features = calculate_advanced_features(
        &prices,
        &volumes,
        &highs,
        &lows,
        &opens,
    );

    // 展示特征结果
    println!("📈 动量特征：");
    println!("  ROC-5日: {:.2}%", advanced_features.momentum_features.roc_5 * 100.0);
    println!("  ROC-10日: {:.2}%", advanced_features.momentum_features.roc_10 * 100.0);
    println!("  ROC-20日: {:.2}%", advanced_features.momentum_features.roc_20 * 100.0);
    println!("  价格加速度: {:.4}", advanced_features.momentum_features.acceleration);
    println!("  价格急动度: {:.4}", advanced_features.momentum_features.jerk);

    println!("\n📊 波动率特征：");
    println!("  已实现波动率: {:.2}%", advanced_features.volatility_features.realized_volatility * 100.0);
    println!("  Parkinson波动率: {:.2}%", advanced_features.volatility_features.parkinson_volatility * 100.0);
    println!("  波动率比率: {:.2}", advanced_features.volatility_features.volatility_ratio);
    println!("  波动率趋势: {:.2}%", advanced_features.volatility_features.volatility_trend * 100.0);
    println!("  标准化ATR: {:.2}%", advanced_features.volatility_features.atr_normalized * 100.0);
    println!("  波动率状态: {}",
        match advanced_features.volatility_features.volatility_regime {
            1 => "高波动 ⚡",
            -1 => "低波动 📉",
            _ => "中等波动 📊",
        }
    );

    println!("\n📐 统计特征：");
    println!("  收益率均值: {:.4}%", advanced_features.statistical_features.returns_mean * 100.0);
    println!("  收益率标准差: {:.4}", advanced_features.statistical_features.returns_std);
    println!("  偏度(Skewness): {:.3}", advanced_features.statistical_features.returns_skewness);
    println!("  峰度(Kurtosis): {:.3}", advanced_features.statistical_features.returns_kurtosis);
    println!("  Hurst指数: {:.3}", advanced_features.statistical_features.hurst_exponent);
    let hurst_interpretation = if advanced_features.statistical_features.hurst_exponent > 0.6 {
        "强趋势性"
    } else if advanced_features.statistical_features.hurst_exponent < 0.4 {
        "强均值回归"
    } else {
        "随机游走"
    };
    println!("    → 市场特征: {}", hurst_interpretation);
    println!("  自相关(1阶): {:.3}", advanced_features.statistical_features.autocorr_1);
    println!("  信息熵: {:.3}", advanced_features.statistical_features.entropy);

    println!("\n💰 价量特征：");
    println!("  VWAP: {:.2}", advanced_features.price_volume_features.vwap);
    println!("  价量相关性: {:.3}", advanced_features.price_volume_features.volume_price_correlation);
    println!("  成交量趋势: {:.2}%", advanced_features.price_volume_features.volume_trend * 100.0);
    println!("  成交量突破: {:.2}%", advanced_features.price_volume_features.volume_breakout * 100.0);
    println!("  买盘压力: {:.1}%", advanced_features.price_volume_features.buying_pressure * 100.0);
    println!("  卖盘压力: {:.1}%", advanced_features.price_volume_features.selling_pressure * 100.0);
    println!("  资金流量指标: {:.1}", advanced_features.price_volume_features.money_flow_index);

    println!("\n📈 趋势特征：");
    println!("  趋势强度: {:.1}%", advanced_features.trend_features.trend_strength * 100.0);
    println!("  趋势一致性: {:.1}%", advanced_features.trend_features.trend_consistency * 100.0);
    println!("  ADX: {:.1}", advanced_features.trend_features.adx);
    println!("  均线排列度: {:.2}", advanced_features.trend_features.ma_alignment);
    println!("  价格相对位置(布林带): {:.2}", advanced_features.trend_features.price_position);
    println!("  距离支撑位: {:.2}%", advanced_features.trend_features.support_distance * 100.0);
    println!("  距离阻力位: {:.2}%", advanced_features.trend_features.resistance_distance * 100.0);

    println!("\n🏗️  市场结构特征：");
    println!("  连续更高高点: {}", advanced_features.market_structure_features.higher_highs);
    println!("  连续更低低点: {}", advanced_features.market_structure_features.lower_lows);
    println!("  摆动强度: {:.2}%", advanced_features.market_structure_features.swing_strength * 100.0);
    println!("  区间扩张度: {:.2}%", advanced_features.market_structure_features.range_expansion * 100.0);
    println!("  跳空比例: {:.1}%", advanced_features.market_structure_features.gap_ratio * 100.0);
    println!("  影线比例: {:.1}%", advanced_features.market_structure_features.tail_ratio * 100.0);
    println!("  实体比例: {:.1}%", advanced_features.market_structure_features.body_ratio * 100.0);
    println!("  盘整得分: {:.1}%", advanced_features.market_structure_features.consolidation_score * 100.0);

    // 基于特征的市场分析
    println!("\n🔮 市场状态分析：");

    // 趋势分析
    if advanced_features.trend_features.trend_strength > 0.7 {
        println!("  ✓ 市场处于强趋势状态");
        if advanced_features.trend_features.ma_alignment > 0.5 {
            println!("    → 多头排列，上升趋势");
        } else if advanced_features.trend_features.ma_alignment < -0.5 {
            println!("    → 空头排列，下降趋势");
        }
    } else if advanced_features.market_structure_features.consolidation_score > 0.6 {
        println!("  ✓ 市场处于盘整状态");
    }

    // 波动率分析
    match advanced_features.volatility_features.volatility_regime {
        1 => println!("  ✓ 高波动环境，注意风险控制"),
        -1 => println!("  ✓ 低波动环境，可能酝酿突破"),
        _ => println!("  ✓ 正常波动范围"),
    }

    // 动量分析
    if advanced_features.momentum_features.acceleration > 0.001 {
        println!("  ✓ 价格加速上涨，动能强劲");
    } else if advanced_features.momentum_features.acceleration < -0.001 {
        println!("  ✓ 价格加速下跌，动能减弱");
    }

    // 价量配合分析
    if advanced_features.price_volume_features.volume_price_correlation > 0.5 {
        println!("  ✓ 价量配合良好，趋势可靠性高");
    } else if advanced_features.price_volume_features.volume_price_correlation < -0.3 {
        println!("  ⚠️  价量背离，需警惕反转");
    }

    Ok(())
}

// 示例：模型性能追踪
fn example_performance_tracking() {
    println!("\n\n=== 模型性能追踪示例 ===\n");

    let mut performances: HashMap<String, ModelPerformance> = HashMap::new();

    // 模拟一些预测和实际结果
    let predictions_and_actuals = vec![
        ("深度学习模型", 0.025, 0.022),
        ("深度学习模型", 0.015, 0.018),
        ("深度学习模型", -0.012, -0.010),
        ("技术分析模型", 0.020, 0.025),
        ("技术分析模型", 0.010, 0.008),
        ("统计模型", 0.018, 0.020),
        ("统计模型", -0.015, -0.012),
    ];

    // 更新性能统计
    for (model_name, predicted, actual) in predictions_and_actuals {
        update_model_performance(model_name, predicted, actual, &mut performances);
    }

    // 显示各模型性能
    println!("📊 模型性能统计：\n");
    for (name, perf) in performances.iter() {
        println!("🤖 {}:", name);
        println!("  总预测次数: {}", perf.total_predictions);
        println!("  正确预测: {}", perf.correct_predictions);
        println!("  方向准确率: {:.1}%", perf.direction_accuracy * 100.0);
        println!("  价格准确率: {:.1}%", perf.price_accuracy * 100.0);
        println!("  综合准确率: {:.1}%", perf.recent_accuracy * 100.0);
        println!("  平均误差: {:.4}", perf.avg_error);
        println!("  最后更新: {}\n", perf.last_updated);
    }

    // 推荐最佳模型
    let best_model = performances
        .iter()
        .max_by(|a, b| {
            a.1.recent_accuracy
                .partial_cmp(&b.1.recent_accuracy)
                .unwrap()
        });

    if let Some((name, perf)) = best_model {
        println!("🏆 当前最佳模型: {}", name);
        println!("   综合准确率: {:.1}%", perf.recent_accuracy * 100.0);
    }
}

// 完整工作流示例
async fn example_complete_workflow() -> Result<(), String> {
    println!("\n\n=== 完整预测工作流示例 ===\n");

    let stock_code = "000001"; // 平安银行
    println!("📌 股票代码: {}\n", stock_code);

    // 步骤1: 收集多个模型的预测
    println!("步骤1: 收集多个模型的预测...");
    let model_predictions = vec![
        ModelPrediction {
            model_name: "DNN主模型".to_string(),
            model_type: "深度学习".to_string(),
            predicted_direction: 1,
            predicted_change: 0.032,
            confidence: 0.85,
            weight: 1.2,
            features_used: vec![],
        },
        ModelPrediction {
            model_name: "技术分析辅助".to_string(),
            model_type: "技术指标".to_string(),
            predicted_direction: 1,
            predicted_change: 0.025,
            confidence: 0.78,
            weight: 1.0,
            features_used: vec![],
        },
        ModelPrediction {
            model_name: "量价分析".to_string(),
            model_type: "价量关系".to_string(),
            predicted_direction: 1,
            predicted_change: 0.028,
            confidence: 0.80,
            weight: 1.1,
            features_used: vec![],
        },
    ];
    println!("  ✓ 已收集{}个模型的预测\n", model_predictions.len());

    // 步骤2: 集成预测
    println!("步骤2: 执行集成预测...");
    let config = EnsembleConfig::default();
    let result = ensemble_predictions(model_predictions, &config);
    println!("  ✓ 集成完成，策略: {}\n", result.strategy_used);

    // 步骤3: 风险评估
    println!("步骤3: 风险评估...");
    println!("  风险等级: {}", result.risk_assessment.risk_level);
    println!("  置信度: {:.1}%", result.ensemble_confidence * 100.0);
    println!("  一致性: {:.1}%\n", result.consensus_score * 100.0);

    // 步骤4: 生成操作建议
    println!("步骤4: 生成操作建议...");

    let should_trade = result.ensemble_confidence > 0.7
        && result.consensus_score > 0.7
        && result.risk_assessment.risk_level != "极高";

    if should_trade {
        let position_size = if result.ensemble_confidence > 0.85 {
            "60-80%"
        } else if result.ensemble_confidence > 0.75 {
            "40-60%"
        } else {
            "20-40%"
        };

        println!("  📈 建议操作: {}",
            if result.final_direction > 0 { "买入" }
            else if result.final_direction < 0 { "卖出" }
            else { "观望" }
        );
        println!("  💰 建议仓位: {}", position_size);
        println!("  🎯 目标收益: {:.2}%", result.final_change * 100.0);

        // 止损止盈建议
        let stop_loss = result.final_change * -0.5;
        let take_profit = result.final_change * 1.5;
        println!("  🛡️  止损位: {:.2}%", stop_loss * 100.0);
        println!("  🎁 止盈位: {:.2}%", take_profit * 100.0);
    } else {
        println!("  ⏸️  建议: 信号不够明确，建议观望");
        println!("  原因: {}", result.risk_assessment.recommendation);
    }

    println!("\n✅ 预测工作流完成！");

    Ok(())
}

// 主函数
#[tokio::main]
async fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      股票预测模块优化 - 快速入门示例                    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // 运行示例
    example_ensemble_prediction().await?;
    example_advanced_features()?;
    example_performance_tracking();
    example_complete_workflow().await?;

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  更多信息请参考: STOCK_PREDICTION_OPTIMIZATION.md       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    Ok(())
}

// ===== 为了示例能够编译，这里包含必要的类型定义 =====
// 实际使用时应该从主模块导入

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_name: String,
    pub model_type: String,
    pub predicted_direction: i8,
    pub predicted_change: f64,
    pub confidence: f64,
    pub weight: f64,
    pub features_used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    pub final_direction: i8,
    pub final_change: f64,
    pub ensemble_confidence: f64,
    pub model_predictions: Vec<ModelPrediction>,
    pub consensus_score: f64,
    pub strategy_used: String,
    pub risk_assessment: RiskAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: String,
    pub uncertainty_score: f64,
    pub model_disagreement: f64,
    pub market_volatility: f64,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub min_models: usize,
    pub confidence_threshold: f64,
    pub adaptive_weights: bool,
    pub outlier_removal: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    Hybrid,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Hybrid,
            min_models: 3,
            confidence_threshold: 0.6,
            adaptive_weights: true,
            outlier_removal: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub model_name: String,
    pub recent_accuracy: f64,
    pub direction_accuracy: f64,
    pub price_accuracy: f64,
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub avg_error: f64,
    pub last_updated: String,
}

// 简化的函数实现（实际应该使用模块中的实现）
fn ensemble_predictions(
    predictions: Vec<ModelPrediction>,
    _config: &EnsembleConfig,
) -> EnsemblePrediction {
    // 简单的加权平均实现
    let total_weight: f64 = predictions.iter().map(|p| p.weight * p.confidence).sum();
    let weighted_change: f64 = predictions
        .iter()
        .map(|p| p.predicted_change * p.weight * p.confidence)
        .sum::<f64>()
        / total_weight;

    let weighted_confidence: f64 = predictions
        .iter()
        .map(|p| p.confidence * p.weight)
        .sum::<f64>()
        / predictions.iter().map(|p| p.weight).sum::<f64>();

    let final_direction = if weighted_change > 0.005 {
        1
    } else if weighted_change < -0.005 {
        -1
    } else {
        0
    };

    let consensus_score = 0.85; // 简化

    EnsemblePrediction {
        final_direction,
        final_change: weighted_change,
        ensemble_confidence: weighted_confidence,
        model_predictions: predictions,
        consensus_score,
        strategy_used: "混合策略".to_string(),
        risk_assessment: RiskAssessment {
            risk_level: if weighted_confidence > 0.8 { "低".to_string() }
                       else if weighted_confidence > 0.6 { "中".to_string() }
                       else { "高".to_string() },
            uncertainty_score: 1.0 - weighted_confidence,
            model_disagreement: 0.1,
            market_volatility: 0.02,
            recommendation: "根据实际情况决策".to_string(),
        },
    }
}

fn calculate_advanced_features(
    _prices: &[f64],
    _volumes: &[i64],
    _highs: &[f64],
    _lows: &[f64],
    _opens: &[f64],
) -> AdvancedFeatures {
    // 返回模拟数据
    AdvancedFeatures {
        momentum_features: MomentumFeatures {
            roc_5: 0.029,
            roc_10: 0.095,
            roc_20: 0.295,
            momentum_7: 8.5,
            momentum_14: 15.5,
            momentum_28: 29.5,
            rsi_divergence: 0.0,
            macd_momentum: 0.0,
            acceleration: 0.0005,
            jerk: 0.0002,
        },
        volatility_features: VolatilityFeatures {
            realized_volatility: 0.28,
            parkinson_volatility: 0.26,
            garman_klass_volatility: 0.29,
            volatility_ratio: 1.15,
            volatility_trend: 0.05,
            volatility_persistence: 0.5,
            atr_normalized: 0.025,
            volatility_regime: 0,
        },
        statistical_features: StatisticalFeatures {
            returns_mean: 0.0095,
            returns_std: 0.015,
            returns_skewness: 0.25,
            returns_kurtosis: 0.8,
            hurst_exponent: 0.68,
            autocorr_1: 0.15,
            autocorr_5: 0.08,
            entropy: 2.1,
            fractal_dimension: 1.32,
        },
        price_volume_features: PriceVolumeFeatures {
            vwap: 122.5,
            volume_price_correlation: 0.65,
            volume_trend: 0.35,
            volume_breakout: 0.45,
            buying_pressure: 0.62,
            selling_pressure: 0.38,
            volume_momentum: 0.35,
            money_flow_index: 68.5,
            accumulation_distribution: 0.0,
        },
        trend_features: T
