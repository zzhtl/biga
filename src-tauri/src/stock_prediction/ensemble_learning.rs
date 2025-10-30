//! 集成学习模块 - 通过组合多个模型提升预测准确性
//!
//! 核心策略:
//! 1. 多模型投票：集成不同算法的预测结果
//! 2. 动态权重：根据历史表现调整模型权重
//! 3. 分层策略：结合技术分析、深度学习和统计模型
//! 4. 信心度评估：为每个预测提供可靠性评分

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 单个模型的预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_name: String,
    pub model_type: String,
    pub predicted_direction: i8, // -1: 下跌, 0: 横盘, 1: 上涨
    pub predicted_change: f64,
    pub confidence: f64,
    pub weight: f64,
    pub features_used: Vec<String>,
}

/// 集成预测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    pub final_direction: i8,
    pub final_change: f64,
    pub ensemble_confidence: f64,
    pub model_predictions: Vec<ModelPrediction>,
    pub consensus_score: f64, // 0-1，表示模型间的一致性
    pub strategy_used: String,
    pub risk_assessment: RiskAssessment,
}

/// 风险评估
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub risk_level: String, // "低", "中", "高", "极高"
    pub uncertainty_score: f64,
    pub model_disagreement: f64,
    pub market_volatility: f64,
    pub recommendation: String,
}

/// 模型性能追踪
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

/// 集成学习器配置
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub min_models: usize,
    pub confidence_threshold: f64,
    pub adaptive_weights: bool,
    pub outlier_removal: bool,
}

/// 集成策略
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    WeightedAverage,  // 加权平均
    Voting,           // 投票机制
    Stacking,         // 堆叠泛化
    DynamicSelection, // 动态选择
    Hybrid,           // 混合策略
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

/// 集成多个模型的预测结果
pub fn ensemble_predictions(
    predictions: Vec<ModelPrediction>,
    config: &EnsembleConfig,
) -> EnsemblePrediction {
    if predictions.len() < config.min_models {
        // 如果模型数量不足，使用简单平均
        return simple_ensemble(&predictions);
    }

    // 移除异常值
    let filtered_predictions = if config.outlier_removal {
        remove_outliers(&predictions)
    } else {
        predictions.clone()
    };

    // 根据策略选择集成方法
    match config.strategy {
        EnsembleStrategy::WeightedAverage => weighted_average_ensemble(&filtered_predictions),
        EnsembleStrategy::Voting => voting_ensemble(&filtered_predictions),
        EnsembleStrategy::Stacking => stacking_ensemble(&filtered_predictions),
        EnsembleStrategy::DynamicSelection => dynamic_selection_ensemble(&filtered_predictions),
        EnsembleStrategy::Hybrid => hybrid_ensemble(&filtered_predictions, config),
    }
}

/// 简单集成（平均法）
fn simple_ensemble(predictions: &[ModelPrediction]) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    let avg_change: f64 =
        predictions.iter().map(|p| p.predicted_change).sum::<f64>() / predictions.len() as f64;

    let avg_confidence: f64 =
        predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;

    let final_direction = if avg_change > 0.005 {
        1
    } else if avg_change < -0.005 {
        -1
    } else {
        0
    };

    EnsemblePrediction {
        final_direction,
        final_change: avg_change,
        ensemble_confidence: avg_confidence * 0.7, // 降低置信度因为是简单平均
        model_predictions: predictions.to_vec(),
        consensus_score: calculate_consensus(predictions),
        strategy_used: "简单平均".to_string(),
        risk_assessment: assess_risk(predictions, avg_confidence),
    }
}

/// 加权平均集成
fn weighted_average_ensemble(predictions: &[ModelPrediction]) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    // 计算总权重
    let total_weight: f64 = predictions.iter().map(|p| p.weight * p.confidence).sum();

    // 加权平均变化率
    let weighted_change: f64 = predictions
        .iter()
        .map(|p| p.predicted_change * p.weight * p.confidence)
        .sum::<f64>()
        / total_weight;

    // 加权平均置信度
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

    let consensus = calculate_consensus(predictions);
    let adjusted_confidence = weighted_confidence * (0.8 + 0.2 * consensus);

    EnsemblePrediction {
        final_direction,
        final_change: weighted_change,
        ensemble_confidence: adjusted_confidence,
        model_predictions: predictions.to_vec(),
        consensus_score: consensus,
        strategy_used: "加权平均".to_string(),
        risk_assessment: assess_risk(predictions, adjusted_confidence),
    }
}

/// 投票集成
fn voting_ensemble(predictions: &[ModelPrediction]) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    // 统计各方向的投票
    let mut votes: HashMap<i8, f64> = HashMap::new();
    let mut weighted_changes: HashMap<i8, Vec<f64>> = HashMap::new();

    for pred in predictions {
        let vote_weight = pred.weight * pred.confidence;
        *votes.entry(pred.predicted_direction).or_insert(0.0) += vote_weight;
        weighted_changes
            .entry(pred.predicted_direction)
            .or_insert_with(Vec::new)
            .push(pred.predicted_change);
    }

    // 找出得票最多的方向
    let final_direction = *votes
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(dir, _)| dir)
        .unwrap_or(&0);

    // 计算该方向的平均变化率
    let final_change = if let Some(changes) = weighted_changes.get(&final_direction) {
        changes.iter().sum::<f64>() / changes.len() as f64
    } else {
        0.0
    };

    // 计算投票一致性
    let max_votes = votes
        .values()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);
    let total_votes: f64 = votes.values().sum();
    let vote_consensus = if total_votes > 0.0 {
        max_votes / total_votes
    } else {
        0.0
    };

    let avg_confidence: f64 =
        predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64;

    EnsemblePrediction {
        final_direction,
        final_change,
        ensemble_confidence: avg_confidence * vote_consensus,
        model_predictions: predictions.to_vec(),
        consensus_score: vote_consensus,
        strategy_used: "投票机制".to_string(),
        risk_assessment: assess_risk(predictions, avg_confidence * vote_consensus),
    }
}

/// 堆叠泛化集成（Stacking）
fn stacking_ensemble(predictions: &[ModelPrediction]) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    // 第一层：基础模型预测（已有）
    // 第二层：元学习器 - 使用简单的加权策略作为元学习器

    // 根据模型类型分组
    let mut tech_predictions = Vec::new();
    let mut ml_predictions = Vec::new();
    let mut stat_predictions = Vec::new();

    for pred in predictions {
        if pred.model_type.contains("技术分析") || pred.model_type.contains("指标") {
            tech_predictions.push(pred.clone());
        } else if pred.model_type.contains("深度") || pred.model_type.contains("神经") {
            ml_predictions.push(pred.clone());
        } else {
            stat_predictions.push(pred.clone());
        }
    }

    // 为不同类型的模型分配权重
    let tech_weight = 0.35;
    let ml_weight = 0.45;
    let stat_weight = 0.20;

    let mut stacked_change = 0.0;
    let mut stacked_confidence = 0.0;
    let mut direction_votes: HashMap<i8, f64> = HashMap::new();

    // 技术分析层
    if !tech_predictions.is_empty() {
        let avg_change: f64 = tech_predictions
            .iter()
            .map(|p| p.predicted_change)
            .sum::<f64>()
            / tech_predictions.len() as f64;
        let avg_conf: f64 = tech_predictions.iter().map(|p| p.confidence).sum::<f64>()
            / tech_predictions.len() as f64;
        stacked_change += avg_change * tech_weight;
        stacked_confidence += avg_conf * tech_weight;

        let dir = if avg_change > 0.005 {
            1
        } else if avg_change < -0.005 {
            -1
        } else {
            0
        };
        *direction_votes.entry(dir).or_insert(0.0) += tech_weight;
    }

    // 机器学习层
    if !ml_predictions.is_empty() {
        let avg_change: f64 = ml_predictions
            .iter()
            .map(|p| p.predicted_change)
            .sum::<f64>()
            / ml_predictions.len() as f64;
        let avg_conf: f64 =
            ml_predictions.iter().map(|p| p.confidence).sum::<f64>() / ml_predictions.len() as f64;
        stacked_change += avg_change * ml_weight;
        stacked_confidence += avg_conf * ml_weight;

        let dir = if avg_change > 0.005 {
            1
        } else if avg_change < -0.005 {
            -1
        } else {
            0
        };
        *direction_votes.entry(dir).or_insert(0.0) += ml_weight;
    }

    // 统计模型层
    if !stat_predictions.is_empty() {
        let avg_change: f64 = stat_predictions
            .iter()
            .map(|p| p.predicted_change)
            .sum::<f64>()
            / stat_predictions.len() as f64;
        let avg_conf: f64 = stat_predictions.iter().map(|p| p.confidence).sum::<f64>()
            / stat_predictions.len() as f64;
        stacked_change += avg_change * stat_weight;
        stacked_confidence += avg_conf * stat_weight;

        let dir = if avg_change > 0.005 {
            1
        } else if avg_change < -0.005 {
            -1
        } else {
            0
        };
        *direction_votes.entry(dir).or_insert(0.0) += stat_weight;
    }

    // 找出得票最多的方向
    let final_direction = *direction_votes
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(dir, _)| dir)
        .unwrap_or(&0);

    let consensus = calculate_consensus(predictions);

    EnsemblePrediction {
        final_direction,
        final_change: stacked_change,
        ensemble_confidence: stacked_confidence * (0.85 + 0.15 * consensus),
        model_predictions: predictions.to_vec(),
        consensus_score: consensus,
        strategy_used: "堆叠泛化".to_string(),
        risk_assessment: assess_risk(predictions, stacked_confidence),
    }
}

/// 动态选择集成 - 根据当前市场状况选择最佳模型
fn dynamic_selection_ensemble(predictions: &[ModelPrediction]) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    // 选择置信度最高的前N个模型
    let mut sorted_predictions = predictions.to_vec();
    sorted_predictions.sort_by(|a, b| {
        let score_a = a.confidence * a.weight;
        let score_b = b.confidence * b.weight;
        score_b.partial_cmp(&score_a).unwrap()
    });

    // 只使用前50%的高质量预测
    let top_n = (sorted_predictions.len() / 2).max(1);
    let selected_predictions = &sorted_predictions[..top_n];

    // 对选出的模型进行加权平均
    weighted_average_ensemble(selected_predictions)
}

/// 混合策略集成 - 结合多种方法
fn hybrid_ensemble(
    predictions: &[ModelPrediction],
    _config: &EnsembleConfig,
) -> EnsemblePrediction {
    if predictions.is_empty() {
        return default_prediction();
    }

    // 1. 先使用投票获取方向共识
    let voting_result = voting_ensemble(predictions);

    // 2. 使用加权平均获取精确变化率
    let weighted_result = weighted_average_ensemble(predictions);

    // 3. 使用堆叠获取层次化见解
    let stacking_result = stacking_ensemble(predictions);

    // 4. 综合三种方法的结果
    let final_direction = if voting_result.final_direction == weighted_result.final_direction {
        voting_result.final_direction
    } else {
        stacking_result.final_direction
    };

    let final_change = weighted_result.final_change * 0.5 + stacking_result.final_change * 0.5;

    let consensus = calculate_consensus(predictions);
    let ensemble_confidence = (voting_result.ensemble_confidence * 0.3
        + weighted_result.ensemble_confidence * 0.4
        + stacking_result.ensemble_confidence * 0.3)
        * (0.85 + 0.15 * consensus);

    EnsemblePrediction {
        final_direction,
        final_change,
        ensemble_confidence,
        model_predictions: predictions.to_vec(),
        consensus_score: consensus,
        strategy_used: "混合策略(投票+加权+堆叠)".to_string(),
        risk_assessment: assess_risk(predictions, ensemble_confidence),
    }
}

/// 计算模型间的一致性得分
fn calculate_consensus(predictions: &[ModelPrediction]) -> f64 {
    if predictions.len() < 2 {
        return 1.0;
    }

    // 统计各方向的预测数量
    let mut direction_counts: HashMap<i8, usize> = HashMap::new();
    for pred in predictions {
        *direction_counts
            .entry(pred.predicted_direction)
            .or_insert(0) += 1;
    }

    // 最多的方向占比
    let max_count = direction_counts.values().max().unwrap_or(&0);
    let consensus = *max_count as f64 / predictions.len() as f64;

    // 考虑变化率的标准差
    let mean_change: f64 =
        predictions.iter().map(|p| p.predicted_change).sum::<f64>() / predictions.len() as f64;
    let variance: f64 = predictions
        .iter()
        .map(|p| (p.predicted_change - mean_change).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;
    let std_dev = variance.sqrt();

    // 标准差越小，一致性越高
    let change_consistency = if std_dev < 0.01 {
        1.0
    } else if std_dev < 0.03 {
        0.8
    } else if std_dev < 0.05 {
        0.6
    } else {
        0.4
    };

    // 综合方向一致性和变化率一致性
    (consensus * 0.6 + change_consistency * 0.4)
}

/// 评估预测风险
fn assess_risk(predictions: &[ModelPrediction], confidence: f64) -> RiskAssessment {
    let consensus = calculate_consensus(predictions);

    // 计算模型分歧度
    let mean_change: f64 =
        predictions.iter().map(|p| p.predicted_change).sum::<f64>() / predictions.len() as f64;
    let variance: f64 = predictions
        .iter()
        .map(|p| (p.predicted_change - mean_change).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;
    let disagreement = variance.sqrt();

    // 计算不确定性
    let uncertainty = (1.0 - confidence) * 0.5 + (1.0 - consensus) * 0.3 + disagreement * 2.0;

    // 模拟市场波动率（实际应从历史数据计算）
    let market_volatility = predictions
        .iter()
        .map(|p| p.predicted_change.abs())
        .sum::<f64>()
        / predictions.len() as f64;

    // 综合风险评分
    let risk_score = uncertainty * 0.4 + disagreement * 0.3 + market_volatility * 0.3;

    let (risk_level, recommendation) = if risk_score < 0.3 && confidence > 0.7 {
        ("低".to_string(), "市场信号明确，可考虑积极操作".to_string())
    } else if risk_score < 0.5 && confidence > 0.6 {
        (
            "中".to_string(),
            "市场信号较为明确，建议谨慎操作".to_string(),
        )
    } else if risk_score < 0.7 {
        (
            "高".to_string(),
            "市场信号不确定，建议观望或小仓位操作".to_string(),
        )
    } else {
        (
            "极高".to_string(),
            "市场信号混乱，强烈建议观望，避免重仓操作".to_string(),
        )
    };

    RiskAssessment {
        risk_level,
        uncertainty_score: uncertainty.min(1.0),
        model_disagreement: disagreement,
        market_volatility,
        recommendation,
    }
}

/// 移除异常值
fn remove_outliers(predictions: &[ModelPrediction]) -> Vec<ModelPrediction> {
    if predictions.len() < 3 {
        return predictions.to_vec();
    }

    // 计算变化率的中位数和四分位距
    let mut changes: Vec<f64> = predictions.iter().map(|p| p.predicted_change).collect();
    changes.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1_idx = changes.len() / 4;
    let q3_idx = (changes.len() * 3) / 4;
    let q1 = changes[q1_idx];
    let q3 = changes[q3_idx];
    let iqr = q3 - q1;

    // 使用1.5倍IQR作为异常值阈值
    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    // 过滤异常值
    predictions
        .iter()
        .filter(|p| p.predicted_change >= lower_bound && p.predicted_change <= upper_bound)
        .cloned()
        .collect()
}

/// 默认预测（当没有可用预测时）
fn default_prediction() -> EnsemblePrediction {
    EnsemblePrediction {
        final_direction: 0,
        final_change: 0.0,
        ensemble_confidence: 0.0,
        model_predictions: Vec::new(),
        consensus_score: 0.0,
        strategy_used: "默认".to_string(),
        risk_assessment: RiskAssessment {
            risk_level: "极高".to_string(),
            uncertainty_score: 1.0,
            model_disagreement: 0.0,
            market_volatility: 0.0,
            recommendation: "无可用预测数据，不建议操作".to_string(),
        },
    }
}

/// 更新模型性能统计
pub fn update_model_performance(
    model_name: &str,
    predicted: f64,
    actual: f64,
    performances: &mut HashMap<String, ModelPerformance>,
) {
    let entry = performances
        .entry(model_name.to_string())
        .or_insert_with(|| ModelPerformance {
            model_name: model_name.to_string(),
            recent_accuracy: 0.0,
            direction_accuracy: 0.0,
            price_accuracy: 0.0,
            total_predictions: 0,
            correct_predictions: 0,
            avg_error: 0.0,
            last_updated: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        });

    entry.total_predictions += 1;

    // 方向正确性
    let predicted_dir = if predicted > 0.005 {
        1
    } else if predicted < -0.005 {
        -1
    } else {
        0
    };
    let actual_dir = if actual > 0.005 {
        1
    } else if actual < -0.005 {
        -1
    } else {
        0
    };

    if predicted_dir == actual_dir {
        entry.correct_predictions += 1;
    }

    // 更新准确率
    entry.direction_accuracy = entry.correct_predictions as f64 / entry.total_predictions as f64;

    // 更新价格误差
    let error = (predicted - actual).abs();
    entry.avg_error = (entry.avg_error * (entry.total_predictions - 1) as f64 + error)
        / entry.total_predictions as f64;

    // 价格准确率（基于误差）
    entry.price_accuracy = (1.0 - entry.avg_error.min(1.0)).max(0.0);

    // 综合准确率
    entry.recent_accuracy = entry.direction_accuracy * 0.7 + entry.price_accuracy * 0.3;

    entry.last_updated = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
}

/// 根据历史性能动态调整模型权重
pub fn adjust_model_weights(
    predictions: &mut [ModelPrediction],
    performances: &HashMap<String, ModelPerformance>,
) {
    for pred in predictions.iter_mut() {
        if let Some(perf) = performances.get(&pred.model_name) {
            // 基于历史准确率调整权重
            let accuracy_factor = perf.recent_accuracy;
            let consistency_factor = if perf.total_predictions > 10 {
                1.0
            } else {
                0.8 // 预测次数少的模型降低权重
            };

            // 新权重 = 原权重 * 准确率因子 * 一致性因子
            pred.weight = pred.weight * accuracy_factor * consistency_factor;

            // 归一化到 0.5-1.5 范围
            pred.weight = pred.weight.clamp(0.5, 1.5);
        }
    }

    // 归一化所有权重，使其和为模型数量
    let total_weight: f64 = predictions.iter().map(|p| p.weight).sum();
    let avg_weight = total_weight / predictions.len() as f64;
    for pred in predictions.iter_mut() {
        pred.weight = pred.weight / avg_weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_average_ensemble() {
        let predictions = vec![
            ModelPrediction {
                model_name: "Model1".to_string(),
                model_type: "深度学习".to_string(),
                predicted_direction: 1,
                predicted_change: 0.02,
                confidence: 0.8,
                weight: 1.0,
                features_used: vec![],
            },
            ModelPrediction {
                model_name: "Model2".to_string(),
                model_type: "技术分析".to_string(),
                predicted_direction: 1,
                predicted_change: 0.015,
                confidence: 0.7,
                weight: 0.8,
                features_used: vec![],
            },
        ];

        let result = weighted_average_ensemble(&predictions);
        assert_eq!(result.final_direction, 1);
        assert!(result.final_change > 0.015);
        assert!(result.ensemble_confidence > 0.0);
    }

    #[test]
    fn test_consensus_calculation() {
        let predictions = vec![
            ModelPrediction {
                model_name: "Model1".to_string(),
                model_type: "测试".to_string(),
                predicted_direction: 1,
                predicted_change: 0.02,
                confidence: 0.8,
                weight: 1.0,
                features_used: vec![],
            },
            ModelPrediction {
                model_name: "Model2".to_string(),
                model_type: "测试".to_string(),
                predicted_direction: 1,
                predicted_change: 0.021,
                confidence: 0.8,
                weight: 1.0,
                features_used: vec![],
            },
        ];

        let consensus = calculate_consensus(&predictions);
        assert!(consensus > 0.9); // 高度一致
    }

    #[test]
    fn test_voting_ensemble() {
        let predictions = vec![
            ModelPrediction {
                model_name: "Model1".to_string(),
                model_type: "测试".to_string(),
                predicted_direction: 1,
                predicted_change: 0.02,
                confidence: 0.8,
                weight: 1.0,
                features_used: vec![],
            },
            ModelPrediction {
                model_name: "Model2".to_string(),
                model_type: "测试".to_string(),
                predicted_direction: 1,
                predicted_change: 0.015,
                confidence: 0.7,
                weight: 0.8,
                features_used: vec![],
            },
            ModelPrediction {
                model_name: "Model3".to_string(),
                model_type: "测试".to_string(),
                predicted_direction: -1,
                predicted_change: -0.01,
                confidence: 0.6,
                weight: 0.5,
                features_used: vec![],
            },
        ];

        let result = voting_ensemble(&predictions);
        assert_eq!(result.final_direction, 1); // 多数投上涨
    }
}
