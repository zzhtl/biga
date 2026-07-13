/**
 * 预测服务
 */

import { invokeCommand } from './core';
import type {
  TrainingRequest,
  TrainingResult,
  PredictionRequest,
  PredictionResponse,
  TechnicalOnlyRequest,
  ModelInfo,
  EvaluationResult,
  BacktestRequest,
  BacktestReport,
  MultiTimeframeSignal,
  ProfessionalPredictionResponse,
  OptimizationSuggestions,
} from '../types';

// =============================================================================
// 模型管理
// =============================================================================

/**
 * 列出模型
 */
export async function listModels(symbol: string): Promise<ModelInfo[]> {
  return invokeCommand<ModelInfo[]>('list_stock_prediction_models', { symbol });
}

/**
 * 删除模型
 */
export async function deleteModel(modelId: string): Promise<void> {
  await invokeCommand('delete_stock_prediction_model', { modelId });
}

// =============================================================================
// 训练
// =============================================================================

/**
 * 训练模型
 */
export async function trainModel(request: TrainingRequest): Promise<TrainingResult> {
  return invokeCommand<TrainingResult>('train_stock_prediction_model', { request });
}

/**
 * 使用 Candle 训练模型
 */
export async function trainCandleModel(request: TrainingRequest): Promise<TrainingResult> {
  return invokeCommand<TrainingResult>('train_candle_model', { request });
}

/**
 * 重新训练模型
 */
export async function retrainModel(
  modelId: string,
  epochs: number,
  batchSize: number,
  learningRate: number
): Promise<void> {
  await invokeCommand('retrain_candle_model', {
    modelId,
    epochs,
    batchSize,
    learningRate,
  });
}

// =============================================================================
// 预测
// =============================================================================

/**
 * 预测股价
 */
export async function predictStockPrice(request: PredictionRequest): Promise<PredictionResponse> {
  return invokeCommand<PredictionResponse>('predict_stock_price', { request });
}

/**
 * 使用 Candle 预测
 */
export async function predictWithCandle(request: PredictionRequest): Promise<PredictionResponse> {
  return invokeCommand<PredictionResponse>('predict_with_candle', { request });
}

/**
 * 简化策略预测
 */
export async function predictSimple(request: PredictionRequest): Promise<PredictionResponse> {
  return invokeCommand<PredictionResponse>('predict_candle_price_simple', { request });
}

/**
 * 专业策略预测
 */
export async function predictProfessional(
  request: PredictionRequest
): Promise<ProfessionalPredictionResponse> {
  return invokeCommand<ProfessionalPredictionResponse>('predict_with_professional_strategy', { request });
}

/**
 * 纯技术分析预测
 */
export async function predictTechnicalOnly(
  request: TechnicalOnlyRequest
): Promise<ProfessionalPredictionResponse> {
  return invokeCommand<ProfessionalPredictionResponse>('predict_with_technical_only', { request });
}

// =============================================================================
// 评估与回测
// =============================================================================

/**
 * 评估模型
 */
export async function evaluateModel(modelId: string): Promise<EvaluationResult> {
  return invokeCommand<EvaluationResult>('evaluate_candle_model', { modelId });
}

/**
 * 执行回测
 */
export async function runBacktest(request: BacktestRequest): Promise<BacktestReport> {
  return invokeCommand<BacktestReport>('run_model_backtest', { request });
}

/**
 * 获取优化建议
 */
export async function getOptimizationSuggestions(
  stockCode: string,
  modelName: string,
  backtestReport: BacktestReport
): Promise<OptimizationSuggestions> {
  return invokeCommand<OptimizationSuggestions>('get_optimization_suggestions', {
    stockCode,
    modelName,
    backtestReport,
  });
}

// =============================================================================
// 多周期分析
// =============================================================================

/**
 * 获取多周期信号
 */
export async function getMultiTimeframeSignals(symbol: string): Promise<MultiTimeframeSignal[]> {
  return invokeCommand<MultiTimeframeSignal[]>('get_multi_timeframe_signals', { symbol });
}

/**
 * 获取最新多周期信号
 */
export async function getLatestMultiTimeframeSignal(
  symbol: string
): Promise<MultiTimeframeSignal | null> {
  return invokeCommand<MultiTimeframeSignal | null>('get_latest_multi_timeframe_signal', { symbol });
}

/**
 * 分析多周期预测价值
 */
export async function analyzeMultiTimeframePredictionValue(
  symbol: string
): Promise<Record<string, number>> {
  return invokeCommand<Record<string, number>>('analyze_multi_timeframe_prediction_value', { symbol });
}
