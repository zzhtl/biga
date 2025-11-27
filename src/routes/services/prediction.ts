/**
 * 预测服务
 */

import { invoke } from '@tauri-apps/api/core';
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
  return invoke<ModelInfo[]>('list_stock_prediction_models', { symbol });
}

/**
 * 删除模型
 */
export async function deleteModel(modelId: string): Promise<void> {
  await invoke('delete_stock_prediction_model', { modelId });
}

// =============================================================================
// 训练
// =============================================================================

/**
 * 训练模型
 */
export async function trainModel(request: TrainingRequest): Promise<TrainingResult> {
  return invoke<TrainingResult>('train_stock_prediction_model', { request });
}

/**
 * 使用 Candle 训练模型
 */
export async function trainCandleModel(request: TrainingRequest): Promise<TrainingResult> {
  return invoke<TrainingResult>('train_candle_model', { request });
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
  await invoke('retrain_candle_model', {
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
  return invoke<PredictionResponse>('predict_stock_price', { request });
}

/**
 * 使用 Candle 预测
 */
export async function predictWithCandle(request: PredictionRequest): Promise<PredictionResponse> {
  return invoke<PredictionResponse>('predict_with_candle', { request });
}

/**
 * 简化策略预测
 */
export async function predictSimple(request: PredictionRequest): Promise<PredictionResponse> {
  return invoke<PredictionResponse>('predict_candle_price_simple', { request });
}

/**
 * 专业策略预测
 */
export async function predictProfessional(
  request: PredictionRequest
): Promise<ProfessionalPredictionResponse> {
  return invoke<ProfessionalPredictionResponse>('predict_with_professional_strategy', { request });
}

/**
 * 纯技术分析预测
 */
export async function predictTechnicalOnly(
  request: TechnicalOnlyRequest
): Promise<ProfessionalPredictionResponse> {
  return invoke<ProfessionalPredictionResponse>('predict_with_technical_only', { request });
}

// =============================================================================
// 评估与回测
// =============================================================================

/**
 * 评估模型
 */
export async function evaluateModel(modelId: string): Promise<EvaluationResult> {
  return invoke<EvaluationResult>('evaluate_candle_model', { modelId });
}

/**
 * 执行回测
 */
export async function runBacktest(request: BacktestRequest): Promise<BacktestReport> {
  return invoke<BacktestReport>('run_model_backtest', { request });
}

/**
 * 获取优化建议
 */
export async function getOptimizationSuggestions(
  stockCode: string,
  modelName: string,
  backtestReport: BacktestReport,
  currentFeatures: string[],
  currentConfig: any
): Promise<OptimizationSuggestions> {
  return invoke<OptimizationSuggestions>('get_optimization_suggestions', {
    stockCode,
    modelName,
    backtestReport,
    currentFeatures,
    currentConfig,
  });
}

// =============================================================================
// 多周期分析
// =============================================================================

/**
 * 获取多周期信号
 */
export async function getMultiTimeframeSignals(symbol: string): Promise<MultiTimeframeSignal[]> {
  return invoke<MultiTimeframeSignal[]>('get_multi_timeframe_signals', { symbol });
}

/**
 * 获取最新多周期信号
 */
export async function getLatestMultiTimeframeSignal(
  symbol: string
): Promise<MultiTimeframeSignal | null> {
  return invoke<MultiTimeframeSignal | null>('get_latest_multi_timeframe_signal', { symbol });
}

/**
 * 分析多周期预测价值
 */
export async function analyzeMultiTimeframePredictionValue(
  symbol: string
): Promise<Record<string, number>> {
  return invoke<Record<string, number>>('analyze_multi_timeframe_prediction_value', { symbol });
}

