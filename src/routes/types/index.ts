/**
 * 前端类型定义
 */

// =============================================================================
// 股票基本信息
// =============================================================================

export interface StockInfo {
  symbol: string;
  name: string;
  exchange: string;
}

export interface Stock {
  symbol: string;
  name: string;
  area: string;
  industry: string;
  market: string;
  exchange: string;
  list_date: string;
  act_name: string;
  act_ent_type: string;
}

// =============================================================================
// 历史数据
// =============================================================================

export interface HistoricalData {
  symbol: string;
  date: string;
  open: number;
  close: number;
  high: number;
  low: number;
  volume: number;
  amount: number;
  amplitude: number;
  turnover_rate: number;
  change_percent: number;
  change: number;
}

// =============================================================================
// 实时数据
// =============================================================================

export interface RealtimeData {
  symbol: string;
  name: string;
  date: string;
  close: number;
  volume: number;
  amount: number;
  amplitude: number;
  turnover_rate: number;
  change_percent: number;
  change: number;
}

// =============================================================================
// 预测相关
// =============================================================================

export interface TrainingRequest {
  stock_code: string;
  model_name: string;
  start_date: string;
  end_date: string;
  features: string[];
  target: string;
  prediction_days: number;
  model_type: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  dropout: number;
  train_test_split: number;
}

export interface PredictionRequest {
  stock_code: string;
  model_name?: string;
  prediction_days: number;
  use_candle: boolean;
}

export interface TechnicalOnlyRequest {
  stock_code: string;
  prediction_days: number;
}

export interface Prediction {
  target_date: string;
  predicted_price: number;
  predicted_change_percent: number;
  confidence: number;
  trading_signal?: string;
  signal_strength?: number;
  technical_indicators?: TechnicalIndicatorValues;
  prediction_reason?: string;
  key_factors?: string[];
}

export interface TechnicalIndicatorValues {
  rsi: number;
  macd_histogram: number;
  kdj_j: number;
  cci: number;
  obv_trend: number;
  macd_dif: number;
  macd_dea: number;
  kdj_k: number;
  kdj_d: number;
  macd_golden_cross: boolean;
  macd_death_cross: boolean;
  kdj_golden_cross: boolean;
  kdj_death_cross: boolean;
  kdj_overbought: boolean;
  kdj_oversold: boolean;
}

export interface LastRealData {
  date: string;
  price: number;
  change_percent: number;
}

export interface PredictionResponse {
  predictions: Prediction[];
  last_real_data?: LastRealData;
}

// =============================================================================
// 模型相关
// =============================================================================

export interface ModelInfo {
  id: string;
  name: string;
  stock_code: string;
  created_at: number;
  model_type: string;
  features: string[];
  target: string;
  prediction_days: number;
  accuracy: number;
}

export interface TrainingResult {
  metadata: ModelInfo;
  accuracy: number;
}

export interface EvaluationResult {
  model_id: string;
  model_name: string;
  stock_code: string;
  test_samples: number;
  accuracy: number;
  direction_accuracy: number;
  mse: number;
  mae: number;
  rmse: number;
  evaluation_date: string;
}

// =============================================================================
// 回测相关
// =============================================================================

export interface BacktestRequest {
  stock_code: string;
  model_name?: string;
  start_date: string;
  end_date: string;
  prediction_days: number;
  backtest_interval: number;
}

export interface BacktestReport {
  stock_code: string;
  model_name: string;
  backtest_period: string;
  total_predictions: number;
  overall_price_accuracy: number;
  overall_direction_accuracy: number;
  average_prediction_error: number;
}

// =============================================================================
// 专业分析相关
// =============================================================================

export interface BuySellPoint {
  point_type: string;
  signal_strength: number;
  price_level: number;
  stop_loss: number;
  take_profit: number[];
  risk_reward_ratio: number;
  reasons: string[];
  confidence: number;
  accuracy_rate?: number;
}

export interface SupportResistance {
  support_levels: number[];
  resistance_levels: number[];
  current_position: string;
}

export interface MultiTimeframeSignal {
  date: string;
  daily_trend: string;
  weekly_trend: string;
  monthly_trend: string;
  resonance_level: number;
  resonance_direction: string;
  signal_quality: number;
  buy_signal: boolean;
  sell_signal: boolean;
}

export interface VolumePriceDivergence {
  has_bullish_divergence: boolean;
  has_bearish_divergence: boolean;
  divergence_strength: number;
  warning_message: string;
}

export interface PatternRecognition {
  pattern_type: string;
  is_bullish: boolean;
  reliability: number;
  description: string;
}

export interface VolumeAnalysisInfo {
  volume_trend: string;
  volume_price_sync: boolean;
  accumulation_signal: number;
  obv_trend: string;
}

export interface MultiFactorScore {
  total_score: number;
  trend_score: number;
  volume_price_score: number;
  momentum_score: number;
  pattern_score: number;
  support_resistance_score: number;
  sentiment_score: number;
  volatility_score: number;
  signal: string;
  signal_strength: number;
}

export interface ProfessionalPrediction {
  buy_points: BuySellPoint[];
  sell_points: BuySellPoint[];
  support_resistance: SupportResistance;
  multi_timeframe: MultiTimeframeSignal;
  divergence: VolumePriceDivergence;
  current_advice: string;
  risk_level: string;
  candle_patterns: PatternRecognition[];
  volume_analysis: VolumeAnalysisInfo;
  multi_factor_score: MultiFactorScore;
}

export interface ProfessionalPredictionResponse {
  predictions: PredictionResponse;
  professional_analysis: ProfessionalPrediction;
}

// =============================================================================
// 优化建议相关
// =============================================================================

export interface ImplementationStep {
  step_number: number;
  description: string;
  estimated_time: string;
  difficulty: string;
  expected_improvement: number;
}

export interface OptimizationSuggestions {
  stock_code: string;
  model_name: string;
  feature_optimization: any;
  hyperparameter_optimization: any;
  implementation_steps: ImplementationStep[];
  expected_overall_improvement: number;
}

// =============================================================================
// 分页相关
// =============================================================================

export interface PagedResponse<T> {
  data: T[];
  total: number;
  page: number;
  page_size: number;
}

