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
  ts_code: string;
  list_date: string;
  act_name: string;
  act_ent_type: string;
  category: string;
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
  volume_ratio: number;
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
  history_days?: number;
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
  interval?: PredictionInterval | null;
  stress_interval?: PredictionInterval | null;
}

/** 校准涨跌区间带：方向不可测但波动可测，区间才是诚实的不确定性表达 */
export interface PredictionInterval {
  confidence: number;
  lower_change_percent: number;
  upper_change_percent: number;
  lower_price: number;
  upper_price: number;
  method: string;
  lookback_days: number;
}

export type RiskLevel = 'low' | 'medium' | 'high';
export type RiskCategory =
  | 'data'
  | 'uncertainty'
  | 'volatility'
  | 'trend'
  | 'signal'
  | 'liquidity'
  | 'model';

export interface RiskWarning {
  code: string;
  category: RiskCategory;
  severity: RiskLevel;
  title: string;
  detail: string;
  evidence: string[];
}

export interface RiskMetrics {
  history_samples: number;
  data_staleness_days?: number | null;
  daily_volatility_percent: number;
  volatility_percentile: number;
  interval_80_width_percent?: number | null;
  interval_80_lower_percent?: number | null;
  stress_95_lower_percent?: number | null;
  support_distance_percent?: number | null;
  resistance_distance_percent?: number | null;
  atr_percent?: number | null;
}

export interface RiskSummary {
  level: RiskLevel;
  level_label: string;
  warnings: RiskWarning[];
  metrics: RiskMetrics;
}

export interface PredictionDiagnostics {
  point_estimate_kind: 'historical_unconditional_drift' | 'candle_model' | string;
  point_estimate_note: string;
  uncertainty_method: string;
  risk_summary: RiskSummary;
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
  diagnostics?: PredictionDiagnostics | null;
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
  training_start_date?: string | null;
  training_end_date?: string | null;
  training_samples?: number | null;
  test_samples?: number | null;
  mae?: number | null;
  rmse?: number | null;
}

export interface TrainingResult {
  metadata: ModelInfo;
  accuracy: number;
  test_samples: number;
  mae: number;
  rmse: number;
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
  evaluation_scope: string;
  evaluation_note: string;
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

export interface BacktestEntry {
  prediction_date: string;
  predictions: Prediction[];
  actual_prices: number[];
  actual_changes: number[];
  price_accuracy: number;
  direction_accuracy: number;
  avg_prediction_error: number;
}

export interface DailyAccuracy {
  date: string;
  price_accuracy: number;
  direction_accuracy: number;
  prediction_count: number;
  market_volatility: number;
}

export interface BacktestReport {
  stock_code: string;
  model_name: string;
  backtest_period: string;
  total_predictions: number;
  backtest_entries: BacktestEntry[];
  overall_price_accuracy: number;
  overall_direction_accuracy: number;
  average_prediction_error: number;
  accuracy_trend: number[];
  daily_accuracy: DailyAccuracy[];
  price_error_distribution: number[];
  direction_correct_rate: number;
  volatility_vs_accuracy: Array<[number, number]>;
  rmse: number;
  baseline_direction_accuracy: number;
  direction_edge: number;
  predicted_up_ratio: number;
  actual_up_ratio: number;
  interval_80_samples: number;
  interval_80_coverage: number;
  stress_95_samples: number;
  stress_95_coverage: number;
  average_interval_80_width: number;
  average_stress_95_width: number;
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
  adaptive_score: number;
  confirmation_count: number;
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

export interface OptimizationSuggestions {
  stock_code: string;
  model_name: string;
  suggestions: string[];
  expected_improvement: number;
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

export type View =
  | 'favorites'
  | 'stock'
  | 'list'
  | 'realtime'
  | 'historical'
  | 'settings';

export interface NavTarget {
  view: View;
  symbol?: string;
  name?: string;
  action?: 'history' | 'predict';
}

export type ApiTokenSource = 'keyring' | 'environment' | 'none';

export interface ApiTokenStatus {
  configured: boolean;
  source: ApiTokenSource;
  masked?: string | null;
}

export const REALTIME_SORT_COLUMNS = [
  'symbol',
  'name',
  'volume',
  'amount',
  'change',
  'change_percent',
] as const;

export type RealtimeSortColumn = (typeof REALTIME_SORT_COLUMNS)[number];
export type SortDirection = 'asc' | 'desc';
