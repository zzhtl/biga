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
  | 'discipline'
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

// =============================================================================
// 交易纪律
// =============================================================================

export type DisciplineCategory = 'exit' | 'entry' | 'sizing' | 'portfolio' | 'data';
/** 语义强度递增：warn < blocked < must_reduce < must_sell */
export type DisciplineAction = 'warn' | 'blocked' | 'must_reduce' | 'must_sell';
export type StopBasis = 'fixed' | 'atr' | 'support' | 'trailing';
export type SizingConstraint = 'risk' | 'single_position' | 'total_position' | 'cash';
export type EventResolution = 'pending' | 'complied' | 'violated';

export interface DisciplineItem {
  code: string;
  category: DisciplineCategory;
  severity: RiskLevel;
  action: DisciplineAction;
  title: string;
  detail: string;
  evidence: string[];
  trigger_price: number | null;
  threshold_price: number | null;
  suggested_quantity: number | null;
}

export interface DisciplineRules {
  fixed_stop_pct: number;
  atr_mult: number;
  atr_period: number;
  support_buffer_pct: number;
  max_stop_pct: number;
  trail_arm_pct: number;
  trail_pct: number;
  scale_out_tiers: [number, number];
  scale_out_fraction: number;
  time_stop_bars: number;
  time_stop_min_gain_pct: number;
  max_risk_pct: number;
  min_rr: number;
  max_entry_risk_level: RiskLevel;
  max_single_pct: number;
  max_total_pct: number;
  max_holdings: number;
  cooldown_bars: number;
  breaker_losses: number;
  breaker_bars: number;
  stale_days: number;
}

export interface PositionMetrics {
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  distance_to_stop_percent: number;
  drawdown_from_high_percent: number | null;
  risk_exposure: number;
  holding_trading_days: number;
}

export interface ExitVerdict {
  position_id: string;
  symbol: string;
  event_date: string;
  level: RiskLevel;
  level_label: string;
  action: DisciplineAction;
  effective_stop: number;
  stop_basis: StopBasis;
  stop_raised_to: number | null;
  highest_price: number | null;
  highest_price_date: string | null;
  metrics: PositionMetrics;
  t1_locked: boolean;
  tradable: boolean;
  items: DisciplineItem[];
}

export interface PositionSizing {
  risk_per_share: number;
  risk_budget: number;
  cap_by_risk: number;
  cap_by_single_position: number;
  cap_by_total_position: number;
  cap_by_cash: number;
  binding_constraint: SizingConstraint;
  max_shares: number;
  max_amount: number;
  min_lot: number;
  lot_step: number;
  below_one_lot: boolean;
  evidence: string[];
}

export interface EntryVerdict {
  symbol: string;
  /** 语义是「没有违反纪律」，不是「建议买入」——展示时必须带 disclaimer */
  allowed: boolean;
  level: RiskLevel;
  level_label: string;
  sizing: PositionSizing;
  reward_risk_ratio: number | null;
  items: DisciplineItem[];
  disclaimer: string;
}

export interface AccountView {
  cash: number;
  /** 现金 + 持仓市值，后端现算 */
  total_equity: number;
  total_market_value: number;
  open_position_count: number;
  rules: DisciplineRules;
  updated_at: string;
}

/** 后端 PositionView 用 serde(flatten) 把 Position 各列摊平在顶层 */
export interface PositionRow {
  id: string;
  symbol: string;
  status: 'open' | 'closed';
  open_date: string;
  close_date: string | null;
  cost_price: number;
  quantity: number;
  initial_quantity: number;
  initial_stop: number;
  stop_price: number;
  stop_basis: StopBasis;
  target_price: number | null;
  highest_price: number | null;
  highest_price_date: string | null;
  scale_out_done: number;
  realized_pnl: number;
  /** 1 = 检测到除权/除息，卖出规则已挂起 */
  basis_suspect: number;
  note: string | null;
  name: string;
  last_close: number | null;
  verdict: ExitVerdict | null;
  unavailable_reason: string | null;
}

export interface DisciplineEvent {
  id: string;
  position_id: string | null;
  symbol: string;
  event_date: string;
  rule_code: string;
  severity: RiskLevel;
  action_required: 'exit_all' | 'reduce_half' | 'blocked_entry' | 'warn';
  resolution: EventResolution;
  reason: string | null;
  trigger_close: number;
  evidence_json: string;
  created_at: string;
  resolved_at: string | null;
}

export interface DisciplineBoard {
  account: AccountView;
  positions: PositionRow[];
  pending_events: DisciplineEvent[];
  disclaimer: string;
}

export interface ReplayFill {
  rule_code: string;
  trigger_date: string;
  fill_date: string;
  fill_price: number;
  quantity: number;
  deferred_bars: number;
  defer_reason: string | null;
}

export interface StopStep {
  date: string;
  stop_price: number;
  basis: StopBasis;
}

export interface ReplayOutcome {
  position_id: string;
  symbol: string;
  fills: ReplayFill[];
  /** true = 因跌停/停牌顺延超限，已从统计剔除 */
  unresolved: boolean;
  unresolved_reason: string | null;
  same_as_actual: boolean;
  disciplined_pnl: number;
  actual_pnl: number;
  /** 守纪 − 实际。正数 = 守纪本可少亏或多赚 */
  difference: number;
  stop_path: StopStep[];
}

export interface ReviewRow {
  position_id: string;
  symbol: string;
  name: string;
  open_date: string;
  close_date: string;
  cost_price: number;
  quantity: number;
  outcome: ReplayOutcome;
  violations: DisciplineEvent[];
}

export interface RuleStat {
  rule_code: string;
  violated_count: number;
  difference_total: number;
}

export interface DisciplineReview {
  closed_count: number;
  replayed_count: number;
  unresolved_count: number;
  complied_count: number;
  violated_count: number;
  compliance_rate: number | null;
  actual_pnl_total: number;
  disciplined_pnl_total: number;
  difference_total: number;
  fee_recorded: boolean;
  by_rule: RuleStat[];
  rows: ReviewRow[];
  disclaimer: string;
}
