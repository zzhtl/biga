<script lang="ts">
    import { onMount } from 'svelte';
    import { confirm } from '@tauri-apps/plugin-dialog';
    import { BrainCircuit, FlaskConical, History, LoaderCircle, Play, ShieldCheck, Star } from 'lucide-svelte';
    import PredictionRangeChart from './prediction_range_chart.svelte';
    import RiskAlertPanel from './risk_alert_panel.svelte';
    import { errorMessage as readableError, invokeCommand } from '../services';
    import type { PredictionDiagnostics, RiskCategory, RiskLevel, RiskSummary } from '../types';

    // 跨页导航（收藏页等跳转进入）：navSymbol 带入股票代码，navAction="predict" 时自动运行一键综合预测
    export let navSymbol: string | null = null;
    export let navAction: "predict" | null = null;
    export let onNavConsumed: () => void = () => {};

    let stockCode = "";
    let selectedModelName = "";
    let daysToPredict = 5;
    let useExistingModel = true;
    let isTraining = false;
    let isPredicting = false;
    let errorMessage = "";
    
    // 定义接口
    interface ModelInfo {
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
    
    interface TechnicalIndicatorValues {
        rsi: number;
        macd_histogram: number;
        kdj_j: number;
        cci: number;
        obv_trend: number;
        // 新增MACD和KDJ字段
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
    
    // 校准预测区间（由已实现波动率×√H 校准，约80%覆盖；非方向预测）
    interface PredictionInterval {
        lower_price: number;
        upper_price: number;
        lower_change_percent: number;
        upper_change_percent: number;
        confidence: number;
        method: string;
        lookback_days: number;
    }

    interface Prediction {
        target_date: string;
        predicted_price: number;
        predicted_change_percent: number;
        confidence: number;
        trading_signal?: string;
        signal_strength?: number;
        technical_indicators?: TechnicalIndicatorValues;
        prediction_reason?: string;  // 预测理由
        key_factors?: string[];      // 关键因素
        interval?: PredictionInterval | null;  // 校准预测区间
        stress_interval?: PredictionInterval | null; // 95%压力区间
    }
    
    interface TrainingLog {
        epoch: number;
        loss: string;
        timestamp: string;
        accuracy?: string;
    }
    
    interface ModelComparisonItem {
        name: string;
        type: string;
        training_days: number;
        accuracy: number;
        training_samples?: number | null;
        test_samples?: number | null;
        created_at: string;
    }
    
    // 回测相关接口
    interface BacktestRequest {
        stock_code: string;
        model_name?: string;
        start_date: string;
        end_date: string;
        prediction_days: number;
        backtest_interval: number;
    }
    
    interface BacktestEntry {
        prediction_date: string;
        predictions: Prediction[];
        actual_prices: number[];
        actual_changes: number[];
        price_accuracy: number;
        direction_accuracy: number;
        avg_prediction_error: number;
    }
    
    interface DailyAccuracy {
        date: string;
        price_accuracy: number;
        direction_accuracy: number;
        prediction_count: number;
        market_volatility: number;
    }
    
    interface BacktestReport {
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
    
    // 新增：最新真实数据接口
    interface LastRealData {
        date: string;
        price: number;
        change_percent: number;
    }
    
    // 新增：预测结果返回接口
    interface PredictionResult {
        predictions: Prediction[];
        last_real_data?: {
            date: string;
            price: number;
            change_percent: number;
        };
        diagnostics?: PredictionDiagnostics | null;
    }
    
    // 新增：买卖点信号接口
    interface BuySellPoint {
        point_type: string;
        signal_strength: number;
        price_level: number;
        stop_loss: number;
        take_profit: number[];
        risk_reward_ratio: number;
        reasons: string[];
        confidence: number;
    }
    
    // 新增：支撑压力位接口
    interface SupportResistance {
        support_levels: number[];
        resistance_levels: number[];
        current_position: string;
    }
    
    // 新增：多周期共振接口
    interface MultiTimeframeSignalProf {
        daily_trend: string;
        weekly_trend: string;
        monthly_trend: string;
        resonance_level: number;
        resonance_direction: string;
        signal_quality: number;
    }
    
    // 新增：量价背离接口
    interface VolumePriceDivergence {
        has_bullish_divergence: boolean;
        has_bearish_divergence: boolean;
        divergence_strength: number;
        warning_message: string;
    }
    
    // 新增：专业预测分析接口
    interface ProfessionalPrediction {
        buy_points: BuySellPoint[];
        sell_points: BuySellPoint[];
        support_resistance: SupportResistance;
        multi_timeframe: MultiTimeframeSignalProf;
        divergence: VolumePriceDivergence;
        current_advice: string;
        risk_level: string;
        candle_patterns: any[];
        volume_analysis: any;
        multi_factor_score: {
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
        };
    }
    
    // 新增：专业预测响应接口
    interface ProfessionalPredictionResponse {
        predictions: PredictionResult;
        professional_analysis: ProfessionalPrediction;
    }
    
    // 使用类型
    let modelList: ModelInfo[] = [];
    let modelListRequestSeq = 0;
    let modelSelectionManuallySelected = false;
    let predictions: Prediction[] = [];
    let modelAccuracy: number | null = null;
    let lastRealData: LastRealData | null = null; // 新增：最新真实数据
    let professionalAnalysis: ProfessionalPrediction | null = null; // 新增：专业分析结果
    let predictionDiagnostics: PredictionDiagnostics | null = null;
    let showProfessionalAnalysis = false; // 是否显示专业分析

    // 估值上下文（PE/PB + 最新基本面）——仅参考展示，非收益预测，数据随"刷新"更新
    interface ValuationContext {
        symbol: string;
        pe: number | null;
        pb: number | null;
        circulating_market_cap_yi: number | null;
        report_date: string | null;
        roe: number | null;
        eps: number | null;
        bps: number | null;
        revenue_growth: number | null;
        profit_growth: number | null;
    }
    let valuationContext: ValuationContext | null = null;

    // 一键综合预测报告（comprehensive_predict 返回；prediction 字段复用现有渲染管线，
    // 决策摘要/基准率字段全部为后端透传的引擎既有输出与描述性统计，无新增方向规则）
    interface ComprehensiveReport {
        symbol: string;
        name: string;
        generated_at: string;
        latest_date: string;
        staleness_days: number;
        current_price: number;
        prediction_days: number;
        direction: string;
        signal_strength: number;
        expected_change_percent: number;
        interval: PredictionInterval | null;
        current_advice: string;
        risk_level: string;
        risk_summary: RiskSummary;
        adaptive_score: number;
        buy_point_count: number;
        sell_point_count: number;
        nearest_support: number | null;
        nearest_resistance: number | null;
        key_factors: string[];
        momentum_5d: number | null;
        momentum_20d: number | null;
        momentum_60d: number | null;
        week52_position: number | null;
        up_ratio_20d: number | null;
        up_ratio_60d: number | null;
        up_ratio_250d: number | null;
        avg_daily_change_250d: number | null;
        prediction: ProfessionalPredictionResponse;
        valuation: ValuationContext;
        disclaimer: string;
    }
    let comprehensiveReport: ComprehensiveReport | null = null;
    let isComprehensivePredicting = false;

    // 收藏星标（收藏池，见 stock_favorites 页）
    let isWatched = false;

    // 模型训练参数
    let newModelName = "模型-" + new Date().toISOString().slice(0, 10);
    let modelType = "candle_mlp_horizon"; // 默认使用按预测天数训练的 Candle MLP 模型
    let lookbackDays = 1500; // 默认使用更长真实历史数据
    let trainTestSplit = 0.8;
    let features = ["close", "volume", "change_percent", "ma_trend", "price_position", "volatility", "rsi_signal", "macd_momentum", "ma5", "ma10", "ma20", "rsi", "macd", "bollinger", "stochastic_k", "stochastic_d", "momentum", "kdj_k", "kdj_d", "kdj_j", "cci", "obv", "macd_dif", "macd_dea", "macd_histogram"];
    let epochs = 100; // 训练轮数
    let batchSize = 32; // 批处理大小
    let learningRate = 0.001; // 学习率
    let dropout = 0.2; // Dropout率
    let advancedOptions = false; // 是否显示高级选项

    let trainingProgress = 0;
    let trainingLogs: TrainingLog[] = [];
    let showTrainingLogs = false;
    let modelComparison: ModelComparisonItem[] = [];
    let showModelComparison = false;
    
    // 回测相关变量
    let backtestReport: BacktestReport | null = null;
    let isBacktesting = false;
    let showBacktestReport = false;
    let backtestStartDate = "";
    let backtestEndDate = "";
    let backtestInterval = 7; // 默认每7天进行一次预测
    let backtestMode: "model" | "rule" = "model";

    function getModelTrainingDays(model: ModelInfo): number {
        return model.model_type === "candle_mlp_horizon" ? Math.max(model.prediction_days || 1, 1) : 1;
    }

    function formatModelOption(model: ModelInfo): string {
        return `${model.name} (${getModelTrainingDays(model)}日模型, 测试准确率: ${(model.accuracy * 100).toFixed(2)}%)`;
    }

    function chooseDefaultModel(models: ModelInfo[]): ModelInfo {
        const targetDays = Math.max(daysToPredict || 1, 1);
        const matched = models.filter(model => getModelTrainingDays(model) === targetDays);
        const candidates = matched.length > 0 ? matched : models;

        return candidates.reduce((a, b) => (
            b.accuracy > a.accuracy || (b.accuracy === a.accuracy && b.created_at > a.created_at) ? b : a
        ));
    }
    let expandedEntryIndex: number | null = null; // 展开查看的回测条目索引

    // 预测结果展示：折叠/展开状态
    let showMoreAnalysis = false;            // 更多分析（多周期共振 + 量价背离）默认收起
    let showDailyTable = false;              // 逐日明细表默认收起
    let expandedPredIndex: number | null = null; // 逐日表中展开查看理由/技术指标的行

    // 纯技术分析预测（无需模型）- 默认展示
    let showTechnicalOnly = true;
    let tabManuallySelected = false; // 用户是否手动切过 tab（手动后不再自动跳转）
    let technicalHistoryDays = 1500; // 使用多少天历史数据
    let technicalPredictionDays = 5; // 预测未来多少天
    let isTechnicalPredicting = false;

    function normalizeNumber(value: any, fallback = 0): number {
        // 统一处理后端返回的 number/string/undefined，避免模板渲染阶段直接异常。
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    }

    function normalizedStockCode(): string {
        return stockCode.trim();
    }

    function normalizePrediction(raw: any): Prediction {
        const normalizeInterval = (interval: any): PredictionInterval | undefined =>
            interval
                ? {
                      confidence: normalizeNumber(interval.confidence),
                      lower_change_percent: normalizeNumber(interval.lower_change_percent),
                      upper_change_percent: normalizeNumber(interval.upper_change_percent),
                      lower_price: normalizeNumber(interval.lower_price),
                      upper_price: normalizeNumber(interval.upper_price),
                      method: String(interval.method ?? "realized_volatility_calibrated"),
                      lookback_days: normalizeNumber(interval.lookback_days, 20),
                  }
                : undefined;

        return {
            ...raw,
            target_date: String(raw?.target_date ?? ""),
            predicted_price: normalizeNumber(raw?.predicted_price),
            predicted_change_percent: normalizeNumber(raw?.predicted_change_percent),
            confidence: normalizeNumber(raw?.confidence),
            trading_signal: raw?.trading_signal ?? undefined,
            signal_strength: raw?.signal_strength != null ? normalizeNumber(raw.signal_strength) : undefined,
            prediction_reason: raw?.prediction_reason ?? undefined,
            key_factors: Array.isArray(raw?.key_factors) ? raw.key_factors.map(String) : undefined,
            technical_indicators: raw?.technical_indicators
                ? {
                      rsi: normalizeNumber(raw.technical_indicators.rsi),
                      macd_histogram: normalizeNumber(raw.technical_indicators.macd_histogram),
                      kdj_j: normalizeNumber(raw.technical_indicators.kdj_j),
                      cci: normalizeNumber(raw.technical_indicators.cci),
                      obv_trend: normalizeNumber(raw.technical_indicators.obv_trend),
                      macd_dif: normalizeNumber(raw.technical_indicators.macd_dif),
                      macd_dea: normalizeNumber(raw.technical_indicators.macd_dea),
                      kdj_k: normalizeNumber(raw.technical_indicators.kdj_k),
                      kdj_d: normalizeNumber(raw.technical_indicators.kdj_d),
                      macd_golden_cross: Boolean(raw.technical_indicators.macd_golden_cross),
                      macd_death_cross: Boolean(raw.technical_indicators.macd_death_cross),
                      kdj_golden_cross: Boolean(raw.technical_indicators.kdj_golden_cross),
                      kdj_death_cross: Boolean(raw.technical_indicators.kdj_death_cross),
                      kdj_overbought: Boolean(raw.technical_indicators.kdj_overbought),
                      kdj_oversold: Boolean(raw.technical_indicators.kdj_oversold),
                  }
                : undefined,
            interval: normalizeInterval(raw?.interval),
            stress_interval: normalizeInterval(raw?.stress_interval),
        };
    }

    function normalizeRiskSummary(raw: any): RiskSummary {
        const level: RiskLevel = ["low", "medium", "high"].includes(raw?.level)
            ? raw.level
            : "low";
        const validCategories: RiskCategory[] = [
            "data",
            "uncertainty",
            "volatility",
            "trend",
            "signal",
            "liquidity",
            "model",
        ];
        const metrics = raw?.metrics ?? {};

        return {
            level,
            level_label: String(raw?.level_label ?? (level === "high" ? "高风险" : level === "medium" ? "中风险" : "低风险")),
            warnings: Array.isArray(raw?.warnings)
                ? raw.warnings.map((warning: any) => ({
                      code: String(warning?.code ?? "UNKNOWN"),
                      category: validCategories.includes(warning?.category)
                          ? warning.category
                          : "signal",
                      severity: ["low", "medium", "high"].includes(warning?.severity)
                          ? warning.severity
                          : "medium",
                      title: String(warning?.title ?? "风险提示"),
                      detail: String(warning?.detail ?? ""),
                      evidence: Array.isArray(warning?.evidence) ? warning.evidence.map(String) : [],
                  }))
                : [],
            metrics: {
                history_samples: normalizeNumber(metrics.history_samples),
                data_staleness_days: metrics.data_staleness_days == null ? null : normalizeNumber(metrics.data_staleness_days),
                daily_volatility_percent: normalizeNumber(metrics.daily_volatility_percent),
                volatility_percentile: normalizeNumber(metrics.volatility_percentile),
                interval_80_width_percent: metrics.interval_80_width_percent == null ? null : normalizeNumber(metrics.interval_80_width_percent),
                interval_80_lower_percent: metrics.interval_80_lower_percent == null ? null : normalizeNumber(metrics.interval_80_lower_percent),
                stress_95_lower_percent: metrics.stress_95_lower_percent == null ? null : normalizeNumber(metrics.stress_95_lower_percent),
                support_distance_percent: metrics.support_distance_percent == null ? null : normalizeNumber(metrics.support_distance_percent),
                resistance_distance_percent: metrics.resistance_distance_percent == null ? null : normalizeNumber(metrics.resistance_distance_percent),
                atr_percent: metrics.atr_percent == null ? null : normalizeNumber(metrics.atr_percent),
            },
        };
    }

    function normalizeDiagnostics(raw: any): PredictionDiagnostics | null {
        if (!raw?.risk_summary) return null;
        return {
            point_estimate_kind: String(raw.point_estimate_kind ?? ""),
            point_estimate_note: String(raw.point_estimate_note ?? ""),
            uncertainty_method: String(raw.uncertainty_method ?? ""),
            risk_summary: normalizeRiskSummary(raw.risk_summary),
        };
    }

    function normalizeLastRealData(raw: any): LastRealData {
        return {
            date: String(raw?.date ?? ""),
            price: normalizeNumber(raw?.price),
            change_percent: normalizeNumber(raw?.change_percent),
        };
    }

    function normalizeBuySellPoint(raw: any): BuySellPoint {
        return {
            point_type: String(raw?.point_type ?? ""),
            signal_strength: normalizeNumber(raw?.signal_strength),
            price_level: normalizeNumber(raw?.price_level),
            stop_loss: normalizeNumber(raw?.stop_loss),
            take_profit: Array.isArray(raw?.take_profit) ? raw.take_profit.map(normalizeNumber) : [],
            risk_reward_ratio: normalizeNumber(raw?.risk_reward_ratio),
            reasons: Array.isArray(raw?.reasons) ? raw.reasons.map(String) : [],
            confidence: normalizeNumber(raw?.confidence),
        };
    }

    function normalizeProfessionalPrediction(raw: any): ProfessionalPrediction {
        const supportResistanceRaw = raw?.support_resistance ?? {};
        const multiTimeframeRaw = raw?.multi_timeframe ?? {};
        const divergenceRaw = raw?.divergence ?? {};
        const multiFactorScoreRaw = raw?.multi_factor_score ?? {};

        return {
            buy_points: Array.isArray(raw?.buy_points) ? raw.buy_points.map(normalizeBuySellPoint) : [],
            sell_points: Array.isArray(raw?.sell_points) ? raw.sell_points.map(normalizeBuySellPoint) : [],
            support_resistance: {
                support_levels: Array.isArray(supportResistanceRaw?.support_levels)
                    ? supportResistanceRaw.support_levels.map(normalizeNumber)
                    : [],
                resistance_levels: Array.isArray(supportResistanceRaw?.resistance_levels)
                    ? supportResistanceRaw.resistance_levels.map(normalizeNumber)
                    : [],
                current_position: String(supportResistanceRaw?.current_position ?? ""),
            },
            multi_timeframe: {
                daily_trend: String(multiTimeframeRaw?.daily_trend ?? ""),
                weekly_trend: String(multiTimeframeRaw?.weekly_trend ?? ""),
                monthly_trend: String(multiTimeframeRaw?.monthly_trend ?? ""),
                resonance_level: normalizeNumber(multiTimeframeRaw?.resonance_level),
                resonance_direction: String(multiTimeframeRaw?.resonance_direction ?? ""),
                signal_quality: normalizeNumber(multiTimeframeRaw?.signal_quality),
            },
            divergence: {
                has_bullish_divergence: Boolean(divergenceRaw?.has_bullish_divergence),
                has_bearish_divergence: Boolean(divergenceRaw?.has_bearish_divergence),
                divergence_strength: normalizeNumber(divergenceRaw?.divergence_strength),
                warning_message: String(divergenceRaw?.warning_message ?? ""),
            },
            current_advice: String(raw?.current_advice ?? ""),
            risk_level: String(raw?.risk_level ?? ""),
            candle_patterns: Array.isArray(raw?.candle_patterns) ? raw.candle_patterns : [],
            volume_analysis: raw?.volume_analysis ?? {},
            multi_factor_score: {
                total_score: normalizeNumber(multiFactorScoreRaw?.total_score, 50),
                trend_score: normalizeNumber(multiFactorScoreRaw?.trend_score, 50),
                volume_price_score: normalizeNumber(multiFactorScoreRaw?.volume_price_score, 50),
                momentum_score: normalizeNumber(multiFactorScoreRaw?.momentum_score, 50),
                pattern_score: normalizeNumber(multiFactorScoreRaw?.pattern_score, 50),
                support_resistance_score: normalizeNumber(multiFactorScoreRaw?.support_resistance_score, 50),
                sentiment_score: normalizeNumber(multiFactorScoreRaw?.sentiment_score, 50),
                volatility_score: normalizeNumber(multiFactorScoreRaw?.volatility_score, 50),
                signal: String(multiFactorScoreRaw?.signal ?? "中性"),
                signal_strength: normalizeNumber(multiFactorScoreRaw?.signal_strength, 0.5),
                adaptive_score: normalizeNumber(multiFactorScoreRaw?.adaptive_score, 50),
                confirmation_count: normalizeNumber(multiFactorScoreRaw?.confirmation_count, 0),
            },
        };
    }
    
    // 把专业预测响应灌入现有渲染管线（纯技术分析与一键综合预测共用的提取逻辑）；
    // 返回归一化后的专业分析结果，便于调用方在 TS 窄化下直接使用
    function applyProfessionalResponse(result: ProfessionalPredictionResponse): ProfessionalPrediction | null {
        // 提取预测数据
        if (result.predictions) {
            if (Array.isArray(result.predictions)) {
                predictions = result.predictions.map(normalizePrediction);
            } else if ('predictions' in result.predictions && Array.isArray(result.predictions.predictions)) {
                predictions = result.predictions.predictions.map(normalizePrediction);
                // 提取最新真实数据
                if (result.predictions.last_real_data) {
                    lastRealData = normalizeLastRealData(result.predictions.last_real_data);
                }
                predictionDiagnostics = normalizeDiagnostics(result.predictions.diagnostics);
            }
        }

        // 提取专业分析结果
        if (result.professional_analysis) {
            professionalAnalysis = normalizeProfessionalPrediction(result.professional_analysis);
            showProfessionalAnalysis = true;
        }

        return professionalAnalysis;
    }

    // 一键综合预测：一次调用聚合专业策略预测 + 估值 + 动量/52周位置 + 历史基准率。
    // 纯本地计算不消耗 API 额度；无弹窗，适合跨页自动触发。
    async function runComprehensivePredict() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先选择股票";
            return;
        }

        isComprehensivePredicting = true;
        isTechnicalPredicting = true;
        errorMessage = "";
        predictions = [];
        professionalAnalysis = null;
        predictionDiagnostics = null;
        lastRealData = null;
        showProfessionalAnalysis = false;
        comprehensiveReport = null;

        try {
            const report = await invokeCommand<ComprehensiveReport>('comprehensive_predict', {
                symbol,
                days: technicalPredictionDays,
            });
            comprehensiveReport = report;
            valuationContext = report.valuation;
            applyProfessionalResponse(report.prediction);
        } catch (error) {
            errorMessage = `综合预测失败：${readableError(error, "请稍后重试")}`;
            console.error('综合预测错误:', error);
        } finally {
            isComprehensivePredicting = false;
            isTechnicalPredicting = false;
        }
    }

    // 纯技术分析预测函数
    async function predictWithTechnicalOnly() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先选择股票";
            return;
        }

        isTechnicalPredicting = true;
        errorMessage = "";
        predictions = [];
        professionalAnalysis = null;
        predictionDiagnostics = null;
        lastRealData = null;
        showProfessionalAnalysis = false;
        comprehensiveReport = null;
        loadValuationContext(symbol);

        try {
            const request = {
                stock_code: symbol,
                history_days: technicalHistoryDays,
                prediction_days: technicalPredictionDays
            };

            const result = await invokeCommand<ProfessionalPredictionResponse>('predict_with_technical_only', { request });

            const appliedAnalysis = applyProfessionalResponse(result);

            if (predictions.length > 0 && appliedAnalysis) {
                const totalScoreRaw =
                    appliedAnalysis.multi_factor_score?.adaptive_score ??
                    appliedAnalysis.multi_factor_score?.total_score;
                const totalScore =
                    typeof totalScoreRaw === "number" ? totalScoreRaw : Number(totalScoreRaw);
                const totalScoreText = Number.isFinite(totalScore) ? `${totalScore.toFixed(1)}/100` : "—/100";

                await alert(
                    `✅ 纯技术分析预测成功！\n基于${technicalHistoryDays}天历史数据\n预测未来${technicalPredictionDays}天走势\n\n综合评分: ${totalScoreText}`
                );
            }
        } catch (error) {
            errorMessage = `纯技术分析预测失败：${readableError(error, "请稍后重试")}`;
            console.error('纯技术分析预测错误:', error);
            await alert(errorMessage);
            predictions = [];
            professionalAnalysis = null;
            predictionDiagnostics = null;
            lastRealData = null;
        } finally {
            isTechnicalPredicting = false;
        }
    }

    onMount(async () => {
        try {
            if (navSymbol) {
                // 跨页导航进入：带入股票代码；navAction="predict" 时自动运行一键综合预测
                stockCode = navSymbol;
                await handleStockCodeChange();
                if (navAction === "predict") {
                    tabManuallySelected = true;
                    showTechnicalOnly = true;
                    showBacktestReport = false;
                    await runComprehensivePredict();
                }
                onNavConsumed();
            } else if (stockCode) {
                // 如果用户选择了股票代码，尝试加载模型列表
                await loadModelList();
                loadWatchStatus();
            }
        } catch (error) {
            errorMessage = `加载失败：${readableError(error, "请稍后重试")}`;
        }
    });

    async function loadModelList(preferredModelId = "") {
        const requestSeq = ++modelListRequestSeq;
        const symbol = normalizedStockCode();
        if (!symbol) {
            modelList = [];
            selectedModelName = "";
            modelSelectionManuallySelected = false;
            return;
        }
        
        try {
            const models = await invokeCommand<ModelInfo[]>('list_stock_prediction_models', { symbol });
            if (requestSeq !== modelListRequestSeq || symbol !== normalizedStockCode()) {
                return;
            }
            modelList = models;
            if (modelList.length > 0) {
                const preferredExists = preferredModelId !== "" && modelList.some(model => model.id === preferredModelId);
                const currentExists = selectedModelName !== "" && modelList.some(model => model.id === selectedModelName);
                if (preferredExists) {
                    selectedModelName = preferredModelId;
                    modelSelectionManuallySelected = true;
                } else if (modelSelectionManuallySelected && currentExists) {
                    // 保留用户手动指定的模型，避免后台刷新覆盖选择。
                } else {
                    // 默认优先选中训练周期匹配当前预测天数的模型；否则回退到准确率最高的可用模型
                    selectedModelName = chooseDefaultModel(modelList).id;
                    modelSelectionManuallySelected = false;
                }
                // 默认保持综合风险分析；模型仅作为用户主动进入的实验工具。
                if (!tabManuallySelected) {
                    showTechnicalOnly = true;
                    showBacktestReport = false;
                    useExistingModel = false;
                }
            } else if (!tabManuallySelected) {
                // 无可用模型时回退到「纯技术分析」（无需训练、始终可用）
                selectedModelName = "";
                modelSelectionManuallySelected = false;
                showTechnicalOnly = true;
                showBacktestReport = false;
            } else {
                selectedModelName = "";
                modelSelectionManuallySelected = false;
            }
        } catch (error) {
            if (requestSeq !== modelListRequestSeq) {
                return;
            }
            errorMessage = `加载模型列表失败：${readableError(error, "请稍后重试")}`;
            modelList = [];
            selectedModelName = "";
            modelSelectionManuallySelected = false;
        }
    }

    // 训练进度监控
    async function simulateTrainingProgress() {
        trainingProgress = 0;
        trainingLogs = [];
        showTrainingLogs = true;
        
        const progressInterval = setInterval(() => {
            if (trainingProgress < 90) {
                trainingProgress += Math.random() * 15;
                trainingLogs = [...trainingLogs, {
                    epoch: Math.floor(trainingProgress / 90 * epochs),
                    loss: (1.0 - trainingProgress / 100).toFixed(4),
                    timestamp: new Date().toLocaleTimeString()
                }];
            }
        }, 500);
        
        return progressInterval;
    }

    async function trainModel() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先输入股票代码";
            return;
        }

        isTraining = true;
        errorMessage = "";
        trainingProgress = 0;
        
        // 开始进度模拟
        const progressInterval = await simulateTrainingProgress();
        
        try {
            // 计算训练日期范围 - 考虑A股节假日因素
            const endDate = new Date().toISOString().slice(0, 10);
            // 使用训练窗口 + 30天节假日缓冲期
            const totalDays = lookbackDays + 30;
            const startDateObj = new Date(Date.now() - totalDays * 24 * 60 * 60 * 1000);
            const startDate = startDateObj.toISOString().slice(0, 10);

            console.log(`📅 训练数据范围: ${startDate} 到 ${endDate} (${totalDays}天，含节假日缓冲)`);

            const trainRequest = {
                stock_code: symbol,
                model_name: newModelName,
                start_date: startDate,
                end_date: endDate,
                features: features,
                target: "close",
                prediction_days: daysToPredict,
                model_type: modelType,
                epochs: epochs,
                batch_size: batchSize,
                learning_rate: learningRate,
                dropout: dropout,
                train_test_split: trainTestSplit
            };

            const result = await invokeCommand<{metadata: ModelInfo, accuracy: number, test_samples: number, mae: number, rmse: number}>('train_candle_model', { request: trainRequest });
            
            clearInterval(progressInterval);
            trainingProgress = 100;
            
            const metadata = result.metadata;
            modelAccuracy = result.accuracy;
            
            // 添加训练完成日志
            trainingLogs = [...trainingLogs, {
                epoch: epochs,
                loss: `MAE ${result.mae.toFixed(3)} / RMSE ${result.rmse.toFixed(3)}`,
                timestamp: new Date().toLocaleTimeString(),
                accuracy: (modelAccuracy * 100).toFixed(2) + "%"
            }];
            
            await loadModelList(metadata.id);
            useExistingModel = true;
            alert(`模型训练成功: ${metadata.name}\n测试样本: ${result.test_samples}\n方向准确率: ${(modelAccuracy * 100).toFixed(2)}%\nMAE: ${result.mae.toFixed(3)}\nRMSE: ${result.rmse.toFixed(3)}`);
            
            // 自动加载模型对比数据
            await loadModelComparison();
            
        } catch (error) {
            clearInterval(progressInterval);
            errorMessage = `训练失败：${readableError(error, "请检查训练参数")}`;
        } finally {
            isTraining = false;
        }
    }

    // 加载模型性能对比数据
    async function loadModelComparison() {
        const symbol = normalizedStockCode();
        if (!symbol) return;
        
        try {
            const models = await invokeCommand<ModelInfo[]>('list_stock_prediction_models', { symbol });
            modelComparison = models.map((model: ModelInfo) => ({
                name: model.name,
                type: model.model_type,
                training_days: getModelTrainingDays(model),
                accuracy: model.accuracy * 100,
                training_samples: model.training_samples,
                test_samples: model.test_samples,
                created_at: new Date(model.created_at < 1000000000000 ? model.created_at * 1000 : model.created_at).toLocaleString('zh-CN', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit'
                })
            }));
            showModelComparison = true;
        } catch (error) {
            console.error("加载模型对比数据失败:", error);
        }
    }

    async function predictStock() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!selectedModelName && useExistingModel) {
            errorMessage = "请先选择模型或训练新模型";
            return;
        }
        
        isPredicting = true;
        errorMessage = "";
        professionalAnalysis = null; // 重置专业分析数据，避免上次技术分析结果残留到摘要卡
        predictionDiagnostics = null;
        showProfessionalAnalysis = false; // 重置专业分析显示
        comprehensiveReport = null; // 重置综合报告，避免补充卡与新预测口径不一致
        loadValuationContext(symbol);

        try {
            const request = {
                stock_code: symbol,
                model_name: useExistingModel ? selectedModelName : null,
                prediction_days: daysToPredict,
                use_candle: true
            };

            const result = await invokeCommand<Prediction[] | PredictionResult>('predict_with_candle', { request });
            
            // 处理返回结果，提取预测数据和最新真实数据
            if (result) {
                if (Array.isArray(result)) {
                    // 旧格式，只返回预测数组
                    predictions = result.map(normalizePrediction);
                } else if ('predictions' in result && Array.isArray(result.predictions)) {
                    // 新格式，包含预测和最新真实数据
                    predictions = result.predictions.map(normalizePrediction);
                    if (result.last_real_data) {
                        lastRealData = normalizeLastRealData(result.last_real_data);
                    }
                    predictionDiagnostics = normalizeDiagnostics(result.diagnostics);
                }
            } else {
                predictions = [];
                lastRealData = null;
            }
            
            if (!predictions || predictions.length === 0) {
                console.warn("预测结果为空，无法生成图表");
            }
            
        } catch (error) {
            console.error("预测失败:", error);
            errorMessage = `预测失败：${readableError(error, "请稍后重试")}`;
            predictions = [];
            lastRealData = null;
            predictionDiagnostics = null;
        } finally {
            isPredicting = false;
        }
    }
    
    // 新增：使用专业策略预测
    async function predictWithProfessionalStrategy() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!selectedModelName && useExistingModel) {
            errorMessage = "请先选择模型或训练新模型";
            return;
        }
        
        isPredicting = true;
        errorMessage = "";
        professionalAnalysis = null;
        predictionDiagnostics = null;
        showProfessionalAnalysis = false;
        comprehensiveReport = null;
        loadValuationContext(symbol);

        try {
            const request = {
                stock_code: symbol,
                model_name: useExistingModel ? selectedModelName : null,
                prediction_days: daysToPredict,
                use_candle: true
            };

            const result = await invokeCommand<ProfessionalPredictionResponse>('predict_with_professional_strategy', { request });
            
            if (result) {
                // 提取预测数据
                if (result.predictions) {
                    if (Array.isArray(result.predictions)) {
                        predictions = result.predictions.map(normalizePrediction);
                    } else if ('predictions' in result.predictions && Array.isArray(result.predictions.predictions)) {
                        predictions = result.predictions.predictions.map(normalizePrediction);
                        if (result.predictions.last_real_data) {
                            lastRealData = normalizeLastRealData(result.predictions.last_real_data);
                        }
                        predictionDiagnostics = normalizeDiagnostics(result.predictions.diagnostics);
                    }
                }
                
                // 提取专业分析结果
                if (result.professional_analysis) {
                    professionalAnalysis = normalizeProfessionalPrediction(result.professional_analysis);
                    showProfessionalAnalysis = true;
                }
                
            }
            
        } catch (error) {
            console.error("专业预测失败:", error);
            errorMessage = `专业预测失败：${readableError(error, "请稍后重试")}`;
            predictions = [];
            lastRealData = null;
            professionalAnalysis = null;
            predictionDiagnostics = null;
        } finally {
            isPredicting = false;
        }
    }
    
    // 拉取估值上下文（PE/PB + 最新基本面）。失败不影响预测主流程，仅置空面板。
    async function loadValuationContext(symbol: string) {
        try {
            valuationContext = await invokeCommand<ValuationContext>('get_valuation_context', { symbol });
        } catch (error) {
            console.warn("获取估值上下文失败:", error);
            valuationContext = null;
        }
    }

    async function handleStockCodeChange() {
        selectedModelName = "";
        modelSelectionManuallySelected = false;
        await loadModelList();
        loadWatchStatus();
    }

    // ===== 收藏星标（与收藏池联动）=====
    // 与后端 canonical_symbol 同口径：提取到恰好 6 位数字则用之，否则原样 trim
    function sixDigitCode(value: string): string {
        const digits = value.replace(/\D/g, "");
        return digits.length === 6 ? digits : value.trim();
    }

    async function loadWatchStatus() {
        try {
            const symbols = await invokeCommand<string[]>("get_watchlist_symbols");
            isWatched = symbols.includes(sixDigitCode(stockCode));
        } catch (error) {
            console.warn("获取收藏状态失败:", error);
        }
    }

    async function toggleWatch() {
        const symbol = normalizedStockCode();
        if (!symbol) return;
        try {
            await invokeCommand(isWatched ? "remove_from_watchlist" : "add_to_watchlist", { symbol });
            isWatched = !isWatched;
        } catch (error) {
            errorMessage = `收藏操作失败：${readableError(error, "请稍后重试")}`;
        }
    }

    function selectModel(modelId: string) {
        selectedModelName = modelId;
        modelSelectionManuallySelected = true;
    }

    function handlePredictionDaysChange() {
        if (!modelSelectionManuallySelected && modelList.length > 0) {
            selectedModelName = chooseDefaultModel(modelList).id;
        }
    }

    // ===== 核心结论（展示在预测结果最顶部，先给实质判断）=====
    // 预测区间累计涨跌：优先用最新真实价到末日预测价的累计涨幅，无真实价时退化为逐日涨跌之和
    $: predHorizonChange =
        predictions.length && lastRealData && lastRealData.price > 0
            ? ((predictions[predictions.length - 1].predicted_price - lastRealData.price) / lastRealData.price) * 100
            : predictions.reduce((s, p) => s + p.predicted_change_percent, 0);
    $: predAvgConfidence = predictions.length
        ? (predictions.reduce((s, p) => s + p.confidence, 0) / predictions.length) * 100
        : 0;
    // 校准区间带（约80%覆盖）：末日预测的累计涨跌区间，作为诚实的主不确定性表达（方向不可测但波动可测）
    $: predHorizonBand =
        predictions.length && predictions[predictions.length - 1].interval
            ? predictions[predictions.length - 1].interval
            : null;
    $: predStressBand =
        predictions.length && predictions[predictions.length - 1].stress_interval
            ? predictions[predictions.length - 1].stress_interval
            : null;
    $: technicalSignal = predictions.length
        ? predictions[predictions.length - 1].trading_signal ?? '中性'
        : '中性';
    $: conclusionDirection = /涨|多|买/.test(technicalSignal)
        ? 'bull'
        : /跌|空|卖/.test(technicalSignal)
          ? 'bear'
          : 'neutral';
    $: conclusionLabel =
        conclusionDirection === 'bull' ? '技术偏多' : conclusionDirection === 'bear' ? '技术偏空' : '技术中性';
    $: conclusionAdvice =
        professionalAnalysis?.current_advice ||
        '该结果为模型实验输出，请结合朴素基准、走步回测和校准区间判断';
    $: conclusionRisk =
        predictionDiagnostics?.risk_summary.level_label || professionalAnalysis?.risk_level || '';
    $: buyPointsCount = professionalAnalysis?.buy_points?.length ?? 0;
    $: sellPointsCount = professionalAnalysis?.sell_points?.length ?? 0;
    // 摘要卡：当前价 → 目标价（预测区间末日价）
    $: targetPrice = predictions.length ? predictions[predictions.length - 1].predicted_price : 0;
    // 关键价位：当前价下方最近的支撑、上方最近的压力（无符合项时回退到极值）
    $: currentPrice = lastRealData?.price ?? 0;
    $: nearestSupport = (() => {
        const levels = professionalAnalysis?.support_resistance?.support_levels ?? [];
        if (!levels.length) return null;
        const below = levels.filter(l => l < currentPrice);
        return below.length ? Math.max(...below) : Math.min(...levels);
    })();
    $: nearestResistance = (() => {
        const levels = professionalAnalysis?.support_resistance?.resistance_levels ?? [];
        if (!levels.length) return null;
        const above = levels.filter(l => l > currentPrice);
        return above.length ? Math.min(...above) : Math.max(...levels);
    })();

    async function deleteModel(modelId: string) {
        // 使用 Tauri 对话框进行确认
        const confirmed = await confirm('确定要删除此模型吗？', { title: '删除模型' });
        if (!confirmed) {
            return;
        }
        try {
            await invokeCommand('delete_stock_prediction_model', { modelId });
            await loadModelList();
            alert('模型删除成功');
        } catch (error) {
            errorMessage = `删除失败：${readableError(error, "请稍后重试")}`;
        }
    }
    
    // 用于重新训练选定模型
    async function retrainModel(modelId: string, modelName: string) {
        // 先刷新模型列表，确保模型ID是最新的
        await loadModelList();
        
        // 检查模型是否还存在
        const modelExists = modelList.some(m => m.id === modelId);
        if (!modelExists) {
            errorMessage = `模型 ${modelName} 已不存在，可能已被删除。请刷新页面重试。`;
            await alert(errorMessage);
            return;
        }
        
        // 确认重训练
        const confirmed = await confirm(
            `确定要重新训练模型 "${modelName}" 吗？\n\n训练参数:\n- 数据窗口: 最近800个交易日\n- 训练轮数: ${epochs}\n- 批次大小: ${batchSize}\n- 学习率: ${learningRate}`,
            { title: '重新训练模型' }
        );
        
        if (!confirmed) {
            return;
        }
        
        isTraining = true;
        errorMessage = "";
        
        try {
            await invokeCommand('retrain_candle_model', {
                modelId,
                epochs: epochs,
                batchSize: batchSize,
                learningRate: learningRate
            });
            await alert(`模型 ${modelName} 重新训练成功！`);
            await loadModelList();
        } catch (error) {
            errorMessage = `重新训练失败：${readableError(error, "请稍后重试")}`;
            await alert(errorMessage);
        } finally {
            isTraining = false;
        }
    }

    // 获取模型评估信息
    async function evaluateModel(modelId: string) {
        try {
            const result = await invokeCommand<{accuracy: number, test_samples: number, mae: number, rmse: number, evaluation_scope?: string, evaluation_note?: string}>('evaluate_candle_model', { modelId });
            const model = modelList.find(m => m.id === modelId);
            const trainingDays = model ? getModelTrainingDays(model) : null;
            const horizonLine = trainingDays ? `\n训练周期: ${trainingDays}日` : "";
            const scope = result.evaluation_scope || "最近历史样本评估";
            const noteLine = result.evaluation_note ? `\n说明: ${result.evaluation_note}` : "";
            
            alert(`${scope}:${horizonLine}\n测试样本: ${result.test_samples}\n方向准确率: ${(result.accuracy * 100).toFixed(2)}%\nMAE: ${result.mae.toFixed(3)}\nRMSE: ${result.rmse.toFixed(3)}${noteLine}`);
        } catch (error) {
            errorMessage = `评估失败：${readableError(error, "请稍后重试")}`;
        }
    }
    
    // 切换高级选项显示
    function toggleAdvancedOptions() {
        advancedOptions = !advancedOptions;
    }
    
    // 执行回测
    async function runBacktest() {
        const symbol = normalizedStockCode();
        if (!symbol) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!backtestStartDate || !backtestEndDate) {
            errorMessage = "请选择回测日期范围";
            return;
        }

        if (backtestMode === "model" && !selectedModelName) {
            errorMessage = "请先选择模型或切换为规则引擎回测";
            return;
        }
        
        isBacktesting = true;
        errorMessage = "";
        
        try {
            const backtestRequest: BacktestRequest = {
                stock_code: symbol,
                model_name: backtestMode === "model" ? selectedModelName : undefined,
                start_date: backtestStartDate,
                end_date: backtestEndDate,
                prediction_days: daysToPredict,
                backtest_interval: backtestInterval
            };
            
            const result = await invokeCommand<BacktestReport>('run_model_backtest', { request: backtestRequest });
            backtestReport = result;
            showBacktestReport = true;
            
        } catch (error) {
            errorMessage = `回测失败：${readableError(error, "请检查回测条件")}`;
            console.error("回测失败:", error);
        } finally {
            isBacktesting = false;
        }
    }
    
    // 设置默认回测日期（最近3个月）
    function setDefaultBacktestDates() {
        const endDate = new Date();
        const startDate = new Date();
        startDate.setMonth(endDate.getMonth() - 3);
        
        backtestEndDate = endDate.toISOString().slice(0, 10);
        backtestStartDate = startDate.toISOString().slice(0, 10);
    }
</script>

<div class="container">
    <header class="page-header">
        <div>
            <h1>预测与风险工作台</h1>
            <p>以风险区间和回测证据为主，模型输出仅作为辅助观察。</p>
        </div>
    </header>

    <div class="input-group">
        <input
            type="text"
            placeholder="输入股票代码，例如 sh000001"
            bind:value={stockCode}
            on:change={handleStockCodeChange}
            class="search-input"
            aria-label="股票代码"
        />
        <button
            class="watch-star"
            class:watched={isWatched}
            on:click={toggleWatch}
            title={isWatched ? "移出收藏池" : "加入收藏池"}
            aria-label={isWatched ? "移出收藏池" : "加入收藏池"}
        >
            <Star size={19} fill={isWatched ? "currentColor" : "none"} aria-hidden="true" />
        </button>
    </div>
    
    {#if errorMessage}
        <div class="error-message" role="alert">
            {errorMessage}
        </div>
    {/if}

    {#if isComprehensivePredicting}
        <div class="comprehensive-loading"><LoaderCircle size={17} class="spin" aria-hidden="true" />正在生成综合风险报告…</div>
    {/if}

    <div class="tabs">
        <button class:active={showTechnicalOnly} on:click={() => {tabManuallySelected = true; showTechnicalOnly = true; showBacktestReport = false; useExistingModel = false;}}>
            <ShieldCheck size={17} aria-hidden="true" />综合风险分析
        </button>
        <button class:active={!showBacktestReport && !showTechnicalOnly} on:click={() => {tabManuallySelected = true; useExistingModel = true; showBacktestReport = false; showTechnicalOnly = false;}}>
            <FlaskConical size={17} aria-hidden="true" />模型实验
        </button>
        <button class:active={showBacktestReport} on:click={() => {tabManuallySelected = true; showBacktestReport = true; showTechnicalOnly = false; setDefaultBacktestDates();}}>
            <History size={17} aria-hidden="true" />回测验证
        </button>
    </div>

    {#if !showTechnicalOnly && !showBacktestReport}
        <div class="model-lab-switch" aria-label="模型实验模式">
            <button class:active={useExistingModel} on:click={() => useExistingModel = true}>
                <BrainCircuit size={16} aria-hidden="true" />模型预测
            </button>
            <button class:active={!useExistingModel} on:click={() => useExistingModel = false}>
                <FlaskConical size={16} aria-hidden="true" />训练模型
            </button>
        </div>
    {/if}
    
    {#if showTechnicalOnly}
        <section class="analysis-toolbar">
            <div class="analysis-heading">
                <div>
                    <h2>综合风险分析</h2>
                    <span>区间与风险为主，技术信号仅描述当前状态</span>
                </div>
                <div class="horizon-control">
                    <label for="analysis-days">预测周期</label>
                    <div class="horizon-presets">
                        {#each [1, 5, 10, 20] as days}
                            <button class:active={technicalPredictionDays === days} on:click={() => technicalPredictionDays = days}>{days}日</button>
                        {/each}
                    </div>
                    <input id="analysis-days" type="number" bind:value={technicalPredictionDays} min="1" max="30" aria-label="自定义预测天数" />
                </div>
            </div>
            <div class="analysis-action">
                <button
                    on:click={runComprehensivePredict}
                    class:loading={isComprehensivePredicting}
                    disabled={isComprehensivePredicting || !stockCode}
                    class="predict-btn"
                >
                    {#if isComprehensivePredicting}
                        <LoaderCircle size={18} class="spin" aria-hidden="true" />分析中
                    {:else}
                        <Play size={18} aria-hidden="true" />生成综合报告
                    {/if}
                </button>
            </div>
        </section>
    {:else if useExistingModel && !showBacktestReport}
        <div class="model-section">
            <h2>选择预测模型</h2>
            {#if modelList.length === 0}
                <p>没有可用的模型，请先训练一个模型。</p>
            {:else}
                <div class="model-list">
                    {#each modelList as model}
                        <div class="model-item" class:selected={selectedModelName === model.id}>
                            <div class="model-info" role="button" tabindex="0" on:click={() => selectModel(model.id)} on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectModel(model.id); } }}>
                                <h3>{model.name}</h3>
                                <div class="model-details">
                                    <span>类型：{model.model_type}</span>
                                    <span>训练周期：{getModelTrainingDays(model)}日</span>
                                    <span>测试集方向准确率：{(model.accuracy * 100).toFixed(2)}%（朴素基准约50–56%，低于基准即无方向 edge）</span>
                                    {#if model.training_samples != null}
                                        <span>训练样本：{model.training_samples}</span>
                                    {/if}
                                    {#if model.test_samples != null}
                                        <span>测试样本：{model.test_samples}</span>
                                    {/if}
                                    <span>创建时间：{new Date(model.created_at < 1000000000000 ? model.created_at * 1000 : model.created_at).toLocaleString('zh-CN', {
                                        year: 'numeric',
                                        month: '2-digit',
                                        day: '2-digit',
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    })}</span>
                                </div>
                            </div>
                            <div class="model-actions">
                                <button type="button" class="action-btn" on:click={() => evaluateModel(model.id)}>评估</button>
                                <button type="button" class="action-btn" on:click={() => retrainModel(model.id, model.name)}>重训练</button>
                                <button type="button" class="delete-btn" on:click={() => deleteModel(model.id)}>删除</button>
                            </div>
                        </div>
                    {/each}
                </div>
            {/if}
            
            <div class="prediction-settings">
                <label>
                    预测天数:
                    <input type="number" bind:value={daysToPredict} min="1" max="30" on:change={handlePredictionDaysChange} />
                </label>
                
                <button
                    on:click={predictStock}
                    class:loading={isPredicting}
                    disabled={isPredicting || modelList.length === 0}
                >
                    {#if isPredicting}
                        <span class="spinner"></span>
                    {:else}
                        快速预测
                    {/if}
                </button>
                
                <button
                    on:click={predictWithProfessionalStrategy}
                    class:loading={isPredicting}
                    disabled={isPredicting || modelList.length === 0}
                    class="professional-predict-btn"
                >
                    {#if isPredicting}
                        <span class="spinner"></span>
                    {:else}
                        <BrainCircuit size={17} aria-hidden="true" />模型 + 风险分析
                    {/if}
                </button>
            </div>
        </div>
    {:else if !useExistingModel && !showBacktestReport}
        <div class="model-section">
            <h2>训练新模型 (Candle)</h2>
            
            <div class="training-form">
                <div class="form-group">
                    <label for="model-name">模型名称:</label>
                    <input id="model-name" type="text" bind:value={newModelName} />
                </div>
                
                <div class="form-group">
                    <label for="model-type">模型类型:</label>
                    <select id="model-type" bind:value={modelType}>
                        <option value="candle_mlp_horizon">Candle多层感知机（按预测天数训练）</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="lookback-days">历史窗口天数 (实际查询范围+30天节假日缓冲):</label>
                    <input id="lookback-days" type="number" bind:value={lookbackDays} min="120" max="3000" step="100" />
                    <small style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">
                        推荐: 1500天 (约6年交易数据)，实际查询 {lookbackDays + 30} 天
                    </small>
                </div>
                
                <div class="form-group">
                    <label for="train-split">训练/测试集分割比例:</label>
                    <input id="train-split" type="number" bind:value={trainTestSplit} min="0.5" max="0.9" step="0.1" />
                </div>
                
                <button type="button" class="advanced-btn" on:click={toggleAdvancedOptions}>
                    {advancedOptions ? '隐藏高级选项' : '显示高级选项'}
                </button>
                
                {#if advancedOptions}
                    <div class="advanced-options">
                        <div class="form-group">
                            <label for="training-epochs">训练轮数(Epochs):</label>
                            <input id="training-epochs" type="number" bind:value={epochs} min="10" max="1000" />
                        </div>
                        
                        <div class="form-group">
                            <label for="batch-size">批处理大小(Batch Size):</label>
                            <input id="batch-size" type="number" bind:value={batchSize} min="8" max="128" />
                        </div>
                        
                        <div class="form-group">
                            <label for="learning-rate">学习率(Learning Rate):</label>
                            <input id="learning-rate" type="number" bind:value={learningRate} min="0.0001" max="0.1" step="0.0001" />
                        </div>
                        
                        <div class="form-group">
                            <label for="dropout-rate">Dropout率:</label>
                            <input id="dropout-rate" type="number" bind:value={dropout} min="0" max="0.5" step="0.1" />
                        </div>
                    </div>
                {/if}
                
                <div class="form-group features-list">
                    <span class="form-label">特征选择:</span>
                    <div class="features-checkboxes">
                        <label>
                            <input type="checkbox" value="close" checked={features.includes('close')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'close'];
                                else features = features.filter(f => f !== 'close');
                            }} />
                            收盘价
                        </label>
                        <label>
                            <input type="checkbox" value="volume" checked={features.includes('volume')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'volume'];
                                else features = features.filter(f => f !== 'volume');
                            }} />
                            成交量
                        </label>
                        <label>
                            <input type="checkbox" value="change_percent" checked={features.includes('change_percent')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'change_percent'];
                                else features = features.filter(f => f !== 'change_percent');
                            }} />
                            涨跌幅
                        </label>
                        <label>
                            <input type="checkbox" value="ma5" checked={features.includes('ma5')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'ma5'];
                                else features = features.filter(f => f !== 'ma5');
                            }} />
                            5日均线
                        </label>
                        <label>
                            <input type="checkbox" value="ma10" checked={features.includes('ma10')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'ma10'];
                                else features = features.filter(f => f !== 'ma10');
                            }} />
                            10日均线
                        </label>
                        <label>
                            <input type="checkbox" value="ma20" checked={features.includes('ma20')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'ma20'];
                                else features = features.filter(f => f !== 'ma20');
                            }} />
                            20日均线
                        </label>
                        <label>
                            <input type="checkbox" value="rsi" checked={features.includes('rsi')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'rsi'];
                                else features = features.filter(f => f !== 'rsi');
                            }} />
                            RSI指标
                        </label>
                        <label>
                            <input type="checkbox" value="macd" checked={features.includes('macd')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'macd'];
                                else features = features.filter(f => f !== 'macd');
                            }} />
                            MACD指标
                        </label>
                        <label>
                            <input type="checkbox" value="bollinger" checked={features.includes('bollinger')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'bollinger'];
                                else features = features.filter(f => f !== 'bollinger');
                            }} />
                            布林带
                        </label>
                        <label>
                            <input type="checkbox" value="stochastic_k" checked={features.includes('stochastic_k')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'stochastic_k'];
                                else features = features.filter(f => f !== 'stochastic_k');
                            }} />
                            随机K值
                        </label>
                        <label>
                            <input type="checkbox" value="stochastic_d" checked={features.includes('stochastic_d')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'stochastic_d'];
                                else features = features.filter(f => f !== 'stochastic_d');
                            }} />
                            随机D值
                        </label>
                        <label>
                            <input type="checkbox" value="momentum" checked={features.includes('momentum')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'momentum'];
                                else features = features.filter(f => f !== 'momentum');
                            }} />
                            动量指标
                        </label>
                        <label>
                            <input type="checkbox" value="ma_trend" checked={features.includes('ma_trend')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'ma_trend'];
                                else features = features.filter(f => f !== 'ma_trend');
                            }} />
                            均线趋势
                        </label>
                        <label>
                            <input type="checkbox" value="price_position" checked={features.includes('price_position')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'price_position'];
                                else features = features.filter(f => f !== 'price_position');
                            }} />
                            价格位置
                        </label>
                        <label>
                            <input type="checkbox" value="volatility" checked={features.includes('volatility')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'volatility'];
                                else features = features.filter(f => f !== 'volatility');
                            }} />
                            波动率
                        </label>
                        <label>
                            <input type="checkbox" value="rsi_signal" checked={features.includes('rsi_signal')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'rsi_signal'];
                                else features = features.filter(f => f !== 'rsi_signal');
                            }} />
                            RSI信号
                        </label>
                        <label>
                            <input type="checkbox" value="macd_momentum" checked={features.includes('macd_momentum')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'macd_momentum'];
                                else features = features.filter(f => f !== 'macd_momentum');
                            }} />
                            MACD动量
                        </label>
                        <label>
                            <input type="checkbox" value="kdj_k" checked={features.includes('kdj_k')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'kdj_k'];
                                else features = features.filter(f => f !== 'kdj_k');
                            }} />
                            KDJ-K值
                        </label>
                        <label>
                            <input type="checkbox" value="kdj_d" checked={features.includes('kdj_d')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'kdj_d'];
                                else features = features.filter(f => f !== 'kdj_d');
                            }} />
                            KDJ-D值
                        </label>
                        <label>
                            <input type="checkbox" value="kdj_j" checked={features.includes('kdj_j')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'kdj_j'];
                                else features = features.filter(f => f !== 'kdj_j');
                            }} />
                            KDJ-J值
                        </label>
                        <label>
                            <input type="checkbox" value="cci" checked={features.includes('cci')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'cci'];
                                else features = features.filter(f => f !== 'cci');
                            }} />
                            CCI指标
                        </label>
                        <label>
                            <input type="checkbox" value="obv" checked={features.includes('obv')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'obv'];
                                else features = features.filter(f => f !== 'obv');
                            }} />
                            OBV能量潮
                        </label>
                        <label>
                            <input type="checkbox" value="macd_dif" checked={features.includes('macd_dif')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'macd_dif'];
                                else features = features.filter(f => f !== 'macd_dif');
                            }} />
                            MACD-DIF
                        </label>
                        <label>
                            <input type="checkbox" value="macd_dea" checked={features.includes('macd_dea')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'macd_dea'];
                                else features = features.filter(f => f !== 'macd_dea');
                            }} />
                            MACD-DEA
                        </label>
                        <label>
                            <input type="checkbox" value="macd_histogram" checked={features.includes('macd_histogram')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'macd_histogram'];
                                else features = features.filter(f => f !== 'macd_histogram');
                            }} />
                            MACD柱状图
                        </label>
                        <label>
                            <input type="checkbox" value="dmi_plus" checked={features.includes('dmi_plus')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'dmi_plus'];
                                else features = features.filter(f => f !== 'dmi_plus');
                            }} />
                            DMI+DI
                        </label>
                        <label>
                            <input type="checkbox" value="dmi_minus" checked={features.includes('dmi_minus')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'dmi_minus'];
                                else features = features.filter(f => f !== 'dmi_minus');
                            }} />
                            DMI-DI
                        </label>
                        <label>
                            <input type="checkbox" value="adx" checked={features.includes('adx')} on:change={(e) => {
                                const target = e.target as HTMLInputElement;
                                if (target.checked) features = [...features, 'adx'];
                                else features = features.filter(f => f !== 'adx');
                            }} />
                            ADX趋势强度
                        </label>
                    </div>
                </div>
                
                <!-- 训练进度显示 -->
                {#if isTraining}
                    <div class="training-progress">
                        <h3>训练进度</h3>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {trainingProgress}%"></div>
                            <span class="progress-text">{trainingProgress.toFixed(1)}%</span>
                        </div>
                        
                        {#if showTrainingLogs}
                            <div class="training-logs">
                                <h4>训练日志</h4>
                                <div class="logs-container">
                                    {#each trainingLogs as log}
                                        <div class="log-entry">
                                            <span class="log-time">{log.timestamp}</span>
                                            <span class="log-content">
                                                Epoch {log.epoch}: Loss = {log.loss}
                                                {#if log.accuracy}
                                                    | Accuracy = {log.accuracy}
                                                {/if}
                                            </span>
                                        </div>
                                    {/each}
                                </div>
                            </div>
                        {/if}
                    </div>
                {/if}
                
                <button
                    on:click={trainModel}
                    class:loading={isTraining}
                    disabled={isTraining}
                >
                    {#if isTraining}
                        <span class="spinner"></span>
                        训练中...
                    {:else}
                        开始训练
                    {/if}
                </button>
                
                <!-- 模型性能对比 -->
                {#if showModelComparison && modelComparison.length > 0}
                    <div class="model-comparison">
                        <h3>模型性能对比</h3>
                        <div class="comparison-chart">
                            <table class="comparison-table">
                                <thead>
                                    <tr>
                                        <th>模型名称</th>
                                        <th>模型类型</th>
                                        <th>训练周期</th>
                                        <th>训练样本</th>
                                        <th>测试样本</th>
                                        <th>测试集方向准确率</th>
                                        <th>创建时间</th>
                                        <th>性能条</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {#each modelComparison as model}
                                        <tr>
                                            <td>{model.name}</td>
                                            <td>{model.type}</td>
                                            <td>{model.training_days}日</td>
                                            <td>{model.training_samples ?? '-'}</td>
                                            <td>{model.test_samples ?? '-'}</td>
                                            <td>{model.accuracy.toFixed(2)}%</td>
                                            <td>{model.created_at}</td>
                                            <td>
                                                <div class="performance-bar">
                                                    <div 
                                                        class="performance-fill" 
                                                        style="width: {model.accuracy}%; background-color: {model.accuracy > 80 ? '#10b981' : model.accuracy > 60 ? '#f59e0b' : '#ef4444'}"
                                                    ></div>
                                                </div>
                                            </td>
                                        </tr>
                                    {/each}
                                </tbody>
                            </table>
                        </div>
                    </div>
                {/if}
            </div>
        </div>
    {:else if showBacktestReport}
        <div class="model-section">
            <h2>模型回测报告</h2>
            
            <div class="backtest-settings">
                <div class="form-group">
                    <label for="backtest-mode">回测对象:</label>
                    <select id="backtest-mode" bind:value={backtestMode}>
                        <option value="model">已训练模型</option>
                        <option value="rule">规则引擎</option>
                    </select>
                </div>

                {#if backtestMode === "model"}
                    <div class="form-group">
                        <label for="backtest-model">选择模型:</label>
                        <select id="backtest-model" bind:value={selectedModelName} disabled={modelList.length === 0} on:change={() => modelSelectionManuallySelected = true}>
                            <option value="">请选择模型</option>
                            {#each modelList as model}
                                <option value={model.id}>{formatModelOption(model)}</option>
                            {/each}
                        </select>
                    </div>
                {/if}
                
                <div class="form-group">
                    <label for="backtest-start">回测开始日期:</label>
                    <input id="backtest-start" type="date" bind:value={backtestStartDate} />
                </div>
                
                <div class="form-group">
                    <label for="backtest-end">回测结束日期:</label>
                    <input id="backtest-end" type="date" bind:value={backtestEndDate} />
                </div>
                
                <div class="form-group">
                    <label for="backtest-days">预测天数:</label>
                    <input id="backtest-days" type="number" bind:value={daysToPredict} min="1" max="10" on:change={handlePredictionDaysChange} />
                </div>
                
                <div class="form-group">
                    <label for="backtest-interval">回测间隔(天):</label>
                    <input id="backtest-interval" type="number" bind:value={backtestInterval} min="1" max="30" />
                    <small>每隔几天进行一次预测</small>
                </div>
                
                <button
                    on:click={runBacktest}
                    class:loading={isBacktesting}
                    disabled={isBacktesting}
                >
                    {#if isBacktesting}
                        <span class="spinner"></span>
                        回测中...
                    {:else}
                        开始回测
                    {/if}
                </button>
            </div>
            
            {#if backtestReport}
                <div class="backtest-report">
                    <h3>回测结果</h3>
                    
                    <!-- 总体统计 -->
                    <div class="backtest-summary">
                        <div class="summary-card">
                            <h4>真实走步指标</h4>
                            <div class="summary-stats">
                                <div class="stat-item">
                                    <span class="stat-label">回测对象:</span>
                                    <span class="stat-value model-name-value">{backtestReport.model_name}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">方向准确率:</span>
                                    <span class="stat-value">{(backtestReport.overall_direction_accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">朴素基准:</span>
                                    <span class="stat-value">{(backtestReport.baseline_direction_accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">方向 edge:</span>
                                    <span class="stat-value {backtestReport.direction_edge > 0 ? 'positive' : 'negative'}">{backtestReport.direction_edge > 0 ? '+' : ''}{(backtestReport.direction_edge * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">MAE:</span>
                                    <span class="stat-value">{backtestReport.average_prediction_error.toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">RMSE:</span>
                                    <span class="stat-value">{backtestReport.rmse.toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">80%区间覆盖:</span>
                                    <span class="stat-value">{(backtestReport.interval_80_coverage * 100).toFixed(1)}% / {backtestReport.interval_80_samples}样本</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">95%区间覆盖:</span>
                                    <span class="stat-value">{(backtestReport.stress_95_coverage * 100).toFixed(1)}% / {backtestReport.stress_95_samples}样本</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">总样本:</span>
                                    <span class="stat-value">{backtestReport.total_predictions}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {#if backtestReport.direction_edge <= 0}
                        <div class="backtest-warning">
                            当前方向准确率未超过总猜多数方向的朴素基准，暂无方向预测价值证据。
                        </div>
                    {/if}
                    
                    <!-- 详细回测记录 -->
                    <div class="backtest-details">
                        <h4>详细回测记录</h4>
                        <div class="backtest-table">
                            <table>
                                <thead>
                                    <tr>
                                        <th>预测日期</th>
                                        <th>方向结果</th>
                                        <th>绝对误差</th>
                                        <th>预测次数</th>
                                        <th>详情</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {#each backtestReport.backtest_entries as entry, i}
                                        <tr>
                                            <td>{new Date(entry.prediction_date).toLocaleDateString()}</td>
                                            <td class="accuracy-cell">
                                                <span class={entry.direction_accuracy > 0 ? 'chip-correct' : 'chip-wrong'}>{entry.direction_accuracy > 0 ? '命中' : '未命中'}</span>
                                            </td>
                                            <td>{entry.avg_prediction_error.toFixed(2)}%</td>
                                            <td>{entry.predictions.length}</td>
                                            <td>
                                                <button class="action-btn" type="button" on:click={() => expandedEntryIndex = expandedEntryIndex === i ? null : i}>
                                                    {expandedEntryIndex === i ? '收起' : '详情'}
                                                </button>
                                            </td>
                                        </tr>
                                        {#if expandedEntryIndex === i}
                                            <tr class="details-row">
                                                <td colspan="5">
                                                    <div class="entry-details">
                                                        <h5>详细对比（{new Date(entry.prediction_date).toLocaleDateString()} 发起）</h5>
                                                        <div class="entry-table-wrapper">
                                                            <table class="entry-table">
                                                                <thead>
                                                                    <tr>
                                                                        <th>目标日期</th>
                                                                        <th>预测价格</th>
                                                                        <th>实际价格</th>
                                                                        <th>误差</th>
                                                                        <th>方向</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {#each entry.predictions as p, idx}
                                                                        {#if entry.actual_prices && entry.actual_prices[idx] !== undefined}
                                                                            {@const actual = entry.actual_prices[idx]}
                                                                            {@const errPct = actual !== 0 ? Math.abs((p.predicted_price - actual) / actual) * 100 : 0}
                                                                            {@const dirCorrect = (p.predicted_change_percent > 0 && entry.actual_changes[idx] > 0) || (p.predicted_change_percent < 0 && entry.actual_changes[idx] < 0) || (p.predicted_change_percent === 0 && entry.actual_changes[idx] === 0)}
                                                                            <tr>
                                                                                <td>{new Date(p.target_date).toLocaleDateString()}</td>
                                                                                <td>{p.predicted_price.toFixed(2)}</td>
                                                                                <td>{actual.toFixed(2)}</td>
                                                                                <td class={errPct <= 2 ? 'positive' : errPct >= 5 ? 'negative' : ''}>{errPct.toFixed(2)}%</td>
                                                                                <td>
                                                                                    <span class={dirCorrect ? 'chip-correct' : 'chip-wrong'}>{dirCorrect ? '正确' : '错误'}</span>
                                                                                </td>
                                                                            </tr>
                                                                        {/if}
                                                                    {/each}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        {/if}
                                    {/each}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            {/if}
        </div>
    {/if}
    
    {#if predictions && predictions.length > 0}
        <div class="prediction-results">
            <h2>分析结果</h2>

            <!-- 决策摘要卡：一眼抓住方向 / 目标价 / 关键价位 / 建议 / 风险 / 置信度 -->
            <div class="conclusion-card {conclusionDirection}">
                <div class="conclusion-head">
                    <span class="conclusion-badge">{conclusionLabel}</span>
                    <div class="conclusion-headline">
                        未来 {predictions.length} 个交易日历史漂移中枢：
                        <span class="conclusion-change {predHorizonChange >= 0 ? 'price-up' : 'price-down'}">
                            {predHorizonChange >= 0 ? '+' : ''}{predHorizonChange.toFixed(2)}%
                        </span>
                    </div>
                </div>
                {#if lastRealData}
                    <div class="conclusion-price">
                        <span class="cp-date">{new Date(lastRealData.date).toLocaleDateString()}</span>
                        <span class="cp-current">当前 {lastRealData.price.toFixed(2)}</span>
                        <span class="cp-arrow">→</span>
                        <span class="cp-target {targetPrice >= lastRealData.price ? 'price-up' : 'price-down'}">漂移中枢 {targetPrice.toFixed(2)}</span>
                    </div>
                {/if}
                <div class="conclusion-grid">
                    <div class="conclusion-item">
                        <span class="ci-label">技术状态解读</span>
                        <span class="ci-value">{conclusionAdvice}</span>
                    </div>
                    <div class="conclusion-item">
                        <span class="ci-label">平均信号强度</span>
                        <span class="ci-value" title="技术信号强度，非方向命中概率">{predAvgConfidence.toFixed(1)}%</span>
                    </div>
                    {#if predHorizonBand}
                        <div class="conclusion-item">
                            <span class="ci-label">{(predHorizonBand.confidence * 100).toFixed(0)}%校准区间</span>
                            <span class="ci-value">{predHorizonBand.lower_change_percent > 0 ? '+' : ''}{predHorizonBand.lower_change_percent.toFixed(1)}% ~ {predHorizonBand.upper_change_percent > 0 ? '+' : ''}{predHorizonBand.upper_change_percent.toFixed(1)}%</span>
                        </div>
                    {/if}
                    {#if predStressBand}
                        <div class="conclusion-item stress-item">
                            <span class="ci-label">95%压力区间</span>
                            <span class="ci-value">{predStressBand.lower_change_percent > 0 ? '+' : ''}{predStressBand.lower_change_percent.toFixed(1)}% ~ {predStressBand.upper_change_percent > 0 ? '+' : ''}{predStressBand.upper_change_percent.toFixed(1)}%</span>
                        </div>
                    {/if}
                    {#if conclusionRisk}
                        <div class="conclusion-item">
                            <span class="ci-label">风险等级</span>
                            <span class="ci-value risk-level {conclusionRisk.includes('低') ? 'low' : conclusionRisk.includes('高') ? 'high' : 'medium'}">{conclusionRisk}</span>
                        </div>
                    {/if}
                    {#if nearestSupport !== null || nearestResistance !== null}
                        <div class="conclusion-item">
                            <span class="ci-label">关键价位</span>
                            <span class="ci-value">{nearestSupport !== null ? `支撑 ${nearestSupport.toFixed(2)}` : ''}{nearestSupport !== null && nearestResistance !== null ? ' / ' : ''}{nearestResistance !== null ? `压力 ${nearestResistance.toFixed(2)}` : ''}</span>
                        </div>
                    {/if}
                    {#if showProfessionalAnalysis && professionalAnalysis}
                        <div class="conclusion-item">
                            <span class="ci-label">技术观察信号</span>
                            <span class="ci-value">偏多 {buyPointsCount} / 偏空 {sellPointsCount}</span>
                        </div>
                    {/if}
                </div>
                <div class="conclusion-note">单股方向没有稳定预测力；点估计仅为历史无条件漂移中枢，信号强度不是命中概率，不确定性以80%校准区间和95%压力区间为准。</div>
            </div>

            <RiskAlertPanel summary={predictionDiagnostics?.risk_summary ?? comprehensiveReport?.risk_summary ?? null} />

            {#if predictionDiagnostics?.point_estimate_note}
                <div class="method-note">
                    <strong>预测口径</strong>
                    <span>{predictionDiagnostics.point_estimate_note}</span>
                </div>
            {/if}

            <!-- 综合决策补充：动量/52周位置/历史基准率（一键综合预测时展示；均为描述性统计） -->
            {#if comprehensiveReport}
                <div class="comprehensive-card">
                    <div class="comprehensive-head">
                        <span class="comprehensive-title">🎯 综合决策补充</span>
                        <span class="comprehensive-sub">数据截至 {comprehensiveReport.latest_date}{comprehensiveReport.staleness_days > 4 ? '（数据较旧，建议刷新）' : ''} · 生成于 {comprehensiveReport.generated_at}</span>
                    </div>
                    <div class="comprehensive-grid">
                        <div class="comp-item">
                            <span class="comp-label">动量 5日</span>
                            <span class="comp-value {comprehensiveReport.momentum_5d !== null ? (comprehensiveReport.momentum_5d >= 0 ? 'price-up' : 'price-down') : ''}">{comprehensiveReport.momentum_5d !== null ? (comprehensiveReport.momentum_5d >= 0 ? '+' : '') + comprehensiveReport.momentum_5d.toFixed(2) + '%' : '—'}</span>
                        </div>
                        <div class="comp-item">
                            <span class="comp-label">动量 20日</span>
                            <span class="comp-value {comprehensiveReport.momentum_20d !== null ? (comprehensiveReport.momentum_20d >= 0 ? 'price-up' : 'price-down') : ''}">{comprehensiveReport.momentum_20d !== null ? (comprehensiveReport.momentum_20d >= 0 ? '+' : '') + comprehensiveReport.momentum_20d.toFixed(2) + '%' : '—'}</span>
                        </div>
                        <div class="comp-item">
                            <span class="comp-label">动量 60日</span>
                            <span class="comp-value {comprehensiveReport.momentum_60d !== null ? (comprehensiveReport.momentum_60d >= 0 ? 'price-up' : 'price-down') : ''}">{comprehensiveReport.momentum_60d !== null ? (comprehensiveReport.momentum_60d >= 0 ? '+' : '') + comprehensiveReport.momentum_60d.toFixed(2) + '%' : '—'}</span>
                        </div>
                        <div class="comp-item">
                            <span class="comp-label" title="现价在近一年最高/最低区间的位置：0=年内低点，100=年内高点">52周位置</span>
                            <span class="comp-value">{comprehensiveReport.week52_position !== null ? comprehensiveReport.week52_position.toFixed(0) : '—'}</span>
                        </div>
                        <div class="comp-item">
                            <span class="comp-label" title="近20/60/250个交易日中日线收涨的占比——无技能基准，供对照引擎信号">上涨占比 20/60/250日</span>
                            <span class="comp-value">{comprehensiveReport.up_ratio_20d !== null ? (comprehensiveReport.up_ratio_20d * 100).toFixed(0) + '%' : '—'} / {comprehensiveReport.up_ratio_60d !== null ? (comprehensiveReport.up_ratio_60d * 100).toFixed(0) + '%' : '—'} / {comprehensiveReport.up_ratio_250d !== null ? (comprehensiveReport.up_ratio_250d * 100).toFixed(0) + '%' : '—'}</span>
                        </div>
                        <div class="comp-item">
                            <span class="comp-label">日均涨跌（近250日）</span>
                            <span class="comp-value">{comprehensiveReport.avg_daily_change_250d !== null ? (comprehensiveReport.avg_daily_change_250d >= 0 ? '+' : '') + comprehensiveReport.avg_daily_change_250d.toFixed(3) + '%' : '—'}</span>
                        </div>
                    </div>
                    <div class="comprehensive-note">历史基准率为无技能对照（不看任何信息也能达到的水平），供对照上方引擎信号；动量与52周位置为描述性指标，非收益预测。</div>
                </div>
            {/if}

            <!-- 估值参考：PE/PB + 最新基本面（仅描述估值/质量/成长，非收益预测） -->
            {#if valuationContext}
                <div class="valuation-card">
                    <div class="valuation-head">
                        <span class="valuation-title">📊 估值参考</span>
                        <span class="valuation-sub">仅描述当前估值/质量/成长，非收益预测{valuationContext.report_date ? ` · 报告期 ${valuationContext.report_date}` : ''}</span>
                    </div>
                    {#if valuationContext.pe === null && valuationContext.pb === null && valuationContext.roe === null}
                        <div class="valuation-empty">暂无估值/基本面数据，请点击上方「刷新」按钮更新该股全部数据。</div>
                    {:else}
                        <div class="valuation-grid">
                            <div class="val-item"><span class="val-label">市盈率 PE</span><span class="val-value">{valuationContext.pe !== null ? valuationContext.pe.toFixed(2) : '—'}</span></div>
                            <div class="val-item"><span class="val-label">市净率 PB</span><span class="val-value">{valuationContext.pb !== null ? valuationContext.pb.toFixed(2) : '—'}</span></div>
                            <div class="val-item"><span class="val-label">流通市值</span><span class="val-value">{valuationContext.circulating_market_cap_yi !== null ? valuationContext.circulating_market_cap_yi.toFixed(0) + ' 亿' : '—'}</span></div>
                            <div class="val-item"><span class="val-label">ROE</span><span class="val-value">{valuationContext.roe !== null ? valuationContext.roe.toFixed(2) + '%' : '—'}</span></div>
                            <div class="val-item"><span class="val-label">每股收益 EPS</span><span class="val-value">{valuationContext.eps !== null ? valuationContext.eps.toFixed(2) : '—'}</span></div>
                            <div class="val-item"><span class="val-label">每股净资产 BPS</span><span class="val-value">{valuationContext.bps !== null ? valuationContext.bps.toFixed(2) : '—'}</span></div>
                            <div class="val-item"><span class="val-label">营收增长</span><span class="val-value {valuationContext.revenue_growth !== null ? (valuationContext.revenue_growth >= 0 ? 'price-up' : 'price-down') : ''}">{valuationContext.revenue_growth !== null ? (valuationContext.revenue_growth >= 0 ? '+' : '') + valuationContext.revenue_growth.toFixed(1) + '%' : '—'}</span></div>
                            <div class="val-item"><span class="val-label">利润增长</span><span class="val-value {valuationContext.profit_growth !== null ? (valuationContext.profit_growth >= 0 ? 'price-up' : 'price-down') : ''}">{valuationContext.profit_growth !== null ? (valuationContext.profit_growth >= 0 ? '+' : '') + valuationContext.profit_growth.toFixed(1) + '%' : '—'}</span></div>
                        </div>
                    {/if}
                </div>
            {/if}

            <!-- 新增：专业分析结果展示 -->
            {#if showProfessionalAnalysis && professionalAnalysis}
                <div class="professional-analysis">
                    <h3>技术状态与观察位</h3>

                    <!-- 买入点信号 -->
                    {#if professionalAnalysis.buy_points && professionalAnalysis.buy_points.length > 0}
                        <div class="buy-sell-section buy-section">
                            <h4>偏多观察位 ({professionalAnalysis.buy_points.length}个)</h4>
                            <div class="signals-grid">
                                {#each professionalAnalysis.buy_points as buyPoint, index}
                                    <div class="signal-card buy-card">
                                        <div class="signal-header">
                                            <span class="signal-type">{buyPoint.point_type}</span>
                                            <span class="signal-strength">强度: {(buyPoint.signal_strength * 100).toFixed(0)}分</span>
                                        </div>
                                        <div class="signal-body">
                                            <div class="signal-row">
                                                <span class="label">技术观察位:</span>
                                                <span class="value price-value">{buyPoint.price_level.toFixed(2)}元</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">下行失效参考:</span>
                                                <span class="value stop-loss">{buyPoint.stop_loss.toFixed(2)}元 (↓{Math.abs((buyPoint.stop_loss - buyPoint.price_level) / buyPoint.price_level * 100).toFixed(2)}%)</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">信号强度:</span>
                                                <span class="value confidence">{(buyPoint.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                            <div class="signal-reasons">
                                                <strong>理由：</strong>
                                                <ul>
                                                    {#each buyPoint.reasons as reason}
                                                        <li>{reason}</li>
                                                    {/each}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    {:else}
                        <div class="no-signal-card">
                            <p>当前未识别到偏多技术观察位</p>
                        </div>
                    {/if}
                    
                    <!-- 卖出点信号 -->
                    {#if professionalAnalysis.sell_points && professionalAnalysis.sell_points.length > 0}
                        <div class="buy-sell-section sell-section">
                            <h4>偏空观察位 ({professionalAnalysis.sell_points.length}个)</h4>
                            <div class="signals-grid">
                                {#each professionalAnalysis.sell_points as sellPoint, index}
                                    <div class="signal-card sell-card">
                                        <div class="signal-header">
                                            <span class="signal-type">{sellPoint.point_type}</span>
                                            <span class="signal-strength">强度: {(sellPoint.signal_strength * 100).toFixed(0)}分</span>
                                        </div>
                                        <div class="signal-body">
                                            <div class="signal-row">
                                                <span class="label">技术观察位:</span>
                                                <span class="value price-value">{sellPoint.price_level.toFixed(2)}元</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">上行失效参考:</span>
                                                <span class="value stop-loss">{sellPoint.stop_loss.toFixed(2)}元 (↑{Math.abs((sellPoint.stop_loss - sellPoint.price_level) / sellPoint.price_level * 100).toFixed(2)}%)</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">信号强度:</span>
                                                <span class="value confidence">{(sellPoint.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                            <div class="signal-reasons">
                                                <strong>理由：</strong>
                                                <ul>
                                                    {#each sellPoint.reasons as reason}
                                                        <li>{reason}</li>
                                                    {/each}
                                                </ul>
                                            </div>
                                        </div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    {:else}
                        <div class="no-signal-card">
                            <p>当前未识别到偏空技术观察位</p>
                        </div>
                    {/if}
                    
                    <!-- 支撑压力位 -->
                    <div class="support-resistance-section">
                        <h4>📍 支撑压力位分析</h4>
                        <div class="sr-content">
                            <div class="sr-position">
                                当前位置: <strong>{professionalAnalysis.support_resistance.current_position}</strong>
                            </div>
                            <div class="sr-levels">
                                {#if professionalAnalysis.support_resistance.support_levels.length > 0}
                                    <div class="sr-group support-group">
                                        <h5>🟢 关键支撑位</h5>
                                        <ul>
                                            {#each professionalAnalysis.support_resistance.support_levels as level, i}
                                                <li>{i + 1}. {level.toFixed(2)}元</li>
                                            {/each}
                                        </ul>
                                    </div>
                                {/if}
                                {#if professionalAnalysis.support_resistance.resistance_levels.length > 0}
                                    <div class="sr-group resistance-group">
                                        <h5>🔴 关键压力位</h5>
                                        <ul>
                                            {#each professionalAnalysis.support_resistance.resistance_levels as level, i}
                                                <li>{i + 1}. {level.toFixed(2)}元</li>
                                            {/each}
                                        </ul>
                                    </div>
                                {/if}
                            </div>
                        </div>
                    </div>
                </div>
            {/if}

            <section class="prediction-chart">
                <h3>校准区间路径</h3>
                <PredictionRangeChart {predictions} {lastRealData} />
            </section>
            
            <!-- 更多分析（多周期共振 · 量价背离）：默认折叠 -->
            {#if showProfessionalAnalysis && professionalAnalysis}
                <div class="collapsible-section">
                    <button type="button" class="collapsible-toggle" on:click={() => showMoreAnalysis = !showMoreAnalysis}>
                        {showMoreAnalysis ? '▾' : '▸'} 更多分析（多周期共振 · 量价背离）
                    </button>
                    {#if showMoreAnalysis}
                        <div class="collapsible-body">
                            <!-- 多周期共振 -->
                            <div class="multi-timeframe-section">
                                <h4>🔄 多周期共振分析</h4>
                                <div class="mtf-grid">
                                    <div class="mtf-item">
                                        <span class="mtf-label">日线:</span>
                                        <span class="mtf-value">{professionalAnalysis.multi_timeframe.daily_trend}</span>
                                    </div>
                                    <div class="mtf-item">
                                        <span class="mtf-label">周线:</span>
                                        <span class="mtf-value">{professionalAnalysis.multi_timeframe.weekly_trend}</span>
                                    </div>
                                    <div class="mtf-item">
                                        <span class="mtf-label">月线:</span>
                                        <span class="mtf-value">{professionalAnalysis.multi_timeframe.monthly_trend}</span>
                                    </div>
                                    <div class="mtf-item">
                                        <span class="mtf-label">共振级别:</span>
                                        <span class="mtf-value resonance-level">{professionalAnalysis.multi_timeframe.resonance_level}级</span>
                                    </div>
                                    <div class="mtf-item">
                                        <span class="mtf-label">共振方向:</span>
                                        <span class="mtf-value">{professionalAnalysis.multi_timeframe.resonance_direction}</span>
                                    </div>
                                    <div class="mtf-item">
                                        <span class="mtf-label">信号质量:</span>
                                        <span class="mtf-value quality-score">{professionalAnalysis.multi_timeframe.signal_quality.toFixed(0)}分</span>
                                    </div>
                                </div>
                            </div>

                            <!-- 量价背离 -->
                            <div class="divergence-section">
                                <h4>⚠️ 量价背离分析</h4>
                                <div class="divergence-content">
                                    <div class="divergence-indicators">
                                        {#if professionalAnalysis.divergence.has_bullish_divergence}
                                            <div class="divergence-badge bullish">
                                                🟢 底背离 (强度: {(professionalAnalysis.divergence.divergence_strength * 100).toFixed(0)}%)
                                            </div>
                                        {/if}
                                        {#if professionalAnalysis.divergence.has_bearish_divergence}
                                            <div class="divergence-badge bearish">
                                                🔴 顶背离 (强度: {(professionalAnalysis.divergence.divergence_strength * 100).toFixed(0)}%)
                                            </div>
                                        {/if}
                                    </div>
                                    <div class="divergence-message">
                                        💡 {professionalAnalysis.divergence.warning_message}
                                    </div>
                                </div>
                            </div>
                        </div>
                    {/if}
                </div>
            {/if}

            <!-- 预测统计信息（紧凑信息条） -->
            <div class="prediction-stats">
                <div class="stat-card">
                    <h4>平均预测涨幅</h4>
                    <div class="stat-value {predictions.reduce((sum, p) => sum + p.predicted_change_percent, 0) / predictions.length > 0 ? 'positive' : 'negative'}">
                        {(predictions.reduce((sum, p) => sum + p.predicted_change_percent, 0) / predictions.length).toFixed(2)}%
                    </div>
                </div>
                <div class="stat-card">
                    <h4>最高预测价格</h4>
                    <div class="stat-value">
                        {Math.max(...predictions.map(p => p.predicted_price)).toFixed(2)}
                    </div>
                </div>
                <div class="stat-card">
                    <h4>最低预测价格</h4>
                    <div class="stat-value">
                        {Math.min(...predictions.map(p => p.predicted_price)).toFixed(2)}
                    </div>
                </div>
            </div>

            <!-- 逐日明细表：默认折叠，主表精简核心列，行内"详情"展开理由/技术指标/风险 -->
            <div class="collapsible-section">
                <button type="button" class="collapsible-toggle" on:click={() => showDailyTable = !showDailyTable}>
                    {showDailyTable ? '▾' : '▸'} 逐日明细表（{predictions.length} 天）
                </button>
                {#if showDailyTable}
                    <div class="prediction-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>日期</th>
                                    <th>预测价格</th>
                                    <th title="历史走步样本经验覆盖约80%，由近20日已实现波动率校准；不代表个股结果的确定概率。">预测区间(80%)</th>
                                    <th>涨跌幅</th>
                                    <th title="技术信号强度，非方向命中概率">信号强度</th>
                                    <th>交易信号</th>
                                    <th>详情</th>
                                </tr>
                            </thead>
                            <tbody>
                                {#each predictions as prediction, i}
                                    <tr class:positive={prediction.predicted_change_percent > 0} class:negative={prediction.predicted_change_percent < 0}>
                                        <td>{new Date(prediction.target_date).toLocaleDateString()}</td>
                                        <td>{prediction.predicted_price.toFixed(2)}</td>
                                        <td>
                                            {#if prediction.interval}
                                                <span class="interval-band" title="历史经验覆盖约{(prediction.interval.confidence * 100).toFixed(0)}%（由已实现波动率校准，不代表个股结果的确定概率）">
                                                    {prediction.interval.lower_price.toFixed(2)} ~ {prediction.interval.upper_price.toFixed(2)}
                                                    <small>({prediction.interval.lower_change_percent > 0 ? '+' : ''}{prediction.interval.lower_change_percent.toFixed(1)}% ~ {prediction.interval.upper_change_percent > 0 ? '+' : ''}{prediction.interval.upper_change_percent.toFixed(1)}%)</small>
                                                </span>
                                            {:else}
                                                <span class="no-interval">—</span>
                                            {/if}
                                        </td>
                                        <td class:price-up={prediction.predicted_change_percent > 0} class:price-down={prediction.predicted_change_percent < 0}>
                                            {prediction.predicted_change_percent > 0 ? '+' : ''}{prediction.predicted_change_percent.toFixed(2)}%
                                        </td>
                                        <td>
                                            <div class="confidence-indicator">
                                                <div class="confidence-bar-inline" style="width: {prediction.confidence * 100}%"></div>
                                                <span>{(prediction.confidence * 100).toFixed(2)}%</span>
                                            </div>
                                        </td>
                                        <td>
                                            <span class="signal-badge {prediction.trading_signal?.includes('买入') || prediction.trading_signal?.includes('看涨') ? 'buy-signal' : prediction.trading_signal?.includes('卖出') || prediction.trading_signal?.includes('看跌') ? 'sell-signal' : 'hold-signal'}">
                                                {prediction.trading_signal || '持有'}
                                            </span>
                                            {#if prediction.signal_strength}
                                                <div class="signal-strength">
                                                    强度: {(prediction.signal_strength * 100).toFixed(0)}%
                                                </div>
                                            {/if}
                                        </td>
                                        <td>
                                            <button class="action-btn" type="button" on:click={() => expandedPredIndex = expandedPredIndex === i ? null : i}>
                                                {expandedPredIndex === i ? '收起' : '详情'}
                                            </button>
                                        </td>
                                    </tr>
                                    {#if expandedPredIndex === i}
                                        <tr class="pred-detail-row">
                                            <td colspan="7">
                                                <div class="pred-detail">
                                                    <div class="pred-detail-block">
                                                        <span class="pred-detail-label">风险评级</span>
                                                        <span class="risk-badge {predictionDiagnostics?.risk_summary.level === 'high' ? 'high-risk' : predictionDiagnostics?.risk_summary.level === 'medium' ? 'medium-risk' : 'low-risk'}">
                                                            {predictionDiagnostics?.risk_summary.level_label ?? '未评估'}
                                                        </span>
                                                    </div>
                                                    <div class="pred-detail-block">
                                                        <span class="pred-detail-label">预测理由</span>
                                                        {#if prediction.prediction_reason}
                                                            <div class="prediction-reason">
                                                                <div class="reason-text">{prediction.prediction_reason}</div>
                                                                {#if prediction.key_factors && prediction.key_factors.length > 0}
                                                                    <div class="key-factors">
                                                                        {#each prediction.key_factors as factor}
                                                                            <span class="factor-tag">{factor}</span>
                                                                        {/each}
                                                                    </div>
                                                                {/if}
                                                            </div>
                                                        {:else}
                                                            <span class="no-reason">暂无理由</span>
                                                        {/if}
                                                    </div>
                                                    {#if prediction.technical_indicators}
                                                        <div class="pred-detail-block">
                                                            <span class="pred-detail-label">技术指标</span>
                                                            <div class="tech-indicators">
                                                                <span title="RSI: {prediction.technical_indicators.rsi.toFixed(1)}">
                                                                    RSI: {prediction.technical_indicators.rsi > 70 ? '超买' : prediction.technical_indicators.rsi < 30 ? '超卖' : '正常'}
                                                                </span>

                                                                <!-- 增强MACD指标展示 -->
                                                                <div class="tech-detail-indicator">
                                                                    <span class="tech-label">MACD:</span>
                                                                    <div class="tech-values">
                                                                        <span class="tech-value {prediction.technical_indicators.macd_dif > prediction.technical_indicators.macd_dea ? 'positive' : 'negative'}">
                                                                            DIF: {prediction.technical_indicators.macd_dif.toFixed(2)}
                                                                        </span>
                                                                        <span class="tech-value">
                                                                            DEA: {prediction.technical_indicators.macd_dea.toFixed(2)}
                                                                        </span>
                                                                        <span class="tech-value {prediction.technical_indicators.macd_histogram > 0 ? 'positive' : 'negative'}">
                                                                            HIST: {prediction.technical_indicators.macd_histogram.toFixed(2)}
                                                                        </span>
                                                                    </div>
                                                                    {#if prediction.technical_indicators.macd_golden_cross}
                                                                        <span class="tech-signal buy-signal">金叉</span>
                                                                    {:else if prediction.technical_indicators.macd_death_cross}
                                                                        <span class="tech-signal sell-signal">死叉</span>
                                                                    {/if}
                                                                </div>

                                                                <!-- 增强KDJ指标展示 -->
                                                                <div class="tech-detail-indicator">
                                                                    <span class="tech-label">KDJ:</span>
                                                                    <div class="tech-values">
                                                                        <span class="tech-value {prediction.technical_indicators.kdj_k > prediction.technical_indicators.kdj_d ? 'positive' : 'negative'}">
                                                                            K: {prediction.technical_indicators.kdj_k.toFixed(1)}
                                                                        </span>
                                                                        <span class="tech-value">
                                                                            D: {prediction.technical_indicators.kdj_d.toFixed(1)}
                                                                        </span>
                                                                        <span class="tech-value {prediction.technical_indicators.kdj_j > 80 ? 'overbought' : prediction.technical_indicators.kdj_j < 20 ? 'oversold' : ''}">
                                                                            J: {prediction.technical_indicators.kdj_j.toFixed(1)}
                                                                        </span>
                                                                    </div>
                                                                    {#if prediction.technical_indicators.kdj_golden_cross}
                                                                        <span class="tech-signal buy-signal">金叉</span>
                                                                    {:else if prediction.technical_indicators.kdj_death_cross}
                                                                        <span class="tech-signal sell-signal">死叉</span>
                                                                    {:else if prediction.technical_indicators.kdj_overbought}
                                                                        <span class="tech-signal overbought-signal">超买</span>
                                                                    {:else if prediction.technical_indicators.kdj_oversold}
                                                                        <span class="tech-signal oversold-signal">超卖</span>
                                                                    {/if}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    {/if}
                                                </div>
                                            </td>
                                        </tr>
                                    {/if}
                                {/each}
                            </tbody>
                        </table>
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
        width: 100%;
        box-sizing: border-box;
    }

    h1 {
        font-size: 1.5rem;
        margin: 0 0 1rem;
        text-align: left;
    }
    
    h2 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .input-group {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .search-input {
        flex: 1;
        padding: 0.875rem 1.25rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        color: inherit;
        font-size: 1rem;
    }

    button {
        padding: 0.875rem 2rem;
        background: var(--active-color, #4f46e5);
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }

    button:hover:not(:disabled) {
        opacity: 0.9;
    }
    
    button:disabled {
        background: #6b7280;
        cursor: not-allowed;
    }

    button.loading {
        background: #4f46e5;
        cursor: not-allowed;
    }

    .spinner {
        display: inline-block;
        width: 1.25rem;
        height: 1.25rem;
        border: 2px solid #fff;
        border-top-color: transparent;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
    
    .error-message {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .tabs {
        display: flex;
        gap: 0.5rem;
        margin: 2rem 0 1rem;
    }
    
    .tabs button {
        flex: 1;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.7);
    }
    
    .tabs button.active {
        background: var(--active-color, #4f46e5);
        color: white;
    }
    
    .model-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
    }

    .model-lab-switch {
        width: fit-content;
        display: flex;
        gap: 0.25rem;
        margin: 0 0 0.75rem;
        padding: 0.25rem;
        border: 1px solid #374151;
        border-radius: 8px;
        background: #171b22;
    }

    .model-lab-switch button {
        padding: 0.5rem 0.8rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border-radius: 5px;
        background: transparent;
        color: #9ca3af;
        font-size: 0.82rem;
    }

    .model-lab-switch button.active {
        background: #374151;
        color: #f3f4f6;
    }

    .analysis-toolbar {
        padding: 1rem;
        border: 1px solid #374151;
        border-radius: 8px;
        background: #171b22;
    }

    .analysis-heading {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.25rem;
    }

    .analysis-heading h2 {
        margin: 0 0 0.25rem;
        font-size: 1.05rem;
    }

    .analysis-heading span,
    .horizon-control label {
        color: #9ca3af;
        font-size: 0.76rem;
    }

    .horizon-control {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .horizon-presets {
        display: flex;
        overflow: hidden;
        border: 1px solid #4b5563;
        border-radius: 6px;
    }

    .horizon-presets button {
        min-width: 48px;
        padding: 0.45rem 0.6rem;
        border-right: 1px solid #4b5563;
        border-radius: 0;
        background: transparent;
        color: #cbd5e1;
        font-size: 0.78rem;
    }

    .horizon-presets button:last-child {
        border-right: none;
    }

    .horizon-presets button.active {
        background: #0e7490;
        color: white;
    }

    .horizon-control input {
        width: 58px;
        padding: 0.45rem;
        box-sizing: border-box;
        border: 1px solid #4b5563;
        border-radius: 6px;
        background: #11151b;
        color: #f3f4f6;
    }

    .analysis-action {
        display: flex;
        justify-content: flex-end;
        margin-top: 0.85rem;
    }

    .analysis-action .predict-btn {
        width: auto;
        min-width: 180px;
        margin: 0;
        padding: 0.65rem 1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        background: #0e7490;
        font-size: 0.9rem;
    }

    :global(.spin) {
        animation: spin 0.8s linear infinite;
    }
    
    .model-list {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .model-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        padding: 1rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .model-item.selected {
        background: rgba(79, 70, 229, 0.3);
        border: 1px solid #4f46e5;
    }
    
    .model-item:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    .model-info {
        flex: 1;
    }
    
    .model-info h3 {
        margin: 0 0 0.5rem;
        font-size: 1.1rem;
    }
    
    .model-details {
        display: flex;
        gap: 1rem;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .model-actions {
        display: flex;
        gap: 0.5rem;
    }
    
    .action-btn {
        padding: 0.5rem 1rem;
        background: #4b5563;
        font-size: 0.875rem;
    }
    
    .delete-btn {
        padding: 0.5rem 1rem;
        background: #ef4444;
        font-size: 0.875rem;
    }
    
    .advanced-btn {
        background: transparent;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .advanced-options {
        background: rgba(0, 0, 0, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .prediction-settings {
        display: flex;
        gap: 1rem;
        align-items: center;
        margin-top: 1.5rem;
    }
    
    .prediction-settings label {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    .prediction-settings input {
        width: 4rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.25rem;
        color: inherit;
    }

    .professional-predict-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        background: #0e7490;
    }
    
    .training-form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    .form-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .training-form input,
    .training-form select {
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.25rem;
        color: inherit;
    }
    
    .training-form select {
        color: #000000;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .training-form select option {
        color: #000000;
        background: #ffffff;
    }
    
    .features-list {
        margin-bottom: 1rem;
    }
    
    .features-checkboxes {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.75rem;
    }
    
    .features-checkboxes label {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prediction-results {
        margin-top: 2rem;
        width: 100%;
        box-sizing: border-box;
    }

    /* 核心结论卡 */
    .conclusion-card {
        margin: 0 0 1.5rem;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #6366f1;
        background: rgba(99, 102, 241, 0.08);
    }
    .conclusion-card.bull {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
    }
    .conclusion-card.bear {
        border-left-color: #22c55e;
        background: rgba(34, 197, 94, 0.08);
    }
    .conclusion-card.neutral {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
    }
    .conclusion-head {
        display: flex;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .conclusion-badge {
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.3rem 0.9rem;
        border-radius: 0.5rem;
        background: rgba(0, 0, 0, 0.2);
        white-space: nowrap;
    }
    .conclusion-card.bull .conclusion-badge { color: #f87171; }
    .conclusion-card.bear .conclusion-badge { color: #4ade80; }
    .conclusion-card.neutral .conclusion-badge { color: #f59e0b; }
    .conclusion-headline {
        font-size: 1.05rem;
        font-weight: 600;
    }
    .conclusion-change.price-up { color: #ef4444; }
    .conclusion-change.price-down { color: #22c55e; }
    .conclusion-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }
    .conclusion-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        background: rgba(0, 0, 0, 0.15);
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
    }
    .ci-label {
        font-size: 0.75rem;
        opacity: 0.7;
    }
    .ci-value {
        font-size: 0.95rem;
        font-weight: 600;
    }
    .ci-value.risk-level.low { color: #10b981; }
    .ci-value.risk-level.medium { color: #f59e0b; }
    .ci-value.risk-level.high { color: #ef4444; }
    .conclusion-note {
        margin-top: 0.9rem;
        font-size: 0.75rem;
        opacity: 0.65;
    }
    .conclusion-price {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.75rem;
        font-size: 1.05rem;
        font-weight: 600;
        flex-wrap: wrap;
    }
    .cp-date {
        font-size: 0.75rem;
        font-weight: 400;
        opacity: 0.7;
    }
    .cp-arrow {
        opacity: 0.6;
    }
    .cp-target.price-up { color: #ef4444; }
    .cp-target.price-down { color: #22c55e; }

    .stress-item .ci-value {
        color: #fbbf24;
    }

    .method-note {
        margin: 0.75rem 0 1rem;
        padding: 0.7rem 0.85rem;
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        border-left: 3px solid #22d3ee;
        background: rgba(34, 211, 238, 0.06);
        color: #cbd5e1;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    .method-note strong {
        flex: 0 0 auto;
        color: #67e8f9;
    }

    /* 折叠区（更多分析 / 逐日明细表） */
    .collapsible-section {
        margin: 1.25rem 0;
    }
    .collapsible-toggle {
        width: 100%;
        text-align: left;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 0.5rem;
        padding: 0.7rem 1rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: inherit;
        cursor: pointer;
    }
    .collapsible-toggle:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    .collapsible-body {
        margin-top: 0.75rem;
    }
    .chart-empty-note {
        margin: 1rem 0;
        padding: 1rem;
        text-align: center;
        opacity: 0.6;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 0.5rem;
    }

    /* 逐日明细表：行内展开详情 */
    .pred-detail-row td {
        background: rgba(0, 0, 0, 0.15);
    }
    .interval-band {
        display: inline-flex;
        flex-direction: column;
        line-height: 1.2;
        color: #cbd5e1;
        font-variant-numeric: tabular-nums;
        cursor: help;
    }
    .interval-band small {
        color: #94a3b8;
        font-size: 0.75em;
    }
    .no-interval {
        color: #64748b;
    }

    /* 估值参考卡：PE/PB + 最新基本面（仅参考展示） */
    .valuation-card {
        margin: 0.75rem 0 1rem;
        padding: 0.9rem 1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 0.6rem;
    }
    .valuation-head {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }
    .valuation-title {
        font-size: 0.95rem;
        font-weight: 700;
    }
    .valuation-sub {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .valuation-empty {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .valuation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 0.5rem 1rem;
    }
    .val-item {
        display: flex;
        flex-direction: column;
        line-height: 1.3;
    }
    .val-label {
        font-size: 0.72rem;
        color: #94a3b8;
    }
    .val-value {
        font-size: 0.95rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    /* 收藏星标（与收藏池联动） */
    .watch-star {
        flex: 0 0 auto;
        padding: 0 1rem;
        font-size: 1.3rem;
        line-height: 1;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 0.5rem;
        color: #94a3b8;
        cursor: pointer;
        transition: color 0.2s ease, background 0.2s ease;
    }
    .watch-star:hover {
        background: rgba(255, 255, 255, 0.12);
    }
    .watch-star.watched {
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.4);
    }

    /* 综合决策补充卡（一键综合预测）：样式对齐估值参考卡 */
    .comprehensive-card {
        margin: 0.75rem 0 1rem;
        padding: 0.9rem 1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 0.6rem;
    }
    .comprehensive-head {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 0.6rem;
    }
    .comprehensive-title {
        font-size: 0.95rem;
        font-weight: 700;
    }
    .comprehensive-sub {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .comprehensive-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.5rem 1rem;
    }
    .comp-item {
        display: flex;
        flex-direction: column;
        line-height: 1.3;
    }
    .comp-label {
        font-size: 0.72rem;
        color: #94a3b8;
    }
    .comp-value {
        font-size: 0.95rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .comp-value.price-up { color: #ef4444; }
    .comp-value.price-down { color: #22c55e; }
    .comprehensive-note {
        margin-top: 0.7rem;
        font-size: 0.75rem;
        line-height: 1.6;
        color: #94a3b8;
    }
    .comprehensive-loading {
        margin: 0.5rem 0 1rem;
        padding: 0.7rem 1rem;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 0.5rem;
        color: #a5b4fc;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pred-detail {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 0.5rem 0.25rem;
    }
    .pred-detail-block {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .pred-detail-label {
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.7;
    }

    .prediction-table {
        overflow-x: auto;
        width: 100%;
        max-width: 100%;
    }
    
    table {
        width: 100%;
        min-width: 1200px; /* 设置最小宽度，确保内容不被压缩太多 */
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 0.75rem; /* 进一步减小字体 */
    }
    
    th, td {
        padding: 0.4rem 0.5rem; /* 进一步减小padding */
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        word-wrap: break-word;
        overflow: hidden;
        vertical-align: top; /* 顶部对齐 */
    }
    
    /* 优化列宽比例 */
    th:nth-child(1), td:nth-child(1) { width: 80px; } /* 日期 */
    th:nth-child(2), td:nth-child(2) { width: 70px; } /* 价格 */
    th:nth-child(3), td:nth-child(3) { width: 65px; } /* 涨跌幅 */
    th:nth-child(4), td:nth-child(4) { width: 90px; } /* 置信度 */
    th:nth-child(5), td:nth-child(5) { width: 80px; } /* 交易信号 */
    th:nth-child(6), td:nth-child(6) { width: 260px; } /* 预测理由 */
    th:nth-child(7), td:nth-child(7) { width: 280px; } /* 技术指标 */
    th:nth-child(8), td:nth-child(8) { width: 70px; } /* 风险评级 */
    
    th {
        background: rgba(255, 255, 255, 0.05);
        font-weight: 600;
        font-size: 0.75rem; /* 更小的标题字体 */
        white-space: nowrap; /* 标题不换行 */
    }
    
    /* 技术指标紧凑显示 */
    .tech-indicators {
        font-size: 0.65rem; /* 更小 */
        line-height: 1.3;
    }
    
    .tech-detail-indicator {
        margin-bottom: 0.2rem;
    }
    
    .tech-label {
        font-weight: 600;
        display: inline-block;
        min-width: 40px;
        font-size: 0.65rem;
    }
    
    .tech-values {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.15rem;
        font-size: 0.6rem; /* 更小 */
    }
    
    .tech-value {
        padding: 0.05rem 0.25rem;
        border-radius: 0.2rem;
        background: rgba(255, 255, 255, 0.05);
        white-space: nowrap;
    }
    
    .tech-value.positive {
        color: #10b981;
    }
    
    .tech-value.negative {
        color: #ef4444;
    }
    
    .tech-value.overbought {
        color: #fbbf24;
    }
    
    .tech-value.oversold {
        color: #3b82f6;
    }
    
    .tech-signal {
        font-size: 0.6rem; /* 更小 */
        padding: 0.05rem 0.3rem;
        border-radius: 0.2rem;
        margin-left: 0.2rem;
        display: inline-block;
        white-space: nowrap;
    }
    
    .tech-signal.buy-signal {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .tech-signal.sell-signal {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .tech-signal.overbought-signal {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
    }
    
    .tech-signal.oversold-signal {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }
    
    /* 预测理由紧凑显示 */
    .reason-text {
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    .key-factors {
        display: flex;
        flex-wrap: wrap;
        gap: 0.15rem;
    }
    
    .factor-tag {
        font-size: 0.6rem; /* 更小 */
        padding: 0.05rem 0.3rem;
        background: rgba(99, 102, 241, 0.2);
        color: #6366f1;
        border-radius: 0.2rem;
        white-space: nowrap;
    }
    
    /* 置信度条紧凑显示 */
    .confidence-indicator {
        font-size: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    .confidence-bar-inline {
        height: 4px;
        background: linear-gradient(90deg, #ef4444, #fbbf24, #10b981);
        border-radius: 2px;
        min-width: 30px;
        max-width: 60px;
    }
    
    /* 信号徽章紧凑显示 */
    .signal-badge {
        font-size: 0.65rem;
        padding: 0.15rem 0.4rem;
        border-radius: 0.25rem;
        display: inline-block;
        white-space: nowrap;
    }
    
    .signal-badge.buy-signal {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .signal-badge.sell-signal {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    .signal-badge.hold-signal {
        background: rgba(107, 114, 128, 0.2);
        color: #9ca3af;
    }
    
    .signal-strength {
        font-size: 0.6rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.1rem;
    }
    
    /* 风险徽章紧凑显示 */
    .risk-badge {
        font-size: 0.65rem;
        padding: 0.15rem 0.4rem;
        border-radius: 0.25rem;
        display: inline-block;
        white-space: nowrap;
    }
    
    .risk-badge.low-risk {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    .risk-badge.medium-risk {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
    }
    
    .risk-badge.high-risk {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
    }
    
    /* 无理由文字 */
    .no-reason {
        color: rgba(255, 255, 255, 0.4);
        font-style: italic;
        font-size: 0.65rem;
    }
    
    tr.positive td:nth-child(3) {
        color: #10b981;
        font-weight: 600;
    }
    
    tr.negative td:nth-child(3) {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* 价格涨跌样式 */
    .price-up {
        color: #10b981 !important;
        font-weight: 600;
    }
    
    .price-down {
        color: #ef4444 !important;
        font-weight: 600;
    }
    
    /* 日期列紧凑显示 */
    td:nth-child(1) {
        font-size: 0.7rem;
        white-space: nowrap;
    }
    
    /* 价格列紧凑显示 */
    td:nth-child(2) {
        font-weight: 600;
        white-space: nowrap;
    }
    
    .training-progress {
        margin-bottom: 1rem;
    }
    
    .progress-bar {
        height: 1rem;
        background-color: #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background-color: #4f46e5;
    }
    
    .progress-text {
        font-size: 0.875rem;
        font-weight: 600;
        color: #111827;
    }
    
    .training-logs {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f3f4f6;
        border-radius: 0.5rem;
        color: #000000;
    }
    
    .logs-container {
        max-height: 200px;
        overflow-y: auto;
    }
    
    .log-entry {
        margin-bottom: 0.5rem;
    }
    
    .log-time {
        font-size: 0.875rem;
        color: #000000;
    }
    
    .log-content {
        font-size: 0.875rem;
        color: #000000;
    }
    
    .model-comparison {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f3f4f6;
        border-radius: 0.5rem;
        color: #000000;
    }
    
    .comparison-chart {
        margin-top: 1rem;
    }
    
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        color: #000000;
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        color: #000000;
    }
    
    .comparison-table th {
        background-color: rgba(0, 0, 0, 0.05);
        font-weight: 600;
        color: #000000;
    }
    
    .performance-bar {
        height: 1rem;
        background-color: #e5e7eb;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    .performance-fill {
        height: 100%;
        background-color: #10b981;
    }
    
    .prediction-chart {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    .prediction-stats {
        margin-top: 2rem;
        display: flex;
        gap: 1rem;
    }
    
    .stat-card {
        flex: 1;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    
    .positive {
        color: #f3f4f6;
    }

    .stat-value.positive { color: #ef4444; }
    .stat-value.negative { color: #22c55e; }

    .backtest-warning {
        margin: 1rem 0;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        background: rgba(245, 158, 11, 0.08);
        color: #fde68a;
        font-size: 0.85rem;
    }
    
    .negative {
        color: #ef4444;
    }
    
    .confidence-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence-bar-inline {
        height: 1rem;
        background-color: rgb(34, 197, 94);
    }
    
    .risk-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .low-risk {
        background-color: #10b981;
        color: #10b981;
    }
    
    .medium-risk {
        background-color: #f59e0b;
        color: #f59e0b;
    }
    
    .high-risk {
        background-color: #ef4444;
        color: #ef4444;
    }
    
    .signal-badge {
        padding: 0.375rem 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .buy-signal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .sell-signal {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .hold-signal {
        background-color: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid #fbbf24;
    }
    
    .signal-strength {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.25rem;
    }
    
    .tech-indicators {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        font-size: 0.75rem;
    }
    
    .tech-indicators span {
        padding: 0.125rem 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
        white-space: nowrap;
    }
    
    .tech-detail-indicator {
        display: flex;
        flex-direction: column;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 0.25rem;
        padding: 0.25rem;
        gap: 0.25rem;
    }
    
    .tech-label {
        font-weight: bold;
        color: rgba(255, 255, 255, 0.9);
        padding: 0 !important;
        background: transparent !important;
    }
    
    .tech-values {
        display: flex;
        gap: 0.25rem;
    }
    
    .tech-value {
        padding: 0.125rem 0.25rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
        font-size: 0.7rem;
    }
    
    .tech-value.positive {
        color: #10b981;
    }
    
    .tech-value.negative {
        color: #ef4444;
    }
    
    .tech-value.overbought {
        color: #ef4444;
        font-weight: bold;
    }
    
    .tech-value.oversold {
        color: #10b981;
        font-weight: bold;
    }
    
    .tech-signal {
        align-self: flex-start;
        padding: 0.125rem 0.375rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.7rem;
    }
    
    .buy-signal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .sell-signal {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .overbought-signal {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .oversold-signal {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    /* 新增：中国股市风格的涨跌颜色 */
    .price-up {
        color: #ef4444 !important; /* 红色表示上涨 */
        font-weight: bold;
    }
    
    .price-down {
        color: #10b981 !important; /* 绿色表示下跌 */
        font-weight: bold;
    }

    /* 回测功能样式 */
    .backtest-settings {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .backtest-settings .form-group {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .backtest-settings input,
    .backtest-settings select {
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.25rem;
        color: inherit;
    }
    
    .backtest-settings select {
        color: #000000;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .backtest-settings select option {
        color: #000000;
        background: #ffffff;
    }
    
    .backtest-settings small {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.875rem;
    }
    
    .backtest-report {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .backtest-summary {
        margin-bottom: 2rem;
    }
    
    .summary-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1.5rem;
    }
    
    .summary-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
    }
    
    .stat-label {
        font-weight: 600;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: bold;
        color: #10b981;
    }

    .model-name-value {
        font-size: 1rem;
        text-align: right;
        overflow-wrap: anywhere;
    }
    
    .accuracy-trend {
        margin-bottom: 2rem;
    }
    
    .trend-chart {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        height: 200px;
    }
    
    .trend-bars {
        display: flex;
        align-items: flex-end;
        gap: 0.5rem;
        height: 100%;
    }
    
    .trend-bar {
        flex: 1;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
        min-height: 20px;
        position: relative;
    }
    
    .trend-bar-fill {
        width: 100%;
        height: 100%;
        border-radius: 0.25rem;
    }
    
    .backtest-details {
        margin-top: 2rem;
    }
    
    .backtest-table {
        overflow-x: auto;
    }
    
    .backtest-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .backtest-table th,
    .backtest-table td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .backtest-table th {
        background: rgba(255, 255, 255, 0.05);
        font-weight: 600;
    }
    
    .accuracy-cell {
        width: 120px;
    }
    
    .accuracy-bar {
        position: relative;
        height: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.25rem;
        overflow: hidden;
        display: flex;
        align-items: center;
        padding: 0 0.5rem;
    }
    
    .accuracy-fill {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        background: #4f46e5;
        border-radius: 0.25rem;
        transition: width 0.3s ease;
    }
    
    .accuracy-fill.direction-fill {
        background: #10b981;
    }
    
    .accuracy-bar span {
        position: relative;
        z-index: 1;
        font-size: 0.875rem;
        font-weight: 600;
        color: white;
    }

    /* 回测详情扩展 */
    .entry-details {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }
    .entry-table-wrapper {
        overflow-x: auto;
    }
    .entry-table {
        width: 100%;
        border-collapse: collapse;
    }
    .entry-table th,
    .entry-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .chip-correct {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 0.5rem;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .chip-wrong {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 0.5rem;
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .details-row td {
        background: rgba(255, 255, 255, 0.03);
        padding: 0;
    }
    
    /* ==================== 专业分析样式 ==================== */
    .professional-analysis {
        margin: 2rem 0;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid #374151;
        border-radius: 8px;
    }
    
    .professional-analysis h3 {
        margin-top: 0;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        text-align: center;
        color: #e5e7eb;
    }
    
    .risk-level {
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    
    .risk-level.low {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .risk-level.medium {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid #fbbf24;
    }
    
    .risk-level.high {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .buy-sell-section {
        margin: 1.5rem 0;
    }
    
    .buy-sell-section h4 {
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .signals-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
        gap: 1rem;
    }
    
    .signal-card {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 0.5rem;
        overflow: hidden;
        border: 2px solid;
    }
    
    .buy-card {
        border-color: #10b981;
    }
    
    .sell-card {
        border-color: #ef4444;
    }
    
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .buy-card .signal-header {
        background: rgba(16, 185, 129, 0.1);
    }
    
    .sell-card .signal-header {
        background: rgba(239, 68, 68, 0.1);
    }
    
    .signal-type {
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .signal-strength {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
    
    .signal-body {
        padding: 1rem;
    }
    
    .signal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .signal-row:last-child {
        border-bottom: none;
    }
    
    .signal-row .label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    .signal-row .value {
        font-weight: 600;
        font-size: 1rem;
    }
    
    .signal-row .price-value {
        color: #fbbf24;
        font-size: 1.1rem;
    }
    
    .signal-row .stop-loss {
        color: #10b981;  /* 止损(下跌)用绿色 - 符合中国股市习惯 */
    }
    
    .signal-row .confidence {
        color: #34d399;
    }
    
    .signal-reasons {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .signal-reasons ul {
        margin: 0.5rem 0 0 0;
        padding-left: 1.5rem;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .signal-reasons li {
        margin: 0.25rem 0;
    }
    
    .no-signal-card {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .support-resistance-section,
    .multi-timeframe-section,
    .divergence-section {
        margin: 1.5rem 0;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 0.5rem;
    }
    
    .support-resistance-section h4,
    .multi-timeframe-section h4,
    .divergence-section h4 {
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .sr-position {
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.25rem;
        text-align: center;
    }
    
    .sr-levels {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    .sr-group h5 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    .sr-group ul {
        margin: 0;
        padding-left: 1.5rem;
        font-size: 0.9rem;
    }
    
    .sr-group li {
        margin: 0.25rem 0;
    }
    
    .support-group {
        border-left: 3px solid #10b981;
        padding-left: 0.75rem;
    }
    
    .resistance-group {
        border-left: 3px solid #ef4444;
        padding-left: 0.75rem;
    }
    
    .mtf-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
    }
    
    .mtf-item {
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.25rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .mtf-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .mtf-value {
        font-weight: bold;
        font-size: 1rem;
    }
    
    .mtf-value.resonance-level {
        color: #667eea;
        font-size: 1.2rem;
    }
    
    .mtf-value.quality-score {
        color: #10b981;
        font-size: 1.2rem;
    }
    
    .divergence-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    
    .divergence-indicators {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .divergence-badge {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        border: 2px solid;
    }
    
    .divergence-badge.bullish {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10b981;
        color: #10b981;
    }
    
    .divergence-badge.bearish {
        background: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
        color: #ef4444;
    }
    
    .divergence-message {
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.25rem;
        border-left: 3px solid #fbbf24;
    }
    
    /* 预测理由样式 */
    .prediction-reason {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .reason-text {
        font-size: 0.875rem;
        line-height: 1.4;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .key-factors {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
    }
    
    .factor-tag {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        background: rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 0.25rem;
        font-size: 0.75rem;
        color: #93c5fd;
    }
    
    .no-reason {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.875rem;
        font-style: italic;
    }
    
    /* 纯技术分析样式 */
    .section-desc {
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .prediction-settings {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .predict-btn {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* 与应用工作台统一的视觉基线 */
    .container {
        max-width: 1320px;
        padding: 0;
    }

    h2 {
        font-size: 1.05rem;
        margin-top: 1.15rem;
    }

    .input-group { gap: 0.55rem; margin: 0 0 1rem; }
    .search-input { min-height: 42px; padding: 0.55rem 0.75rem; border-color: var(--border-strong); border-radius: 6px; background: var(--surface-2); }
    .watch-star { width: 42px; min-width: 42px; padding: 0; background: var(--surface-2); border: 1px solid var(--border-strong); color: var(--text-muted); }
    .watch-star.watched { color: var(--warning); border-color: rgba(232, 174, 74, 0.5); }
    .error-message { margin: 0 0 1rem; padding: 0.75rem 0.85rem; border: 1px solid rgba(239, 107, 115, 0.42); border-radius: 6px; background: rgba(239, 107, 115, 0.08); color: #ffabb0; }

    button { border-radius: 6px; background: #16899a; }
    button:disabled { background: var(--surface-3); }
    button.loading { background: #16899a; }
    .tabs { gap: 0.4rem; margin: 1rem 0; padding: 0.25rem; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); }
    .tabs button { min-height: 42px; padding: 0.55rem 0.75rem; background: transparent; color: var(--text-secondary); }
    .tabs button.active { background: var(--accent-muted); color: var(--accent-hover); }
    .model-lab-switch, .analysis-toolbar, .model-section { border: 1px solid var(--border); background: var(--surface-1); }
    .model-lab-switch button.active { background: var(--surface-3); color: var(--text-primary); }
    .analysis-heading span, .horizon-control label { color: var(--text-secondary); }
    .horizon-presets { border-color: var(--border-strong); }
    .horizon-presets button { border-right-color: var(--border-strong); color: var(--text-secondary); }
    .horizon-presets button.active, .analysis-action .predict-btn, .professional-predict-btn { background: #16899a; }
    .horizon-control input { border-color: var(--border-strong); background: var(--surface-2); color: var(--text-primary); }
    .model-item { border: 1px solid var(--border); background: var(--surface-2); }
    .model-item:hover { background: var(--surface-3); }
    .model-item.selected { border-color: var(--accent); background: var(--accent-muted); }
    .model-details { color: var(--text-secondary); }
    .action-btn { background: var(--surface-3); }
    .training-form input, .training-form select, .prediction-settings input { border-color: var(--border-strong); background: var(--surface-2); color: var(--text-primary); }
    .training-form select, .training-form select option { background: var(--surface-2); color: var(--text-primary); }
    .advanced-options, .prediction-settings { background: var(--surface-2); }
    .prediction-results { margin-top: 1.25rem; }
    .conclusion-card { border-left-color: var(--accent); background: var(--accent-muted); }
    .conclusion-card.bull { border-left-color: var(--price-up); background: rgba(241, 91, 100, 0.08); }
    .conclusion-card.bear { border-left-color: var(--price-down); background: rgba(53, 200, 137, 0.08); }
    .conclusion-card.neutral { border-left-color: var(--warning); background: rgba(232, 174, 74, 0.08); }

    @media (max-width: 768px) {
        .container {
            padding: 0.5rem;
        }

        .tabs {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.4rem;
        }

        .tabs button {
            min-width: 0;
            padding: 0.65rem 0.2rem;
            gap: 0.25rem;
            font-size: 0.75rem;
            white-space: nowrap;
        }

        .analysis-heading {
            align-items: flex-start;
            flex-direction: column;
        }

        .horizon-control {
            width: 100%;
            align-items: flex-start;
            flex-wrap: wrap;
        }

        .horizon-control label {
            width: 100%;
        }

        .horizon-presets {
            flex: 1;
        }

        .horizon-presets button {
            min-width: 44px;
            flex: 1;
        }

        .analysis-action .predict-btn {
            width: 100%;
        }

        .input-group {
            gap: 0.5rem;
        }

        .search-input {
            min-width: 0;
            box-sizing: border-box;
        }

        .conclusion-card,
        .model-section {
            padding: 0.85rem;
        }

        .summary-stats {
            grid-template-columns: 1fr;
        }

        .model-item,
        .prediction-settings {
            align-items: stretch;
            flex-direction: column;
        }

        .model-details,
        .model-actions {
            flex-wrap: wrap;
        }
    }
</style>
