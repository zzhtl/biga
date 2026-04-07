<script lang="ts">
    import { onMount } from 'svelte';
    import { invoke } from '@tauri-apps/api/core';
    import { confirm } from '@tauri-apps/plugin-dialog';
    
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
        accuracy: number;
        created_at: string;
    }
    
    interface ChartData {
        labels: string[];
        datasets: Array<{
            label: string;
            data: number[];
            borderColor: string;
            backgroundColor: string;
            fill?: boolean;
            tension?: number;
            yAxisID?: string;
        }>;
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
            // 后端某些场景可能缺少这个字段
            total_score?: number;
            factors: any[];
            signal_quality: any;
            operation_suggestion: string;
        };
    }
    
    // 新增：专业预测响应接口
    interface ProfessionalPredictionResponse {
        predictions: PredictionResult;
        professional_analysis: ProfessionalPrediction;
    }
    
    // 使用类型
    let modelList: ModelInfo[] = [];
    let predictions: Prediction[] = [];
    let modelAccuracy: number | null = null;
    let lastRealData: LastRealData | null = null; // 新增：最新真实数据
    let professionalAnalysis: ProfessionalPrediction | null = null; // 新增：专业分析结果
    let showProfessionalAnalysis = false; // 是否显示专业分析
    
    // 模型训练参数
    let newModelName = "模型-" + new Date().toISOString().slice(0, 10);
    let modelType = "candle_mlp"; // 默认使用Candle的MLP模型
    let lookbackDays = 180; // 修改为180天历史数据
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
    let predictionChart: ChartData | null = null;
    
    // 回测相关变量
    let backtestReport: BacktestReport | null = null;
    let isBacktesting = false;
    let showBacktestReport = false;
    let backtestStartDate = "";
    let backtestEndDate = "";
    let backtestInterval = 7; // 默认每7天进行一次预测
    let expandedEntryIndex: number | null = null; // 展开查看的回测条目索引
    
    // 纯技术分析预测（无需模型）- 默认展示
    let showTechnicalOnly = true;
    let technicalHistoryDays = 180; // 使用多少天历史数据
    let technicalPredictionDays = 7; // 预测未来多少天
    let isTechnicalPredicting = false;

    function normalizeNumber(value: any): number {
        // 统一处理后端返回的 number/string/undefined，避免模板渲染阶段直接异常。
        return Number(value);
    }

    function normalizePrediction(raw: any): Prediction {
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
                total_score: multiFactorScoreRaw?.total_score,
                factors: Array.isArray(multiFactorScoreRaw?.factors) ? multiFactorScoreRaw.factors : [],
                signal_quality: multiFactorScoreRaw?.signal_quality ?? null,
                operation_suggestion: String(multiFactorScoreRaw?.operation_suggestion ?? ""),
            },
        };
    }
    
    // 纯技术分析预测函数
    async function predictWithTechnicalOnly() {
        if (!stockCode) {
            errorMessage = "请先选择股票";
            return;
        }
        
        isTechnicalPredicting = true;
        errorMessage = "";
        predictions = [];
        professionalAnalysis = null;
        lastRealData = null;
        predictionChart = null;
        showProfessionalAnalysis = false;
        
        try {
            const request = {
                stock_code: stockCode,
                history_days: technicalHistoryDays,
                prediction_days: technicalPredictionDays
            };
            
            console.log('纯技术分析预测请求:', request);
            
            const result = await invoke<ProfessionalPredictionResponse>('predict_with_technical_only', { request });
            console.log('纯技术分析预测响应:', result);
            
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
                }
            }
            
            // 提取专业分析结果
            if (result.professional_analysis) {
                professionalAnalysis = normalizeProfessionalPrediction(result.professional_analysis);
                showProfessionalAnalysis = true;
            }
            
            // 生成图表数据
            if (predictions && predictions.length > 0) {
                generatePredictionChart(predictions);
                console.log("纯技术分析预测图表数据已生成");
            }
            
            if (predictions.length > 0 && professionalAnalysis) {
                const totalScoreRaw = professionalAnalysis.multi_factor_score?.total_score;
                const totalScore =
                    typeof totalScoreRaw === "number" ? totalScoreRaw : Number(totalScoreRaw);
                const totalScoreText = Number.isFinite(totalScore) ? `${totalScore.toFixed(1)}/100` : "—/100";

                await alert(
                    `✅ 纯技术分析预测成功！\n基于${technicalHistoryDays}天历史数据\n预测未来${technicalPredictionDays}天走势\n\n综合评分: ${totalScoreText}`
                );
            }
        } catch (error) {
            errorMessage = `纯技术分析预测失败: ${error}`;
            console.error('纯技术分析预测错误:', error);
            await alert(errorMessage);
            predictions = [];
            professionalAnalysis = null;
            lastRealData = null;
            predictionChart = null;
        } finally {
            isTechnicalPredicting = false;
        }
    }

    onMount(async () => {
        try {
            // 如果用户选择了股票代码，尝试加载模型列表
            if (stockCode) {
                await loadModelList();
            }
        } catch (error) {
            errorMessage = `加载失败: ${error}`;
        }
    });

    async function loadModelList() {
        if (!stockCode) return;
        
        try {
            const models: ModelInfo[] = await invoke('list_stock_prediction_models', { symbol: stockCode });
            modelList = models;
            if (modelList.length > 0) {
                selectedModelName = modelList[0].name;
            }
        } catch (error) {
            errorMessage = `加载模型列表失败: ${error}`;
            modelList = [];
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
        if (!stockCode) {
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
            // 使用180天训练数据 + 30天节假日缓冲期 = 210天总范围
            const totalDays = lookbackDays + 30; // 180 + 30 = 210天
            const startDateObj = new Date(Date.now() - totalDays * 24 * 60 * 60 * 1000);
            const startDate = startDateObj.toISOString().slice(0, 10);

            console.log(`📅 训练数据范围: ${startDate} 到 ${endDate} (${totalDays}天，含节假日缓冲)`);

            const trainRequest = {
                stock_code: stockCode,
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

            const result = await invoke<{metadata: ModelInfo, accuracy: number}>('train_candle_model', { request: trainRequest });
            
            clearInterval(progressInterval);
            trainingProgress = 100;
            
            const metadata = result.metadata;
            modelAccuracy = result.accuracy;
            
            // 添加训练完成日志
            trainingLogs = [...trainingLogs, {
                epoch: epochs,
                loss: "训练完成",
                timestamp: new Date().toLocaleTimeString(),
                accuracy: (modelAccuracy * 100).toFixed(2) + "%"
            }];
            
            await loadModelList();
            useExistingModel = true;
            alert(`模型训练成功: ${metadata.name}, 准确率: ${(modelAccuracy * 100).toFixed(2)}%`);
            
            // 自动加载模型对比数据
            await loadModelComparison();
            
        } catch (error) {
            clearInterval(progressInterval);
            errorMessage = `训练失败: ${error}`;
        } finally {
            isTraining = false;
        }
    }

    // 加载模型性能对比数据
    async function loadModelComparison() {
        if (!stockCode) return;
        
        try {
            const models = await invoke('list_stock_prediction_models', { symbol: stockCode }) as ModelInfo[];
            modelComparison = models.map((model: ModelInfo) => ({
                name: model.name,
                type: model.model_type,
                accuracy: model.accuracy * 100,
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

    // 生成预测图表数据
    function generatePredictionChart(predictions: Prediction[]) {
        if (!predictions || predictions.length === 0) {
            console.log("没有预测数据，无法生成图表");
            return;
        }
        
        console.log("生成预测图表，数据量:", predictions.length);
        
        const chartData: ChartData = {
            labels: predictions.map((p: Prediction) => {
                const date = new Date(p.target_date);
                return `${date.getMonth() + 1}/${date.getDate()}`;
            }),
            datasets: [{
                label: '预测价格',
                data: predictions.map((p: Prediction) => p.predicted_price),
                borderColor: 'rgb(79, 70, 229)',
                backgroundColor: 'rgba(79, 70, 229, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '置信度 (%)',
                data: predictions.map((p: Prediction) => p.confidence * 100),
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                yAxisID: 'y1',
                fill: false
            }]
        };
        
        predictionChart = chartData;
        console.log("预测图表数据已生成:", predictionChart);
    }

    async function predictStock() {
        if (!stockCode) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!selectedModelName && useExistingModel) {
            errorMessage = "请先选择模型或训练新模型";
            return;
        }
        
        isPredicting = true;
        errorMessage = "";
        predictionChart = null; // 重置图表数据
        showProfessionalAnalysis = false; // 重置专业分析显示
        
        try {
            const request = {
                stock_code: stockCode,
                model_name: useExistingModel ? selectedModelName : null,
                prediction_days: daysToPredict,
                use_candle: true
            };
            
            console.log("发送预测请求:", request);
            const result = await invoke<Prediction[] | PredictionResult>('predict_with_candle', { request });
            console.log("收到预测结果:", result);
            
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
                }
            } else {
                predictions = [];
                lastRealData = null;
            }
            
            // 生成图表数据
            if (predictions && predictions.length > 0) {
                generatePredictionChart(predictions);
                console.log("图表数据生成完成:", predictionChart);
            } else {
                console.warn("预测结果为空，无法生成图表");
            }
            
        } catch (error) {
            console.error("预测失败:", error);
            errorMessage = `预测失败: ${error}`;
            predictions = [];
            lastRealData = null;
            predictionChart = null;
        } finally {
            isPredicting = false;
        }
    }
    
    // 新增：使用专业策略预测
    async function predictWithProfessionalStrategy() {
        if (!stockCode) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!selectedModelName && useExistingModel) {
            errorMessage = "请先选择模型或训练新模型";
            return;
        }
        
        isPredicting = true;
        errorMessage = "";
        predictionChart = null;
        professionalAnalysis = null;
        showProfessionalAnalysis = false;
        
        try {
            const request = {
                stock_code: stockCode,
                model_name: useExistingModel ? selectedModelName : null,
                prediction_days: daysToPredict,
                use_candle: true
            };
            
            console.log("发送专业预测请求:", request);
            const result = await invoke<ProfessionalPredictionResponse>('predict_with_professional_strategy', { request });
            console.log("收到专业预测结果:", result);
            
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
                    }
                }
                
                // 提取专业分析结果
                if (result.professional_analysis) {
                    professionalAnalysis = normalizeProfessionalPrediction(result.professional_analysis);
                    showProfessionalAnalysis = true;
                    console.log("专业分析结果:", professionalAnalysis);
                }
                
                // 生成图表数据
                if (predictions && predictions.length > 0) {
                    generatePredictionChart(predictions);
                }
            }
            
        } catch (error) {
            console.error("专业预测失败:", error);
            errorMessage = `专业预测失败: ${error}`;
            predictions = [];
            lastRealData = null;
            professionalAnalysis = null;
            predictionChart = null;
        } finally {
            isPredicting = false;
        }
    }
    
    async function handleStockCodeChange() {
        await loadModelList();
    }
    
    // 监听stockCode变化，自动刷新模型列表
    $: if (stockCode) {
        loadModelList();
    }
    
    async function deleteModel(modelId: string) {
        // 使用 Tauri 对话框进行确认
        const confirmed = await confirm('确定要删除此模型吗？', { title: '删除模型' });
        if (!confirmed) {
            return;
        }
        try {
            await invoke('delete_stock_prediction_model', { modelId });
            await loadModelList();
            alert('模型删除成功');
        } catch (error) {
            errorMessage = `删除失败: ${error}`;
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
            `确定要重新训练模型 "${modelName}" 吗？\n\n训练参数:\n- 训练轮数: ${epochs}\n- 批次大小: ${batchSize}\n- 学习率: ${learningRate}`,
            { title: '重新训练模型' }
        );
        
        if (!confirmed) {
            return;
        }
        
        isTraining = true;
        errorMessage = "";
        
        try {
            await invoke('retrain_candle_model', { 
                modelId,
                epochs: epochs,
                batchSize: batchSize,
                learningRate: learningRate
            });
            await alert(`模型 ${modelName} 重新训练成功！`);
            await loadModelList();
        } catch (error) {
            errorMessage = `重新训练失败: ${error}`;
            await alert(errorMessage);
        } finally {
            isTraining = false;
        }
    }

    // 获取模型评估信息
    async function evaluateModel(modelId: string) {
        try {
            const result = await invoke<{accuracy: number, confusion_matrix: any}>('evaluate_candle_model', { modelId });
            const accuracy = result.accuracy;
            const confusion_matrix = result.confusion_matrix;
            
            alert(`模型评估结果:\n准确率: ${(accuracy * 100).toFixed(2)}%\n混淆矩阵: ${JSON.stringify(confusion_matrix)}`);
        } catch (error) {
            errorMessage = `评估失败: ${error}`;
        }
    }
    
    // 切换高级选项显示
    function toggleAdvancedOptions() {
        advancedOptions = !advancedOptions;
    }
    
    // 执行回测
    async function runBacktest() {
        if (!stockCode) {
            errorMessage = "请先输入股票代码";
            return;
        }
        
        if (!backtestStartDate || !backtestEndDate) {
            errorMessage = "请选择回测日期范围";
            return;
        }
        
        if (!selectedModelName && useExistingModel) {
            errorMessage = "请先选择模型";
            return;
        }
        
        isBacktesting = true;
        errorMessage = "";
        
        try {
            const backtestRequest: BacktestRequest = {
                stock_code: stockCode,
                model_name: useExistingModel ? selectedModelName : undefined,
                start_date: backtestStartDate,
                end_date: backtestEndDate,
                prediction_days: daysToPredict,
                backtest_interval: backtestInterval
            };
            
            const result = await invoke<BacktestReport>('run_model_backtest', { request: backtestRequest });
            backtestReport = result;
            showBacktestReport = true;
            
            console.log("回测结果:", result);
            
        } catch (error) {
            errorMessage = `回测失败: ${error}`;
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
    <h1>智能股票预测</h1>

    <div class="input-group">
        <input
            type="text"
            placeholder="输入股票代码（例如：sh000001）"
            bind:value={stockCode}
            on:change={handleStockCodeChange}
            class="search-input"
        />
    </div>
    
    {#if errorMessage}
        <div class="error-message">
            {errorMessage}
        </div>
    {/if}
    
    <div class="tabs">
        <button class:active={showTechnicalOnly} on:click={() => {showTechnicalOnly = true; showBacktestReport = false; useExistingModel = false;}}>
            📊 纯技术分析
        </button>
        <button class:active={useExistingModel && !showBacktestReport && !showTechnicalOnly} on:click={() => {useExistingModel = true; showBacktestReport = false; showTechnicalOnly = false;}}>
            使用现有模型
        </button>
        <button class:active={!useExistingModel && !showBacktestReport && !showTechnicalOnly} on:click={() => {useExistingModel = false; showBacktestReport = false; showTechnicalOnly = false;}}>
            训练新模型
        </button>
        <button class:active={showBacktestReport} on:click={() => {showBacktestReport = true; showTechnicalOnly = false; setDefaultBacktestDates();}}>
            回测报告
        </button>
    </div>
    
    {#if showTechnicalOnly}
        <div class="model-section">
            <h2>📊 纯技术分析预测（无需模型训练）</h2>
            <p class="section-desc">
                基于历史数据的纯技术指标分析，无需训练模型即可预测。
                包含：多因子评分、支撑压力位、多周期共振、量价背离、K线形态等专业分析。
            </p>
            
            <div class="prediction-settings">
                <div class="form-group">
                    <label>历史数据天数：</label>
                    <input type="number" bind:value={technicalHistoryDays} min="60" max="365" step="10" />
                    <small>建议120-250天，数据越多越准确</small>
                </div>
                
                <div class="form-group">
                    <label>预测天数：</label>
                    <input type="number" bind:value={technicalPredictionDays} min="1" max="30" />
                    <small>预测未来1-30天的走势</small>
                </div>
                
                <button
                    on:click={predictWithTechnicalOnly}
                    class:loading={isTechnicalPredicting}
                    disabled={isTechnicalPredicting || !stockCode}
                    class="predict-btn"
                >
                    {isTechnicalPredicting ? '分析中...' : '🔮 开始预测'}
                </button>
            </div>
            
            <div class="info-box">
                <h4>💡 纯技术分析优势</h4>
                <ul>
                    <li>✅ <strong>无需模型训练</strong> - 直接基于历史数据分析</li>
                    <li>✅ <strong>实时响应</strong> - 几秒钟即可得到结果</li>
                    <li>✅ <strong>金融级指标</strong> - RSI、MACD、KDJ、ATR、ADX等</li>
                    <li>✅ <strong>多维度分析</strong> - 趋势、量能、形态、情绪综合评估</li>
                    <li>✅ <strong>智能买卖点</strong> - 自动识别支撑压力位和交易机会</li>
                </ul>
            </div>
        </div>
    {:else if useExistingModel && !showBacktestReport}
        <div class="model-section">
            <h2>选择预测模型</h2>
            {#if modelList.length === 0}
                <p>没有可用的模型，请先训练一个模型。</p>
            {:else}
                <div class="model-list">
                    {#each modelList as model}
                        <div class="model-item" class:selected={selectedModelName === model.name}>
                            <div class="model-info" on:click={() => selectedModelName = model.name}>
                                <h3>{model.name}</h3>
                                <div class="model-details">
                                    <span>类型：{model.model_type}</span>
                                    <span>准确率：{(model.accuracy * 100).toFixed(2)}%</span>
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
                    <input type="number" bind:value={daysToPredict} min="1" max="30" />
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
                    style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"
                >
                    {#if isPredicting}
                        <span class="spinner"></span>
                    {:else}
                        💎 金融级预测
                    {/if}
                </button>
            </div>
        </div>
    {:else if !useExistingModel && !showBacktestReport}
        <div class="model-section">
            <h2>训练新模型 (Candle)</h2>
            
            <div class="training-form">
                <div class="form-group">
                    <label>模型名称:</label>
                    <input type="text" bind:value={newModelName} />
                </div>
                
                <div class="form-group">
                    <label>模型类型:</label>
                    <select bind:value={modelType}>
                        <option value="candle_mlp">Candle多层感知机</option>
                        <option value="candle_lstm">Candle LSTM</option>
                        <option value="candle_gru">Candle GRU</option>
                        <option value="candle_transformer">Candle Transformer</option>
                        <option value="linear">线性回归</option>
                        <option value="decision_tree">决策树</option>
                        <option value="svm">支持向量机</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>历史窗口天数 (实际查询范围+30天节假日缓冲):</label>
                    <input type="number" bind:value={lookbackDays} min="60" max="365" step="30" />
                    <small style="color: rgba(255,255,255,0.6); font-size: 0.8rem;">
                        推荐: 180天 (约6个月交易数据)，实际查询 {lookbackDays + 30} 天
                    </small>
                </div>
                
                <div class="form-group">
                    <label>训练/测试集分割比例:</label>
                    <input type="number" bind:value={trainTestSplit} min="0.5" max="0.9" step="0.1" />
                </div>
                
                <button type="button" class="advanced-btn" on:click={toggleAdvancedOptions}>
                    {advancedOptions ? '隐藏高级选项' : '显示高级选项'}
                </button>
                
                {#if advancedOptions}
                    <div class="advanced-options">
                        <div class="form-group">
                            <label>训练轮数(Epochs):</label>
                            <input type="number" bind:value={epochs} min="10" max="1000" />
                        </div>
                        
                        <div class="form-group">
                            <label>批处理大小(Batch Size):</label>
                            <input type="number" bind:value={batchSize} min="8" max="128" />
                        </div>
                        
                        <div class="form-group">
                            <label>学习率(Learning Rate):</label>
                            <input type="number" bind:value={learningRate} min="0.0001" max="0.1" step="0.0001" />
                        </div>
                        
                        <div class="form-group">
                            <label>Dropout率:</label>
                            <input type="number" bind:value={dropout} min="0" max="0.5" step="0.1" />
                        </div>
                    </div>
                {/if}
                
                <div class="form-group features-list">
                    <label>特征选择:</label>
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
                                        <th>准确率</th>
                                        <th>创建时间</th>
                                        <th>性能条</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {#each modelComparison as model}
                                        <tr>
                                            <td>{model.name}</td>
                                            <td>{model.type}</td>
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
                    <label>选择模型:</label>
                    <select bind:value={selectedModelName} disabled={modelList.length === 0}>
                        <option value="">请选择模型</option>
                        {#each modelList as model}
                            <option value={model.name}>{model.name} (准确率: {(model.accuracy * 100).toFixed(2)}%)</option>
                        {/each}
                    </select>
                </div>
                
                <div class="form-group">
                    <label>回测开始日期:</label>
                    <input type="date" bind:value={backtestStartDate} />
                </div>
                
                <div class="form-group">
                    <label>回测结束日期:</label>
                    <input type="date" bind:value={backtestEndDate} />
                </div>
                
                <div class="form-group">
                    <label>预测天数:</label>
                    <input type="number" bind:value={daysToPredict} min="1" max="10" />
                </div>
                
                <div class="form-group">
                    <label>回测间隔(天):</label>
                    <input type="number" bind:value={backtestInterval} min="1" max="30" />
                    <small>每隔几天进行一次预测</small>
                </div>
                
                <button
                    on:click={runBacktest}
                    class:loading={isBacktesting}
                    disabled={isBacktesting || !selectedModelName}
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
                            <h4>总体准确率</h4>
                            <div class="summary-stats">
                                <div class="stat-item">
                                    <span class="stat-label">价格准确率:</span>
                                    <span class="stat-value">{(backtestReport.overall_price_accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">方向准确率:</span>
                                    <span class="stat-value">{(backtestReport.overall_direction_accuracy * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">平均误差:</span>
                                    <span class="stat-value">{(backtestReport.average_prediction_error * 100).toFixed(2)}%</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">总预测次数:</span>
                                    <span class="stat-value">{backtestReport.total_predictions}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                                        <!-- 准确率趋势图 -->
                    <div class="accuracy-trend">
                        <h4>准确率趋势</h4>
                        <div class="trend-chart">
                            <div class="trend-bars">
                                {#each backtestReport.accuracy_trend as accuracy, i}
                                    <div class="trend-bar" style="height: {accuracy * 100}px;" title="第{i + 1}次回测: {(accuracy * 100).toFixed(1)}%">
                                        <div class="trend-bar-fill" style="background-color: rgb(34, 197, 94);"></div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    </div>
                    
                    <!-- 详细回测记录 -->
                    <div class="backtest-details">
                        <h4>详细回测记录</h4>
                        <div class="backtest-table">
                            <table>
                                <thead>
                                    <tr>
                                        <th>预测日期</th>
                                        <th>价格准确率</th>
                                        <th>方向准确率</th>
                                        <th>平均误差</th>
                                        <th>预测次数</th>
                                        <th>详情</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {#each backtestReport.backtest_entries as entry, i}
                                        <tr>
                                            <td>{new Date(entry.prediction_date).toLocaleDateString()}</td>
                                            <td class="accuracy-cell">
                                                <div class="accuracy-bar">
                                                    <div class="accuracy-fill" style="width: {entry.price_accuracy * 100}%"></div>
                                                    <span>{(entry.price_accuracy * 100).toFixed(1)}%</span>
                                                </div>
                                            </td>
                                            <td class="accuracy-cell">
                                                <div class="accuracy-bar">
                                                    <div class="accuracy-fill direction-fill" style="width: {entry.direction_accuracy * 100}%"></div>
                                                    <span>{(entry.direction_accuracy * 100).toFixed(1)}%</span>
                                                </div>
                                            </td>
                                            <td>{(entry.avg_prediction_error * 100).toFixed(2)}%</td>
                                            <td>{entry.predictions.length}</td>
                                            <td>
                                                <button class="action-btn" type="button" on:click={() => expandedEntryIndex = expandedEntryIndex === i ? null : i}>
                                                    {expandedEntryIndex === i ? '收起' : '详情'}
                                                </button>
                                            </td>
                                        </tr>
                                        {#if expandedEntryIndex === i}
                                            <tr class="details-row">
                                                <td colspan="6">
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
            <h2>预测结果</h2>
            
            <!-- 新增：最新真实数据展示 -->
            {#if lastRealData}
                <div class="last-real-data">
                    <h3>最新真实数据</h3>
                    <div class="real-data-card">
                        <div class="real-data-date">{new Date(lastRealData.date).toLocaleDateString()}</div>
                        <div class="real-data-price">{lastRealData.price.toFixed(2)}</div>
                        <div class="real-data-change {lastRealData.change_percent >= 0 ? 'price-up' : 'price-down'}">
                            {lastRealData.change_percent >= 0 ? '+' : ''}{lastRealData.change_percent.toFixed(2)}%
                        </div>
                    </div>
                </div>
            {/if}
            
            <!-- 新增：专业分析结果展示 -->
            {#if showProfessionalAnalysis && professionalAnalysis}
                <div class="professional-analysis">
                    <h3>🎯 金融级策略分析</h3>
                    
                    <!-- 操作建议 -->
                    <div class="advice-card">
                        <div class="advice-content">
                            <strong>操作建议：</strong>{professionalAnalysis.current_advice}
                        </div>
                        <div class="risk-indicator">
                            <strong>风险等级：</strong>
                            <span class="risk-level {professionalAnalysis.risk_level.includes('低') ? 'low' : professionalAnalysis.risk_level.includes('高') ? 'high' : 'medium'}">
                                {professionalAnalysis.risk_level}
                            </span>
                        </div>
                    </div>
                    
                    <!-- 买入点信号 -->
                    {#if professionalAnalysis.buy_points && professionalAnalysis.buy_points.length > 0}
                        <div class="buy-sell-section buy-section">
                            <h4>💚 买入点信号 ({professionalAnalysis.buy_points.length}个)</h4>
                            <div class="signals-grid">
                                {#each professionalAnalysis.buy_points as buyPoint, index}
                                    <div class="signal-card buy-card">
                                        <div class="signal-header">
                                            <span class="signal-type">{buyPoint.point_type}</span>
                                            <span class="signal-strength">强度: {buyPoint.signal_strength.toFixed(0)}分</span>
                                        </div>
                                        <div class="signal-body">
                                            <div class="signal-row">
                                                <span class="label">💵 建议买入价:</span>
                                                <span class="value price-value">{buyPoint.price_level.toFixed(2)}元</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">🛡️ 止损位(跌至卖出):</span>
                                                <span class="value stop-loss">{buyPoint.stop_loss.toFixed(2)}元 (↓{Math.abs((buyPoint.stop_loss - buyPoint.price_level) / buyPoint.price_level * 100).toFixed(2)}%)</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">🎯 止盈位(涨至卖出):</span>
                                                <span class="value take-profit">
                                                    {#each buyPoint.take_profit as tp, i}
                                                        {tp.toFixed(2)}元(↑{((tp - buyPoint.price_level) / buyPoint.price_level * 100).toFixed(2)}%)
                                                        {#if i < buyPoint.take_profit.length - 1}, {/if}
                                                    {/each}
                                                </span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">风险收益比:</span>
                                                <span class="value ratio">1:{buyPoint.risk_reward_ratio.toFixed(2)}</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">置信度:</span>
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
                            <p>🟡 当前无明确买入信号，建议观望等待更好机会</p>
                        </div>
                    {/if}
                    
                    <!-- 卖出点信号 -->
                    {#if professionalAnalysis.sell_points && professionalAnalysis.sell_points.length > 0}
                        <div class="buy-sell-section sell-section">
                            <h4>🔴 卖出点信号 ({professionalAnalysis.sell_points.length}个)</h4>
                            <div class="signals-grid">
                                {#each professionalAnalysis.sell_points as sellPoint, index}
                                    <div class="signal-card sell-card">
                                        <div class="signal-header">
                                            <span class="signal-type">{sellPoint.point_type}</span>
                                            <span class="signal-strength">强度: {sellPoint.signal_strength.toFixed(0)}分</span>
                                        </div>
                                        <div class="signal-body">
                                            <div class="signal-row">
                                                <span class="label">💰 建议卖出价:</span>
                                                <span class="value price-value">{sellPoint.price_level.toFixed(2)}元</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">🛡️ 止损位(跌至卖出):</span>
                                                <span class="value stop-loss">{sellPoint.stop_loss.toFixed(2)}元 (↓{Math.abs((sellPoint.stop_loss - sellPoint.price_level) / sellPoint.price_level * 100).toFixed(2)}%)</span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">🎯 止盈位(涨至卖出):</span>
                                                <span class="value take-profit">
                                                    {#each sellPoint.take_profit as tp, i}
                                                        {tp.toFixed(2)}元(↑{((tp - sellPoint.price_level) / sellPoint.price_level * 100).toFixed(2)}%)
                                                        {#if i < sellPoint.take_profit.length - 1}, {/if}
                                                    {/each}
                                                </span>
                                            </div>
                                            <div class="signal-row">
                                                <span class="label">置信度:</span>
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
                            <p>🟡 当前无明确卖出信号</p>
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
            
            <!-- 预测图表 -->
            {#if predictionChart}
                <div class="prediction-chart">
                    <h3>预测趋势图</h3>
                    <div class="chart-container">
                        <!-- 图表图例 -->
                        <div class="chart-legend">
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: rgb(79, 70, 229);"></div>
                                <span>预测价格</span>
                            </div>
                            <div class="legend-item">
                                <div class="legend-color" style="background-color: rgb(34, 197, 94);"></div>
                                <span>置信度</span>
                            </div>
                        </div>
                        
                        <!-- SVG图表 -->
                        <div class="svg-chart-container">
                            <svg width="100%" height="300" viewBox="0 0 800 300" style="border: 1px solid rgba(255,255,255,0.2); border-radius: 0.5rem; background: rgba(0,0,0,0.1);">
                                {#if predictionChart && predictionChart.datasets[0].data.length > 0}
                                    {@const priceData = predictionChart.datasets[0].data}
                                    {@const confidenceData = predictionChart.datasets[1].data}
                                    {@const minPrice = Math.min(...priceData)}
                                    {@const maxPrice = Math.max(...priceData)}
                                    {@const priceRange = maxPrice - minPrice || 1}
                                    
                                    <!-- 网格线 -->
                                    {#each [0, 1, 2, 3, 4] as i}
                                        <line x1="60" y1={50 + i * 40} x2="750" y2={50 + i * 40} stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
                                    {/each}
                                    {#each predictionChart.labels as label, i}
                                        {@const x = 60 + (i / (predictionChart.labels.length - 1)) * 690}
                                        <line x1={x} y1="50" x2={x} y2="250" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
                                    {/each}
                                    
                                    <!-- 价格曲线 -->
                                    {@const pathD = priceData.map((price, i) => {
                                        const x = 60 + (i / (priceData.length - 1)) * 690;
                                        const y = 250 - ((price - minPrice) / priceRange) * 180;
                                        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                                    }).join(' ')}
                                    <path d={pathD} fill="none" stroke="rgb(79, 70, 229)" stroke-width="3"/>
                                    
                                    <!-- 置信度柱状图 -->
                                    {#each confidenceData as confidence, i}
                                        {@const x = 60 + (i / (confidenceData.length - 1)) * 690}
                                        {@const height = (confidence / 100) * 60}
                                        <rect x={x - 8} y={250 - height} width="16" height={height} fill="rgba(34, 197, 94, 0.6)" rx="2"/>
                                    {/each}
                                    
                                    <!-- 数据点 -->
                                    {#each priceData as price, i}
                                        {@const x = 60 + (i / (priceData.length - 1)) * 690}
                                        {@const y = 250 - ((price - minPrice) / priceRange) * 180}
                                        <circle cx={x} cy={y} r="5" fill="rgb(79, 70, 229)" stroke="white" stroke-width="2"/>
                                        <text x={x} y={y - 15} text-anchor="middle" fill="white" font-size="11" font-weight="bold">{price.toFixed(1)}</text>
                                    {/each}
                                    
                                    <!-- 日期标签 -->
                                    {#each predictionChart.labels as label, i}
                                        {@const x = 60 + (i / (predictionChart.labels.length - 1)) * 690}
                                        <text x={x} y={275} text-anchor="middle" fill="rgba(255,255,255,0.8)" font-size="12">{label}</text>
                                    {/each}
                                    
                                    <!-- Y轴价格标签 -->
                                    {#each [0, 1, 2, 3, 4] as i}
                                        {@const price = minPrice + (priceRange * (4 - i) / 4)}
                                        <text x="50" y={55 + i * 40} text-anchor="end" fill="rgba(255,255,255,0.8)" font-size="11">{price.toFixed(1)}</text>
                                    {/each}
                                    
                                    <!-- 坐标轴标题 -->
                                    <text x="400" y="295" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-size="12" font-weight="bold">预测日期</text>
                                    <text x="25" y="150" text-anchor="middle" fill="rgba(255,255,255,0.8)" font-size="12" font-weight="bold" transform="rotate(-90 25 150)">价格 (元)</text>
                                {/if}
                            </svg>
                        </div>
                        
                        <!-- 备用简单图表 -->
                        <div class="simple-chart-backup" style="margin-top: 1rem;">
                            <div class="chart-bars">
                                {#each predictions as prediction, index}
                                    <div class="chart-bar-item">
                                        <div class="bar-container">
                                            <div 
                                                class="price-bar" 
                                                style="height: {(prediction.predicted_price / Math.max(...predictions.map(p => p.predicted_price))) * 100}px; background-color: rgb(79, 70, 229);"
                                                title="预测价格: {prediction.predicted_price.toFixed(2)}"
                                            ></div>
                                            <div 
                                                class="confidence-bar" 
                                                style="height: {prediction.confidence * 80}px; background-color: rgb(34, 197, 94); margin-left: 5px;"
                                                title="置信度: {(prediction.confidence * 100).toFixed(2)}%"
                                            ></div>
                                        </div>
                                        <div class="bar-label">{new Date(prediction.target_date).toLocaleDateString().slice(5)}</div>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    </div>
                </div>
            {:else}
                <!-- 调试信息：显示为什么图表没有生成 -->
                <div class="debug-info" style="background: rgba(255,0,0,0.1); padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h4>图表调试信息</h4>
                    <p>预测数据长度: {predictions ? predictions.length : 0}</p>
                    <p>图表数据状态: {predictionChart ? '已生成' : '未生成'}</p>
                    {#if predictions && predictions.length > 0}
                        <p>第一个预测: {JSON.stringify(predictions[0])}</p>
                    {/if}
                </div>
            {/if}
            
            <!-- 预测统计信息 -->
            <div class="prediction-stats">
                <div class="stat-card">
                    <h4>平均预测涨幅</h4>
                    <div class="stat-value {predictions.reduce((sum, p) => sum + p.predicted_change_percent, 0) / predictions.length > 0 ? 'positive' : 'negative'}">
                        {(predictions.reduce((sum, p) => sum + p.predicted_change_percent, 0) / predictions.length).toFixed(2)}%
                    </div>
                </div>
                <div class="stat-card">
                    <h4>平均置信度</h4>
                    <div class="stat-value">
                        {(predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(2)}%
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
            
            <div class="prediction-table">
                <table>
                    <thead>
                        <tr>
                            <th>日期</th>
                            <th>预测价格</th>
                            <th>涨跌幅</th>
                            <th>置信度</th>
                            <th>交易信号</th>
                            <th>预测理由</th>
                            <th>技术指标</th>
                            <th>风险评级</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each predictions as prediction}
                            <tr class:positive={prediction.predicted_change_percent > 0} class:negative={prediction.predicted_change_percent < 0}>
                                <td>{new Date(prediction.target_date).toLocaleDateString()}</td>
                                <td>{prediction.predicted_price.toFixed(2)}</td>
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
                                    <span class="signal-badge {prediction.trading_signal?.includes('买入') ? 'buy-signal' : prediction.trading_signal?.includes('卖出') ? 'sell-signal' : 'hold-signal'}">
                                        {prediction.trading_signal || '持有'}
                                    </span>
                                    {#if prediction.signal_strength}
                                        <div class="signal-strength">
                                            强度: {(prediction.signal_strength * 100).toFixed(0)}%
                                        </div>
                                    {/if}
                                </td>
                                <td class="prediction-reason-cell">
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
                                </td>
                                <td>
                                    {#if prediction.technical_indicators}
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
                                    {/if}
                                </td>
                                <td>
                                    <span class="risk-badge {prediction.confidence > 0.8 ? 'low-risk' : prediction.confidence > 0.6 ? 'medium-risk' : 'high-risk'}">
                                        {prediction.confidence > 0.8 ? '低风险' : prediction.confidence > 0.6 ? '中风险' : '高风险'}
                                    </span>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 1rem;
    }

    h1 {
        font-size: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    h2 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .input-group {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
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
        border-radius: 1rem;
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
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem; /* 减小内边距 */
        border-radius: 1rem;
        width: 100%;
        box-sizing: border-box;
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
    .prediction-reason-cell {
        font-size: 0.65rem; /* 更小 */
    }
    
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
        border-radius: 1rem;
    }
    
    .chart-container {
        width: 100%;
    }
    
    .svg-chart-container {
        width: 100%;
        margin: 1rem 0;
    }
    
    .chart-legend {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        margin-bottom: 1rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .legend-color {
        width: 1rem;
        height: 1rem;
        border-radius: 0.25rem;
    }
    
    .simple-chart-backup {
        background: rgba(0, 0, 0, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .chart-bars {
        display: flex;
        justify-content: space-around;
        align-items: flex-end;
        gap: 0.5rem;
        height: 120px;
    }
    
    .chart-bar-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bar-container {
        display: flex;
        align-items: flex-end;
        height: 100px;
    }
    
    .price-bar,
    .confidence-bar {
        width: 12px;
        border-radius: 2px 2px 0 0;
    }
    
    .bar-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
    }
    
    .debug-info {
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .debug-info h4 {
        margin: 0 0 0.5rem 0;
        color: #ef4444;
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
        color: #10b981;
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
    
    /* 新增：最新真实数据样式 */
    .last-real-data {
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .last-real-data h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .real-data-card {
        display: flex;
        align-items: center;
        gap: 2rem;
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }
    
    .real-data-date {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.8);
    }
    
    .real-data-price {
        font-size: 1.5rem;
        font-weight: bold;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .real-data-change {
        font-size: 1.3rem;
        font-weight: bold;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
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
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 1rem;
    }
    
    .professional-analysis h3 {
        margin-top: 0;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .advice-card {
        background: rgba(0, 0, 0, 0.3);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
    
    .advice-content {
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .risk-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
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
    
    .signal-row .take-profit {
        color: #ef4444;  /* 止盈(上涨)用红色 - 符合中国股市习惯 */
    }
    
    .signal-row .ratio {
        color: #667eea;
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
    .prediction-reason-cell {
        min-width: 200px;
        max-width: 300px;
    }
    
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
    
    .factor-tag:has(⚠️) {
        background: rgba(251, 191, 36, 0.2);
        border-color: rgba(251, 191, 36, 0.4);
        color: #fde047;
    }
    
    .factor-tag:has(✅) {
        background: rgba(16, 185, 129, 0.2);
        border-color: rgba(16, 185, 129, 0.4);
        color: #6ee7b7;
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
    
    .info-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .info-box h4 {
        color: #818cf8;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .info-box ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .info-box li {
        padding: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
        line-height: 1.6;
    }
    
    .info-box li strong {
        color: #a5b4fc;
    }
    
    .predict-btn {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
    }
</style>
