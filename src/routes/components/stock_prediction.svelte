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
    
    // 使用类型
    let modelList: ModelInfo[] = [];
    let predictions: Prediction[] = [];
    let modelAccuracy: number | null = null;
    let lastRealData: LastRealData | null = null; // 新增：最新真实数据
    
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
                    predictions = result;
                } else if ('predictions' in result && Array.isArray(result.predictions)) {
                    // 新格式，包含预测和最新真实数据
                    predictions = result.predictions;
                    if (result.last_real_data) {
                        lastRealData = {
                            date: result.last_real_data.date,
                            price: result.last_real_data.price,
                            change_percent: result.last_real_data.change_percent
                        };
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
    
    async function handleStockCodeChange() {
        await loadModelList();
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
        isTraining = true;
        errorMessage = "";
        
        try {
            await invoke('retrain_candle_model', { 
                modelId,
                epochs: epochs,
                batchSize: batchSize,
                learningRate: learningRate
            });
            alert(`模型 ${modelName} 重新训练成功`);
            await loadModelList();
        } catch (error) {
            errorMessage = `重新训练失败: ${error}`;
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
        <button class:active={useExistingModel} on:click={() => useExistingModel = true}>
            使用现有模型
        </button>
        <button class:active={!useExistingModel} on:click={() => useExistingModel = false}>
            训练新模型
        </button>
    </div>
    
    {#if useExistingModel}
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
                        开始预测
                    {/if}
                </button>
            </div>
        </div>
    {:else}
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
        padding: 1.5rem;
        border-radius: 1rem;
    }
    
    .prediction-table {
        overflow-x: auto;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    th {
        background: rgba(255, 255, 255, 0.05);
        font-weight: 600;
    }
    
    tr.positive td:nth-child(3) {
        color: #10b981;
    }
    
    tr.negative td:nth-child(3) {
        color: #ef4444;
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
</style>
