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
    
    interface Prediction {
        target_date: string;
        predicted_price: number;
        predicted_change_percent: number;
        confidence: number;
    }
    
    // 使用类型
    let modelList: ModelInfo[] = [];
    let predictions: Prediction[] = [];
    let modelAccuracy: number | null = null;
    
    // 模型训练参数
    let newModelName = "模型-" + new Date().toISOString().slice(0, 10);
    let modelType = "candle_mlp"; // 默认使用Candle的MLP模型
    let lookbackDays = 60; // 增加历史窗口
    let trainTestSplit = 0.8;
    let features = ["close", "volume", "change_percent", "ma5", "ma10", "ma20", "rsi", "macd", "bollinger", "stochastic_k", "stochastic_d", "momentum"];
    let epochs = 100; // 训练轮数
    let batchSize = 32; // 批处理大小
    let learningRate = 0.001; // 学习率
    let dropout = 0.2; // Dropout率
    let advancedOptions = false; // 是否显示高级选项

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

    async function trainModel() {
        if (!stockCode) {
            errorMessage = "请先输入股票代码";
            return;
        }

        isTraining = true;
        errorMessage = "";
        
        try {
            // 计算训练日期范围
            const endDate = new Date().toISOString().slice(0, 10);
            const startDateObj = new Date(Date.now() - lookbackDays * 24 * 60 * 60 * 1000);
            const startDate = startDateObj.toISOString().slice(0, 10);

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
            const metadata = result.metadata;
            modelAccuracy = result.accuracy;
            
            await loadModelList();
            useExistingModel = true;
            alert(`模型训练成功: ${metadata.name}, 准确率: ${(modelAccuracy * 100).toFixed(2)}%`);
        } catch (error) {
            errorMessage = `训练失败: ${error}`;
        } finally {
            isTraining = false;
        }
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
        
        try {
            const request = {
                stock_code: stockCode,
                model_name: useExistingModel ? selectedModelName : null,
                prediction_days: daysToPredict,
                use_candle: true // 使用Candle进行预测
            };
            const preds: Prediction[] = await invoke('predict_with_candle', { request });
            predictions = preds;
        } catch (error) {
            errorMessage = `预测失败: ${error}`;
            predictions = [];
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
                                    <span>创建时间：{new Date(model.created_at).toLocaleString()}</span>
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
                    <label>历史窗口天数:</label>
                    <input type="number" bind:value={lookbackDays} min="30" max="360" />
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
                    </div>
                </div>
                
                <button
                    on:click={trainModel}
                    class:loading={isTraining}
                    disabled={isTraining}
                >
                    {#if isTraining}
                        <span class="spinner"></span>
                    {:else}
                        开始训练
                    {/if}
                </button>
            </div>
        </div>
    {/if}
    
    {#if predictions && predictions.length > 0}
        <div class="prediction-results">
            <h2>预测结果</h2>
            <div class="prediction-table">
                <table>
                    <thead>
                        <tr>
                            <th>日期</th>
                            <th>预测价格</th>
                            <th>涨跌幅</th>
                            <th>置信度</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each predictions as prediction}
                            <tr class:positive={prediction.predicted_change_percent > 0} class:negative={prediction.predicted_change_percent < 0}>
                                <td>{new Date(prediction.target_date).toLocaleDateString()}</td>
                                <td>{prediction.predicted_price.toFixed(2)}</td>
                                <td>{prediction.predicted_change_percent.toFixed(2)}%</td>
                                <td>{(prediction.confidence * 100).toFixed(2)}%</td>
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
</style>
