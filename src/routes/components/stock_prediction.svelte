<script lang="ts">
    let stockCode = "";
    let prediction = "";
    let isLoading = false;

    async function predictStock() {
        isLoading = true;
        // 模拟API调用
        await new Promise((resolve) => setTimeout(resolve, 1000));
        prediction = `预测结果：${stockCode} 看涨`;
        isLoading = false;
    }
</script>

<div class="container">
    <h1>智能股票预测</h1>

    <div class="input-group">
        <input
            type="text"
            placeholder="输入股票代码（例如：AAPL）"
            bind:value={stockCode}
            class="search-input"
        />
        <button
            on:click={predictStock}
            class:loading={isLoading}
            disabled={isLoading}
        >
            {#if isLoading}
                <span class="spinner"></span>
            {:else}
                开始预测
            {/if}
        </button>
    </div>

    {#if prediction}
        <div class="prediction-card">
            <h3>{prediction}</h3>
            <div class="chart-container">
                <div class="chart-placeholder">
                    <span>趋势图表</span>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .container {
        max-width: 800px;
        margin: 0 auto;
    }

    h1 {
        font-size: 2rem;
        margin-bottom: 2rem;
        text-align: center;
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
        background: var(--active-color);
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: opacity 0.2s ease;
    }

    button:hover {
        opacity: 0.9;
    }

    button.loading {
        background: #4f46e5;
        cursor: not-allowed;
    }

    .spinner {
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

    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 2rem;
    }

    .chart-placeholder {
        height: 300px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        margin-top: 1.5rem;
    }
</style>
