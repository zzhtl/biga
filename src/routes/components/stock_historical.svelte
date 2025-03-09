<script lang="ts">
    let startDate = $state("");
    let endDate = $state("");
    let selectedSymbol = $state("AAPL");

    function fetchHistory() {
        // 这里可以添加实际的数据获取逻辑
        console.log(
            `Fetching ${selectedSymbol} from ${startDate} to ${endDate}`,
        );
    }
</script>

<div class="container">
    <h1>历史数据查询</h1>

    <div class="controls">
        <select bind:value={selectedSymbol}>
            <option value="AAPL">Apple (AAPL)</option>
            <option value="TSLA">Tesla (TSLA)</option>
            <option value="NVDA">NVIDIA (NVDA)</option>
        </select>

        <input type="date" bind:value={startDate} />
        <span>至</span>
        <input type="date" bind:value={endDate} />

        <button on:click={fetchHistory}>查询</button>
    </div>

    <div class="chart-container">
        <div class="chart-placeholder">
            📊 历史趋势图表（{selectedSymbol}）
        </div>
    </div>

    <div class="data-table">
        <div class="table-header">
            <div>日期</div>
            <div>开盘价</div>
            <div>收盘价</div>
            <div>最高价</div>
            <div>最低价</div>
        </div>
        <!-- 示例数据行 -->
        <div class="table-row">
            <div>2023-12-01</div>
            <div>$189.50</div>
            <div>$192.34</div>
            <div>$193.10</div>
            <div>$188.90</div>
        </div>
    </div>
</div>

<style>
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }

    .controls {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }

    select,
    input {
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        color: inherit;
    }

    button {
        padding: 0.5rem 2rem;
        background: var(--active-color);
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
    }

    .chart-container {
        margin: 2rem 0;
    }

    .chart-placeholder {
        height: 400px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
    }

    .data-table {
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .table-header,
    .table-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        padding: 1rem;
    }

    .table-header {
        background: rgba(255, 255, 255, 0.1);
        font-weight: 600;
    }

    .table-row {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    @media (max-width: 768px) {
        .controls {
            flex-direction: column;
        }

        .table-header,
        .table-row {
            grid-template-columns: repeat(3, 1fr);
        }

        .table-header div:nth-child(4),
        .table-row div:nth-child(4),
        .table-header div:nth-child(5),
        .table-row div:nth-child(5) {
            display: none;
        }
    }
</style>
