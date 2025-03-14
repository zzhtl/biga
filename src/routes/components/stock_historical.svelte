<script lang="ts">
    import { onMount } from "svelte";
    import { invoke } from "@tauri-apps/api/core";

    let startDate = $state("");
    let endDate = $state("");
    let selectedSymbol = $state("AAPL");
    let stockSymbols = $state<
        Array<{ symbol: string; name: string; exchange: string }>
    >([]);
    let searchQuery = $state("");
    let isDropdownOpen = $state(false);

    // 新增类型定义
    type StockInfo = {
        symbol: string;
        name: string;
        exchange: string;
    };

    const filteredSymbols = $derived(
        stockSymbols.filter((stock) => {
            const query = searchQuery.toLowerCase();
            return (
                stock.symbol.toLowerCase().includes(query) ||
                stock.name.toLowerCase().includes(query) ||
                stock.exchange.toLowerCase().includes(query)
            );
        }),
    );

    function handleSelect(stock: StockInfo) {
        // 修改参数类型
        selectedSymbol = stock.symbol;
        searchQuery = `${stock.symbol} - ${stock.name}`; // 显示完整信息
        isDropdownOpen = false;
    }

    // 修改后的 refreshStockSymbols 函数
    async function refreshStockSymbols() {
        try {
            await invoke("refresh_stock_infos");

            // 使用泛型指定返回类型
            const symbols = await invoke<StockInfo[]>("get_stock_infos");

            stockSymbols = symbols;
        } catch (error) {
            console.error("Failed to refresh stock symbols:", error);
        }
    }

    // 同时需要修改 onMount 中的调用
    onMount(async () => {
        try {
            // 使用泛型指定返回类型
            const symbols = await invoke<StockInfo[]>("get_stock_infos");
            stockSymbols = symbols;

            if (stockSymbols.length > 0) {
                selectedSymbol = stockSymbols[0].symbol;
            }
        } catch (error) {
            console.error("Failed to fetch stock symbols:", error);
        }
    });

    // 新增文档点击处理函数
    function handleDocumentClick(event: MouseEvent) {
        const target = event.target as HTMLElement;
        const isInside = target.closest(".custom-select");
        if (!isInside && isDropdownOpen) {
            isDropdownOpen = false;
        }
    }

    // 自动清理的effect
    $effect(() => {
        if (isDropdownOpen) {
            document.addEventListener("click", handleDocumentClick);
        } else {
            document.removeEventListener("click", handleDocumentClick);
        }

        return () => {
            document.removeEventListener("click", handleDocumentClick);
        };
    });

    async function refreshHistory() {
        try {
            await invoke("refresh_historical_data", { symbol: selectedSymbol });
        } catch (error) {
            console.error("Failed to refresh history data:", error);
        }
    }

    function fetchHistory() {
        console.log(
            `Fetching ${selectedSymbol} from ${startDate} to ${endDate}`,
        );
    }
</script>

<div class="container">
    <h1>历史数据查询</h1>

    <div class="controls">
        <!-- 自定义下拉选择框 -->
        <div class="custom-select">
            <input
                type="text"
                class="search-input"
                bind:value={searchQuery}
                onfocus={() => (isDropdownOpen = true)}
                placeholder="搜索股票代码或名称..."
            />
            {#if isDropdownOpen}
                <div class="dropdown-list">
                    {#each filteredSymbols as stock (stock.symbol)}
                        <div
                            class="dropdown-item"
                            onclick={() => handleSelect(stock)}
                            class:selected={selectedSymbol === stock.symbol}
                        >
                            <span class="symbol">{stock.symbol}</span>
                            <span class="name">{stock.name}</span>
                            <span class="exchange">{stock.exchange}</span>
                        </div>
                    {:else}
                        <div class="dropdown-empty">未找到匹配的股票</div>
                    {/each}
                </div>
            {/if}
        </div>

        <!-- 刷新按钮组 -->
        <div class="action-buttons">
            <button class="refresh" onclick={refreshStockSymbols}>
                🔄 刷新股票
            </button>
            <button class="refresh" onclick={refreshHistory}>
                🔄 刷新数据
            </button>
        </div>

        <!-- 日期选择 -->
        <input type="date" bind:value={startDate} />
        <span>至</span>
        <input type="date" bind:value={endDate} />
        <button onclick={fetchHistory}>查询</button>
    </div>

    <!-- 图表占位 -->
    <div class="chart-container">
        <div class="chart-placeholder">
            📊 历史趋势图表（{selectedSymbol}）
        </div>
    </div>

    <!-- 数据表格 -->
    <div class="data-table">
        <div class="table-header">
            <div>日期</div>
            <div>开盘价</div>
            <div>收盘价</div>
            <div>最高价</div>
            <div>最低价</div>
        </div>
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

    /* 修改.controls容器样式 */
    .controls {
        display: flex;
        gap: 1.5rem; /* 增大间距 */
        align-items: stretch; /* 垂直对齐 */
        flex-wrap: nowrap; /* 禁止换行 */
    }

    /* 自定义选择框增加弹性布局 */
    .custom-select {
        position: relative;
        min-width: 240px; /* 最小宽度减小 */
        max-width: 400px; /* 增加最大宽度限制 */
        flex: 1 1 auto; /* 改为自动伸缩 */
    }

    /* 增加过渡动画 */
    .dropdown-item {
        transition: transform 0.2s ease;
    }
    .dropdown-item:hover {
        transform: translateX(5px);
    }

    /* 调整输入框内部尺寸 */
    .search-input {
        width: 370px;
        padding: 0.6rem 1rem; /* 减小内边距 */
        font-size: 1rem; /* 适当减小字体 */
    }

    .dropdown-list {
        position: absolute;
        width: 100%;
        max-height: 400px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        z-index: 100;
    }

    .dropdown-item {
        padding: 1rem;
        cursor: pointer;
        display: grid;
        grid-template-columns: 100px 1fr 80px;
        gap: 1rem;
        align-items: center;
        transition: background 0.2s;
    }

    .dropdown-item:hover {
        background: rgba(255, 255, 255, 0.1);
    }

    .dropdown-item.selected {
        background: var(--active-color);
    }

    .symbol {
        font-weight: bold;
        color: #3b82f6;
    }

    .exchange {
        font-size: 0.9em;
        color: #94a3b8;
    }

    .dropdown-empty {
        padding: 1rem;
        text-align: center;
        color: #94a3b8;
    }

    button {
        padding: 0.5rem 2rem;
        background: var(--active-color);
        color: white;
        border: none;
        border-radius: 0.5rem;
        cursor: pointer;
    }

    /* 刷新按钮组样式调整 */
    .action-buttons {
        display: flex;
        gap: 0.75rem;
        flex-shrink: 0; /* 禁止缩小 */
        align-items: center;
    }

    button.refresh {
        background: #3b82f6;
        padding: 0.5rem 1rem;
    }

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

    .action-buttons {
        display: flex;
        gap: 0.5rem;
    }

    button.refresh {
        background: #3b82f6;
        padding: 0.5rem 1rem;
    }

    button.refresh:hover {
        background: #2563eb;
    }

    @media (max-width: 768px) {
        .controls {
            flex-wrap: wrap; /* 小屏幕允许换行 */
            flex-direction: column;
        }

        .custom-select {
            min-width: 100%;
            max-width: 100%;
        }

        .search-input {
            font-size: 0.9rem;
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
