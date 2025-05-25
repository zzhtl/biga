<script lang="ts">
    import { onMount } from "svelte";
    import { invoke } from "@tauri-apps/api/core";
    import * as echarts from "echarts";
    import { formatVolume, formatDate, formatChange, formatChangePercent, formatPrice, formatPercent } from "../utils/utils";

    type EChartsOption = echarts.EChartsOption;

    // 新增图表实例声明
    let chart: echarts.ECharts | null = null;
    let chartContainer: HTMLDivElement;

    // 新增图表初始化
    $effect(() => {
        if (typeof window === "undefined") return;

        // 初始化图表
        chart = echarts.init(chartContainer);

        // 窗口调整时自适应
        const resizeHandler = () => chart?.resize();
        window.addEventListener("resize", resizeHandler);

        return () => {
            window.removeEventListener("resize", resizeHandler);
            chart?.dispose();
            chart = null;
        };
    });

    // 新增图表更新逻辑
    $effect(() => {
        if (!chart || historyData.length === 0) return;

        // 处理数据（按日期正序排列）
        const sortedData = [...historyData].reverse();

        // 准备图表数据
        const xData = sortedData.map((d) => formatDate(d.date));
        const kData = sortedData.map((d) => [d.open, d.close, d.low, d.high]);
        const volumes = sortedData.map((d) => d.volume);
        const changes = sortedData.map((d) => d.change_percent);

        // 图表配置
        const option: EChartsOption = {
            tooltip: {
                trigger: "axis",
                axisPointer: { type: "cross" },
            },
            grid: [
                { left: "5%", right: "5%", top: "5%", height: "70%" }, // K线图区域
                { left: "5%", right: "5%", top: "79%", height: "15%" }, // 成交量区域
            ],
            xAxis: [
                {
                    type: "category",
                    data: xData,
                    axisLabel: { rotate: 45 },
                    boundaryGap: false,
                },
                {
                    type: "category",
                    gridIndex: 1,
                    show: false,
                    data: xData,
                },
            ],
            yAxis: [
                {
                    scale: true,
                    splitArea: { show: true },
                },
                {
                    scale: true,
                    gridIndex: 1,
                    splitNumber: 2,
                    axisLabel: { show: false },
                    axisLine: { show: false },
                    splitLine: { show: false },
                },
            ],
            dataZoom: [
                {
                    type: "inside",
                    xAxisIndex: [0, 1],
                    start: 0,
                    end: 100,
                },
                {
                    type: "slider",
                    xAxisIndex: [0, 1],
                    show: true,
                    top: "95%",
                    height: 20,
                    start: 0,
                    end: 100,
                },
            ],
            series: [
                {
                    name: "K线",
                    type: "candlestick",
                    data: kData,
                    itemStyle: {
                        // 上涨颜色配置
                        color: "#FF0000", // 阳线填充色（默认红色）
                        borderColor: "#FF0000", // 阳线边框色
                        // 下跌颜色配置
                        color0: "#00FF00", // 阴线填充色（默认绿色）
                        borderColor0: "#00FF00", // 阴线边框色
                    },
                    emphasis: {
                        itemStyle: {
                            borderWidth: 2,
                        },
                    },
                },
                {
                    name: "成交量",
                    type: "bar",
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: volumes.map((v, i) => ({
                        value: v,
                        itemStyle: {
                            color:
                                sortedData[i].change > 0
                                    ? "#ef4444" // 上涨红色
                                    : "#10b981", // 下跌绿色
                        },
                    })),
                },
                {
                    name: "涨跌幅",
                    type: "line",
                    smooth: true,
                    data: changes,
                    symbol: "none",
                    lineStyle: { color: "#3b82f6" },
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: "rgba(59, 130, 246, 0.6)" },
                            { offset: 1, color: "rgba(59, 130, 246, 0.02)" },
                        ]),
                    },
                },
            ],
        };

        chart.setOption(option);
    });

    // 日期工具函数
    function getTodayISO() {
        const d = new Date();
        d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
        return d.toISOString().slice(0, 10);
    }

    function getFiveMonthAgoISO() {
        const d = new Date();
        d.setMonth(d.getMonth() - 5); // 自动处理跨年问题
        d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
        return d.toISOString().slice(0, 10);
    }

    // 初始化状态
    let startDate = $state(getFiveMonthAgoISO());
    let endDate = $state(getTodayISO());
    let selectedSymbol = $state(""); // 改为空字符串，等待加载股票列表后设置
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

    // 在原有类型定义后添加
    type HistoricalData = {
        date: string;
        open: number;
        close: number;
        high: number;
        low: number;
        volume: number;
        amount: number;
        amplitude: number;
        turnover_rate: number;
        change: number;
        change_percent: number;
    };

    // 添加历史数据状态
    let historyData = $state<HistoricalData[]>([]);
    let isLoading = $state(false);
    let errorMessage = $state("");

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
        // 选中股票查询历史数据
        fetchHistory();
    }

    // 修改后的 refreshStockSymbols 函数
    async function refreshStockSymbols() {
        console.log("开始刷新股票信息...");
        try {
            // 先刷新远程数据
            console.log("调用 refresh_stock_infos...");
            const refreshResult = await invoke("refresh_stock_infos");
            console.log("刷新结果:", refreshResult);

            // 然后获取本地数据
            console.log("调用 get_stock_infos...");
            const symbols = await invoke<StockInfo[]>("get_stock_infos");
            console.log("获取到股票数量:", symbols.length);

            stockSymbols = symbols;
        } catch (error) {
            console.error("刷新股票列表失败:", error);
            errorMessage = `刷新股票列表失败: ${error}`;
        }
    }

    // 同时需要修改 onMount 中的调用
    onMount(async () => {
        console.log("组件挂载，开始加载股票列表...");
        try {
            // 使用泛型指定返回类型
            const symbols = await invoke<StockInfo[]>("get_stock_infos");
            console.log("初始加载获取到股票数量:", symbols.length);
            stockSymbols = symbols;

            if (stockSymbols.length === 0) {
                console.log("股票列表为空，尝试刷新...");
                await refreshStockSymbols();
            }
        } catch (error) {
            console.error("初始加载股票列表失败:", error);
            errorMessage = `加载股票列表失败: ${error}`;
            // 如果获取失败，尝试刷新
            await refreshStockSymbols();
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
        if (!selectedSymbol) {
            errorMessage = "请先选择股票";
            return;
        }
        
        console.log("开始刷新历史数据，股票代码:", selectedSymbol);
        try {
            const result = await invoke("refresh_historical_data", { symbol: selectedSymbol });
            console.log("刷新历史数据结果:", result);
            await fetchHistory();
        } catch (error) {
            console.error("刷新历史数据失败:", error);
            errorMessage = `刷新历史数据失败: ${error}`;
        }
    }

    async function fetchHistory() {
        if (!selectedSymbol) {
            console.log("没有选中的股票，跳过获取历史数据");
            return;
        }
        
        if (!startDate || !endDate) {
            errorMessage = "请选择开始日期和结束日期";
            return;
        }

        console.log("开始获取历史数据:", {
            symbol: selectedSymbol,
            start: startDate,
            end: endDate
        });

        isLoading = true;
        errorMessage = "";

        try {
            const data = await invoke<HistoricalData[]>("get_historical_data", {
                symbol: selectedSymbol,
                start: startDate,
                end: endDate,
            });
            console.log("获取到历史数据条数:", data.length);
            historyData = data;
            
            if (data.length === 0) {
                errorMessage = "没有找到指定时间范围内的数据，请尝试刷新数据或调整时间范围";
            }
        } catch (error) {
            console.error("获取历史数据失败:", error);
            errorMessage = `获取数据失败: ${error}`;
            historyData = [];
        } finally {
            isLoading = false;
        }
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
        <div
            bind:this={chartContainer}
            style="width: 100%; height: 500px;"
        ></div>
    </div>

    <!-- 数据表格 -->
    <div class="data-table">
        <div class="table-header">
            <div>日期</div>
            <div>开盘价</div>
            <div>收盘价</div>
            <div>最高价</div>
            <div>最低价</div>
            <div>成交量</div>
            <div>成交额</div>
            <div>振幅</div>
            <div>涨跌额</div>
            <div>涨跌幅</div>
        </div>

        {#if isLoading}
            <div class="loading-indicator">⏳ 数据加载中...</div>
        {:else if errorMessage}
            <div class="error-message">
                ❌ {errorMessage}
            </div>
        {:else if historyData.length === 0}
            <div class="no-data">📭 暂无数据</div>
        {:else}
            {#each historyData as data (data.date)}
                <div class="table-row">
                    <div>{formatDate(data.date)}</div>
                    <div>{formatPrice(data.open)}</div>
                    <div>{formatPrice(data.close)}</div>
                    <div>{formatPrice(data.high)}</div>
                    <div>{formatPrice(data.low)}</div>
                    <div>{formatVolume(data.volume)}手</div>
                    <div>{formatVolume(data.amount)}</div>
                    <div>{formatPercent(data.amplitude)}</div>
                    <!-- 涨跌额 -->
                    <div
                        class:up={data.change > 0}
                        class:down={data.change < 0}
                    >
                        {data.change === null ? "-" : formatChange(data.change)}
                    </div>
                    <!-- 涨跌幅 -->
                    <div
                        class:up={data.change_percent > 0}
                        class:down={data.change_percent < 0}
                    >
                        {data.change_percent === null
                            ? "-"
                            : formatChangePercent(data.change_percent)}
                    </div>
                </div>
            {/each}
        {/if}
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
        gap: 1rem; /* 适中间距 */
        align-items: center; /* 垂直居中对齐 */
        flex-wrap: wrap; /* 允许换行以适应不同屏幕 */
        margin-bottom: 2rem;
    }

    /* 自定义选择框增加弹性布局 */
    .custom-select {
        position: relative;
        min-width: 200px; /* 最小宽度减小 */
        max-width: 320px; /* 减少最大宽度限制 */
        flex: 0 0 auto; /* 固定尺寸，不伸缩 */
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
        width: 280px;
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
        grid-template-columns: 100px 1fr;
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

    .name {
        color: #e2e8f0;
        font-size: 0.95em;
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

    .data-table {
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .table-header,
    .table-row {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 1rem;
        padding: 1rem;
    }

    /* 修正颜色定义 */
    .up {
        color: #ef4444; /* 红色表示上涨 */
    }
    .down {
        color: #10b981; /* 绿色表示下跌 */
    }
    /* 添加箭头指示 */
    .up::before {
        content: "↑";
        margin-right: 4px;
        font-size: 0.9em;
    }
    .down::before {
        content: "↓";
        margin-right: 4px;
        font-size: 0.9em;
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

    .loading-indicator,
    .error-message,
    .no-data {
        padding: 2rem;
        text-align: center;
        color: #94a3b8;
    }

    .error-message {
        color: #ef4444;
    }

    .table-row div {
        padding: 0.15rem;
        display: flex;
        align-items: center;
    }

    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
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
            grid-template-columns: repeat(4, 1fr);
        }

        .chart-container {
            margin: 1rem 0;
            padding: 0.5rem;
        }
    }
</style>
