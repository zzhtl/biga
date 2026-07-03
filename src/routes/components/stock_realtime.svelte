<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";
    import { formatVolume } from "../utils/utils";

    // 定义类型
    interface RealtimeData {
        symbol: string;
        name: string;
        date: Date;
        close: number;
        volume: number;
        amount: number;
        amplitude: number;
        turnover_rate: number;
        change: number;
        change_percent: number;
    }

    // 跨页导航：点击行跳转历史数据页
    type NavTarget = {
        view: "favorites" | "stock" | "list" | "realtime" | "historical" | "settings";
        symbol?: string;
        name?: string;
        action?: "history" | "predict";
    };
    export let onNavigate: (target: NavTarget) => void = () => {};

    // 状态管理
    let stocks: RealtimeData[] = [];
    let loading = true;
    let error: string | null = null;
    let searchQuery = "";
    let searchDebounce: number | null = null;
    // 排序状态
    let sortColumn = "change_percent"; // 默认按股票代码排序
    let sortDirection = "desc"; // 默认升序

    // 收藏星标（与收藏池联动）
    let watchSet: Set<string> = new Set();

    // 与后端 canonical_symbol 同口径：提取到恰好 6 位数字则用之，否则原样 trim
    function sixDigit(value: string): string {
        const digits = value.replace(/\D/g, "");
        return digits.length === 6 ? digits : value.trim();
    }

    async function loadWatchSet() {
        try {
            const symbols = await invoke<string[]>("get_watchlist_symbols");
            watchSet = new Set(symbols);
        } catch (e) {
            console.warn("获取收藏列表失败:", e);
        }
    }

    async function toggleWatch(stock: RealtimeData) {
        const key = sixDigit(stock.symbol);
        try {
            await invoke(
                watchSet.has(key) ? "remove_from_watchlist" : "add_to_watchlist",
                { symbol: stock.symbol },
            );
            await loadWatchSet();
        } catch (e) {
            error = `收藏操作失败: ${e}`;
        }
    }

    function goHistory(stock: RealtimeData) {
        onNavigate({
            view: "historical",
            symbol: stock.symbol,
            name: stock.name,
            action: "history",
        });
    }

    // 日期格式化
    const formatDate = (date: Date) => {
        return new Intl.DateTimeFormat("zh-CN").format(date);
    };

    // 数据获取函数（添加排序参数）
    async function fetchData(query?: string, sortBy?: string, order?: string) {
        try {
            loading = true;
            error = null;
            const response = await invoke<RealtimeData[]>("get_realtime_data", {
                search: query,
                column: sortBy,
                sort: order,
            });
            stocks = response; // 保持引用（如果允许修改原始数据）
            for (const item of stocks) {
                item.date = new Date(item.date);
            }
        } catch (err) {
            console.error("获取数据失败:", err);
            error = "无法获取实时数据，请稍后重试";
        } finally {
            loading = false;
        }
    }

    // 初始化加载（传递默认排序参数）
    onMount(async () => {
        await fetchData("", sortColumn, sortDirection);
        loadWatchSet();
    });

    // 监听搜索输入变化
    $: if (searchQuery) {
        if (searchDebounce) clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => {
            // 传递当前排序参数
            fetchData(searchQuery, sortColumn, sortDirection);
        }, 300);
    } else {
        fetchData("", sortColumn, sortDirection);
    }

    // 排序函数（触发数据重新获取）
    function sortData(column: string) {
        if (column === sortColumn) {
            sortDirection = sortDirection === "asc" ? "desc" : "asc";
        } else {
            sortColumn = column;
            sortDirection = "desc";
        }

        // 触发数据重新获取（传递当前排序参数和搜索词）
        fetchData(searchQuery, sortColumn, sortDirection);
    }
</script>

<!-- 页面布局 -->
<div class="container">
    <!-- 搜索框 -->
    <div class="search-container">
        <input
            bind:value={searchQuery}
            placeholder="搜索股票代码或名称"
            class="search-input"
        />
    </div>

    <!-- 数据展示 -->
    <div class="data-table">
        <div class="table-header">
            <div title="收藏"></div>
            <div onclick={() => sortData("symbol")} class="sort-column">
                股票代码
                {#if sortColumn === "symbol"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
            <div onclick={() => sortData("name")} class="sort-column">
                名称
                {#if sortColumn === "name"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
            <div>日期</div>
            <div>最新价</div>
            <div onclick={() => sortData("volume")} class="sort-column">
                成交量
                {#if sortColumn === "volume"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
            <div onclick={() => sortData("amount")} class="sort-column">
                成交额
                {#if sortColumn === "amount"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
            <div>振幅</div>
            <div onclick={() => sortData("change")} class="sort-column">
                涨跌额
                {#if sortColumn === "change"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
            <div onclick={() => sortData("change_percent")} class="sort-column">
                涨跌幅
                {#if sortColumn === "change_percent"}
                    {sortDirection === "asc" ? "↑" : "↓"}
                {/if}
            </div>
        </div>

        {#if loading}
            <div class="loading">加载中...</div>
        {:else if error}
            <div class="error">{error}</div>
        {:else}
            {#each stocks as stock}
                <div
                    class="table-row row-click"
                    role="button"
                    tabindex="0"
                    title="点击查看历史K线"
                    onclick={() => goHistory(stock)}
                    onkeydown={(e) => {
                        if (e.key === "Enter") goHistory(stock);
                    }}
                >
                    <div>
                        <button
                            class="star"
                            class:on={watchSet.has(sixDigit(stock.symbol))}
                            title={watchSet.has(sixDigit(stock.symbol))
                                ? "移出收藏池"
                                : "加入收藏池"}
                            onclick={(e) => {
                                e.stopPropagation();
                                toggleWatch(stock);
                            }}
                        >
                            {watchSet.has(sixDigit(stock.symbol)) ? "★" : "☆"}
                        </button>
                    </div>
                    <div class="symbol">{stock.symbol}</div>
                    <div class="name">{stock.name}</div>
                    <div class="date">{formatDate(stock.date)}</div>
                    <div class="close">{stock.close}</div>
                    <div class="volume">{formatVolume(stock.volume)}手</div>
                    <div class="amount">{formatVolume(stock.amount)}</div>
                    <div class="amplitude">{stock.amplitude}%</div>
                    <div
                        class:negative={stock.change > 0}
                        class:positive={stock.change < 0}
                    >
                        {stock.change > 0 ? "+" : ""}{stock.change.toFixed(2)}
                    </div>
                    <div
                        class:negative={stock.change_percent > 0}
                        class:positive={stock.change_percent < 0}
                    >
                        {stock.change_percent > 0
                            ? "+"
                            : ""}{stock.change_percent}%
                    </div>
                </div>
            {/each}
        {/if}
    </div>
</div>

<style>
    /* 主容器 */
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }

    /* 搜索框样式 */
    .search-container {
        margin: 1rem 0;
        text-align: center;
    }

    .search-input {
        padding: 0.8rem 1.5rem;
        width: 300px;
        border: 2px solid #3b82f6;
        border-radius: 24px;
        background: #2d2d30;
        color: #ffffff;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }

    .search-input:focus {
        outline: none;
        border-color: #0ea5e9;
        box-shadow: 0 0 0 2px rgba(14, 182, 129, 0.2);
    }

    .search-input::placeholder {
        color: #666666;
    }

    @media (max-width: 768px) {
        .search-input {
            width: 100%;
            margin: 0 1rem;
        }
    }

    /* 数据表格样式 */
    .data-table {
        margin-top: 2rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .table-header,
    .table-row {
        display: grid;
        grid-template-columns: 2.2rem repeat(9, 1fr);
        gap: 1rem;
        padding: 1rem;
    }

    .row-click {
        cursor: pointer;
    }

    .row-click:hover {
        background: rgba(99, 102, 241, 0.1);
    }

    .star {
        padding: 0 0.2rem;
        background: none;
        border: none;
        color: #64748b;
        font-size: 1rem;
        line-height: 1;
        cursor: pointer;
    }

    .star:hover {
        color: #cbd5e1;
    }

    .star.on {
        color: #fbbf24;
    }

    .table-header {
        background: rgba(255, 255, 255, 0.1);
        font-weight: 600;
    }

    .table-row {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .table-row div {
        padding: 0.15rem;
        display: flex;
        align-items: center;
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    /* 排序样式 */
    .sort-column {
        cursor: pointer;
    }

    .sort-column:active {
        opacity: 0.8;
    }

    /* 移动端适配 */
    @media (max-width: 768px) {
        .header-row,
        .data-row {
            grid-template-columns: repeat(2, 1fr);
        }

        .header-row div:nth-child(n + 3),
        .data-row div:nth-child(n + 3) {
            display: none;
        }
    }

    /* 加载/错误状态样式 */
    .loading,
    .error {
        padding: 2rem;
        text-align: center;
        color: #ffffff;
    }

    .error {
        color: #ef4444;
    }
</style>
