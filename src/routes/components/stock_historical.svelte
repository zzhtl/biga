<script lang="ts">
    import { onMount } from "svelte";
    import { Database, LoaderCircle, RefreshCw, Search } from "lucide-svelte";
    import HistoricalChart from "./historical_chart.svelte";
    import {
        errorMessage,
        getHistoricalData,
        getStockInfos,
        refreshHistoricalData,
        refreshStockInfos,
    } from "../services";
    import type { HistoricalData, StockInfo } from "../types";
    import {
        formatChange,
        formatChangePercent,
        formatPrice,
        formatVolume,
    } from "../utils/utils";
    import { calculateHistoricalSummary } from "../utils/historical_summary";

    type Props = {
        navTarget?: { symbol: string; name?: string } | null;
        onNavConsumed?: () => void;
    };

    let { navTarget = null, onNavConsumed = () => {} }: Props = $props();
    let startDate = $state(monthsAgoIso(5));
    let endDate = $state(todayIso());
    let selectedSymbol = $state("");
    let searchQuery = $state("");
    let stockSymbols = $state<StockInfo[]>([]);
    let historyData = $state<HistoricalData[]>([]);
    let dropdownOpen = $state(false);
    let loading = $state(false);
    let refreshingHistory = $state(false);
    let refreshingStocks = $state(false);
    let error = $state("");
    let requestSequence = 0;
    let consumedTarget = "";

    const filteredSymbols = $derived.by(() => {
        const query = searchQuery.trim().toLowerCase();
        if (!query || selectedSymbol) return stockSymbols.slice(0, 80);
        return stockSymbols
            .filter((stock) =>
                stock.symbol.toLowerCase().includes(query) ||
                stock.name.toLowerCase().includes(query) ||
                stock.exchange.toLowerCase().includes(query),
            )
            .slice(0, 80);
    });

    const sortedHistory = $derived.by(() =>
        [...historyData].sort((left, right) => right.date.localeCompare(left.date)),
    );
    const latest = $derived(sortedHistory[0] ?? null);
    const historySummary = $derived(calculateHistoricalSummary(historyData));

    function todayIso(): string {
        const date = new Date();
        date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
        return date.toISOString().slice(0, 10);
    }

    function monthsAgoIso(months: number): string {
        const date = new Date();
        date.setMonth(date.getMonth() - months);
        date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
        return date.toISOString().slice(0, 10);
    }

    function displayStock(stock: StockInfo): string {
        return `${stock.symbol} - ${stock.name}`;
    }

    function handleInput(): void {
        selectedSymbol = "";
        dropdownOpen = true;
    }

    async function selectStock(stock: StockInfo): Promise<void> {
        selectedSymbol = stock.symbol;
        searchQuery = displayStock(stock);
        dropdownOpen = false;
        await queryHistory();
    }

    async function loadStockSymbols(): Promise<void> {
        try {
            stockSymbols = await getStockInfos();
        } catch (reason) {
            error = errorMessage(reason, "股票目录加载失败");
        }
    }

    async function refreshStockCatalog(): Promise<void> {
        refreshingStocks = true;
        error = "";
        try {
            await refreshStockInfos();
            await loadStockSymbols();
        } catch (reason) {
            error = errorMessage(reason, "股票目录刷新失败");
        } finally {
            refreshingStocks = false;
        }
    }

    async function queryHistory(symbol = selectedSymbol): Promise<void> {
        if (!symbol) {
            error = "请先选择股票";
            return;
        }
        if (!startDate || !endDate || startDate > endDate) {
            error = "请选择有效的开始和结束日期";
            return;
        }

        const sequence = ++requestSequence;
        loading = true;
        error = "";
        try {
            const data = await getHistoricalData(symbol, startDate, endDate);
            if (sequence === requestSequence) {
                historyData = data;
                if (data.length === 0) error = "当前日期范围内没有历史数据";
            }
        } catch (reason) {
            if (sequence === requestSequence) {
                historyData = [];
                error = errorMessage(reason, "历史数据查询失败");
            }
        } finally {
            if (sequence === requestSequence) loading = false;
        }
    }

    async function refreshSelectedHistory(): Promise<void> {
        if (!selectedSymbol) {
            error = "请先选择股票";
            return;
        }
        refreshingHistory = true;
        error = "";
        try {
            await refreshHistoricalData(selectedSymbol);
            await queryHistory();
        } catch (reason) {
            error = errorMessage(reason, "历史数据刷新失败");
        } finally {
            refreshingHistory = false;
        }
    }

    $effect(() => {
        if (!navTarget?.symbol || navTarget.symbol === consumedTarget) return;
        consumedTarget = navTarget.symbol;
        selectedSymbol = navTarget.symbol;
        searchQuery = navTarget.name
            ? `${navTarget.symbol} - ${navTarget.name}`
            : navTarget.symbol;
        dropdownOpen = false;
        void queryHistory(navTarget.symbol);
        onNavConsumed();
    });

    onMount(() => {
        void loadStockSymbols();
        const closeDropdown = (event: MouseEvent) => {
            if (!(event.target as HTMLElement).closest(".stock-picker")) {
                dropdownOpen = false;
            }
        };
        document.addEventListener("click", closeDropdown);
        return () => {
            requestSequence += 1;
            document.removeEventListener("click", closeDropdown);
        };
    });
</script>

<div class="history-page">
    <header class="page-header">
        <div>
            <h1>历史数据</h1>
            <p>查看 K 线、成交量、MACD 与 KDJ，并核对具体交易日数据。</p>
        </div>
        <button class="secondary-button" onclick={refreshStockCatalog} disabled={refreshingStocks}>
            {#if refreshingStocks}<LoaderCircle size={16} class="spin" aria-hidden="true" />刷新中{:else}<Database size={16} aria-hidden="true" />更新股票目录{/if}
        </button>
    </header>

    <section class="query-bar" aria-label="历史数据查询条件">
        <div class="stock-picker">
            <label for="stock-search">股票</label>
            <div class="search-field">
                <Search size={16} aria-hidden="true" />
                <input id="stock-search" bind:value={searchQuery} oninput={handleInput} onfocus={() => (dropdownOpen = true)} placeholder="搜索股票代码或名称" autocomplete="off" />
            </div>
            {#if dropdownOpen}
                <div class="dropdown" role="listbox" aria-label="股票搜索结果">
                    {#each filteredSymbols as stock (stock.symbol)}
                        <button type="button" role="option" aria-selected={selectedSymbol === stock.symbol} onclick={() => selectStock(stock)}>
                            <span>{stock.symbol}</span><strong>{stock.name}</strong><small>{stock.exchange}</small>
                        </button>
                    {:else}
                        <div class="dropdown-empty">没有匹配的股票</div>
                    {/each}
                </div>
            {/if}
        </div>
        <label class="date-field" for="history-start">开始日期<input id="history-start" type="date" bind:value={startDate} max={endDate} /></label>
        <label class="date-field" for="history-end">结束日期<input id="history-end" type="date" bind:value={endDate} min={startDate} max={todayIso()} /></label>
        <button class="primary-button" onclick={() => queryHistory()} disabled={loading || !selectedSymbol}>
            {#if loading}<LoaderCircle size={16} class="spin" aria-hidden="true" />查询中{:else}<Search size={16} aria-hidden="true" />查询{/if}
        </button>
        <button class="secondary-button" onclick={refreshSelectedHistory} disabled={refreshingHistory || !selectedSymbol}>
            {#if refreshingHistory}<LoaderCircle size={16} class="spin" aria-hidden="true" />刷新中{:else}<RefreshCw size={16} aria-hidden="true" />刷新数据{/if}
        </button>
    </section>

    {#if error}
        <div class="status-panel error" role="alert"><span>{error}</span>{#if selectedSymbol}<button onclick={() => queryHistory()}>重试查询</button>{/if}</div>
    {/if}

    {#if latest}
        <section class="market-summary" aria-label="最新交易日摘要">
            <div><span>最新交易日</span><strong>{latest.date.slice(0, 10)}</strong></div>
            <div><span>收盘价</span><strong>{formatPrice(latest.close)}</strong></div>
            <div><span>涨跌幅</span><strong class:price-up={latest.change_percent > 0} class:price-down={latest.change_percent < 0}>{formatChangePercent(latest.change_percent)}</strong></div>
            <div><span>成交量</span><strong>{formatVolume(latest.volume)}手</strong></div>
        </section>

        {#if historySummary}
            <section class="range-summary" aria-label="查询区间统计">
                <div><span>实际区间</span><strong>{historySummary.startDate.slice(0, 10)} 至 {historySummary.endDate.slice(0, 10)}</strong></div>
                <div><span>交易日</span><strong>{historySummary.tradingDays} 天</strong></div>
                <div><span>区间涨跌额</span><strong class:price-up={historySummary.rangeChange !== null && historySummary.rangeChange > 0} class:price-down={historySummary.rangeChange !== null && historySummary.rangeChange < 0}>{historySummary.rangeChange === null ? "--" : formatChange(historySummary.rangeChange)}</strong></div>
                <div><span>区间涨跌幅</span><strong class:price-up={historySummary.rangeChangePercent !== null && historySummary.rangeChangePercent > 0} class:price-down={historySummary.rangeChangePercent !== null && historySummary.rangeChangePercent < 0}>{historySummary.rangeChangePercent === null ? "--" : formatChangePercent(historySummary.rangeChangePercent)}</strong></div>
                <div><span>区间最高</span><strong>{formatPrice(historySummary.highestPrice)}</strong></div>
                <div><span>区间最低</span><strong>{formatPrice(historySummary.lowestPrice)}</strong></div>
                <div><span>平均日涨跌幅</span><strong class:price-up={historySummary.averageDailyChangePercent > 0} class:price-down={historySummary.averageDailyChangePercent < 0}>{formatChangePercent(historySummary.averageDailyChangePercent)}</strong></div>
                <div><span>上涨 / 下跌 / 平盘</span><strong class="day-counts"><span class="price-up">{historySummary.upDays}</span> / <span class="price-down">{historySummary.downDays}</span> / {historySummary.flatDays}</strong></div>
            </section>
        {/if}

        <section class="chart-section" aria-label="历史技术图表">
            <HistoricalChart data={historyData} />
        </section>

        <div class="table-frame history-table">
            <table>
                <thead><tr><th>日期</th><th>开盘</th><th>收盘</th><th>最高</th><th>最低</th><th>成交量</th><th>成交额</th><th>振幅</th><th>涨跌额</th><th>涨跌幅</th></tr></thead>
                <tbody>
                    {#each sortedHistory as item (item.date)}
                        <tr>
                            <td>{item.date.slice(0, 10)}</td>
                            <td>{formatPrice(item.open)}</td><td>{formatPrice(item.close)}</td><td>{formatPrice(item.high)}</td><td>{formatPrice(item.low)}</td>
                            <td>{formatVolume(item.volume)}手</td><td>{formatVolume(item.amount)}</td><td>{item.amplitude.toFixed(2)}%</td>
                            <td class:price-up={item.change > 0} class:price-down={item.change < 0}>{formatChange(item.change)}</td>
                            <td class:price-up={item.change_percent > 0} class:price-down={item.change_percent < 0}>{formatChangePercent(item.change_percent)}</td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {:else if !loading}
        <div class="empty-state">
            <Database size={24} aria-hidden="true" />
            <h2>选择股票后查看历史数据</h2>
            <p>图表只在查询到有效数据后显示。</p>
        </div>
    {/if}
</div>

<style>
    .history-page { max-width: 1320px; margin: 0 auto; }
    .query-bar { display: grid; grid-template-columns: minmax(260px, 1fr) 158px 158px auto auto; gap: 0.65rem; align-items: end; margin-bottom: 1rem; }
    .stock-picker { position: relative; min-width: 0; }
    .stock-picker > label, .date-field { display: grid; gap: 0.35rem; color: var(--text-muted); font-size: 0.74rem; }
    .dropdown { position: absolute; top: calc(100% + 5px); left: 0; right: 0; z-index: 20; max-height: 320px; overflow: auto; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--surface-2); box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3); }
    .dropdown button { display: grid; grid-template-columns: 92px 1fr auto; gap: 0.7rem; width: 100%; padding: 0.65rem 0.75rem; border: 0; border-bottom: 1px solid var(--border); background: transparent; color: var(--text-secondary); text-align: left; cursor: pointer; }
    .dropdown button:hover, .dropdown button[aria-selected="true"] { background: var(--accent-muted); }
    .dropdown button span { color: var(--accent); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .dropdown button strong { color: var(--text-primary); }
    .dropdown button small { color: var(--text-muted); }
    .dropdown-empty { padding: 1rem; color: var(--text-muted); text-align: center; }
    .market-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); }
    .market-summary div { display: grid; gap: 0.2rem; padding: 0.8rem 1rem; border-right: 1px solid var(--border); }
    .market-summary div:last-child { border-right: 0; }
    .market-summary span { color: var(--text-muted); font-size: 0.72rem; }
    .market-summary strong { font-size: 0.95rem; }
    .range-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 0.65rem; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); }
    .range-summary > div { display: grid; gap: 0.2rem; padding: 0.8rem 1rem; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    .range-summary > div:nth-child(4n) { border-right: 0; }
    .range-summary > div:nth-last-child(-n + 4) { border-bottom: 0; }
    .range-summary > div > span { color: var(--text-muted); font-size: 0.72rem; }
    .range-summary strong { font-size: 0.95rem; }
    .day-counts .price-up { color: var(--price-up); }
    .day-counts .price-down { color: var(--price-down); }
    .price-up { color: var(--price-up); }
    .price-down { color: var(--price-down); }
    .chart-section { margin: 0.9rem 0; border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); padding: 0.5rem; }
    .history-table { max-height: 480px; }
    .empty-state { display: grid; justify-items: center; gap: 0.45rem; min-height: 240px; padding: 3rem 1rem; border: 1px dashed var(--border-strong); border-radius: 8px; color: var(--text-muted); text-align: center; }
    .empty-state h2 { margin: 0.35rem 0 0; color: var(--text-primary); font-size: 1rem; }
    .empty-state p { margin: 0; font-size: 0.84rem; }
    @media (max-width: 1120px) {
        .query-bar { grid-template-columns: minmax(260px, 1fr) 150px 150px; }
        .query-bar > button { width: 100%; }
        .history-table table { min-width: 980px; }
    }
</style>
