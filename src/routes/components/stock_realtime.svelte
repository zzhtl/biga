<script lang="ts">
    import { onMount } from "svelte";
    import {
        ArrowDown,
        ArrowUp,
        ArrowUpDown,
        ChevronLeft,
        ChevronRight,
        History,
        RefreshCw,
        Search,
        Star,
    } from "lucide-svelte";
    import {
        errorMessage,
        getRealtimeData,
        getWatchlistSymbols,
        setWatchlistMembership,
    } from "../services";
    import type {
        NavTarget,
        PagedResponse,
        RealtimeData,
        RealtimeSortColumn,
        SortDirection,
    } from "../types";
    import { formatDate, formatVolume } from "../utils/utils";

    const PAGE_SIZE = 50;

    let { onNavigate = () => {} }: { onNavigate?: (target: NavTarget) => void } =
        $props();
    let result = $state<PagedResponse<RealtimeData>>({
        data: [],
        total: 0,
        page: 1,
        page_size: PAGE_SIZE,
    });
    let watchSet = $state(new Set<string>());
    let searchQuery = $state("");
    let sortColumn = $state<RealtimeSortColumn>("change_percent");
    let sortDirection = $state<SortDirection>("desc");
    let loading = $state(true);
    let error = $state("");
    let searchTimer: ReturnType<typeof setTimeout> | undefined;
    let requestSequence = 0;

    const totalPages = $derived(
        Math.max(1, Math.ceil(result.total / result.page_size)),
    );

    const sortableColumns: Array<{
        key: RealtimeSortColumn;
        label: string;
    }> = [
        { key: "symbol", label: "股票代码" },
        { key: "name", label: "名称" },
        { key: "volume", label: "成交量" },
        { key: "amount", label: "成交额" },
        { key: "change", label: "涨跌额" },
        { key: "change_percent", label: "涨跌幅" },
    ];

    function sixDigit(value: string): string {
        const digits = value.replace(/\D/g, "");
        return digits.length === 6 ? digits : value.trim();
    }

    async function loadQuotes(page = result.page): Promise<void> {
        const sequence = ++requestSequence;
        loading = true;
        error = "";
        try {
            const next = await getRealtimeData(
                searchQuery.trim(),
                sortColumn,
                sortDirection,
                page,
                PAGE_SIZE,
            );
            if (sequence === requestSequence) result = next;
        } catch (reason) {
            if (sequence === requestSequence) {
                error = errorMessage(reason, "实时行情加载失败");
            }
        } finally {
            if (sequence === requestSequence) loading = false;
        }
    }

    async function loadWatchlist(): Promise<void> {
        try {
            watchSet = new Set(await getWatchlistSymbols());
        } catch {
            watchSet = new Set();
        }
    }

    function handleSearch(): void {
        if (searchTimer) clearTimeout(searchTimer);
        searchTimer = setTimeout(() => loadQuotes(1), 300);
    }

    async function sortBy(column: RealtimeSortColumn): Promise<void> {
        if (column === sortColumn) {
            sortDirection = sortDirection === "asc" ? "desc" : "asc";
        } else {
            sortColumn = column;
            sortDirection = "desc";
        }
        await loadQuotes(1);
    }

    async function changePage(page: number): Promise<void> {
        if (page < 1 || page > totalPages || page === result.page) return;
        await loadQuotes(page);
    }

    async function toggleWatch(stock: RealtimeData): Promise<void> {
        const symbol = sixDigit(stock.symbol);
        const watched = watchSet.has(symbol);
        try {
            await setWatchlistMembership(stock.symbol, watched);
            const next = new Set(watchSet);
            watched ? next.delete(symbol) : next.add(symbol);
            watchSet = next;
        } catch (reason) {
            error = errorMessage(reason, "收藏操作失败");
        }
    }

    function goHistory(stock: RealtimeData): void {
        onNavigate({
            view: "historical",
            symbol: stock.symbol,
            name: stock.name,
            action: "history",
        });
    }

    onMount(() => {
        void loadQuotes(1);
        void loadWatchlist();
        return () => {
            requestSequence += 1;
            if (searchTimer) clearTimeout(searchTimer);
        };
    });
</script>

<div class="page-shell">
    <header class="page-header">
        <div>
            <h1>实时行情</h1>
            <p>浏览本地最新行情快照，按关键指标排序并快速进入历史分析。</p>
        </div>
        <span class="result-count">{result.total.toLocaleString("zh-CN")} 条记录</span>
    </header>

    <div class="toolbar-row">
        <label class="search-field">
            <Search size={17} aria-hidden="true" />
            <input bind:value={searchQuery} oninput={handleSearch} placeholder="搜索股票代码或名称" aria-label="搜索实时行情" />
        </label>
        <button class="icon-button" onclick={() => loadQuotes()} title="刷新行情" aria-label="刷新行情">
            <RefreshCw size={17} class={loading ? "spin" : undefined} aria-hidden="true" />
        </button>
    </div>

    {#if error}
        <div class="status-panel error" role="alert">
            <span>{error}</span>
            <button onclick={() => loadQuotes()}>重试</button>
        </div>
    {/if}

    <div class="table-frame" aria-busy={loading}>
        <table>
            <thead>
                <tr>
                    <th class="watch-column"><span class="sr-only">收藏</span></th>
                    {#each sortableColumns.slice(0, 2) as column (column.key)}
                        <th>
                            <button class="sort-button" onclick={() => sortBy(column.key)}>
                                {column.label}
                                {#if column.key !== sortColumn}
                                    <ArrowUpDown size={14} aria-hidden="true" />
                                {:else if sortDirection === "asc"}
                                    <ArrowUp size={14} aria-hidden="true" />
                                {:else}
                                    <ArrowDown size={14} aria-hidden="true" />
                                {/if}
                            </button>
                        </th>
                    {/each}
                    <th>日期</th>
                    <th>最新价</th>
                    {#each sortableColumns.slice(2, 4) as column (column.key)}
                        <th>
                            <button class="sort-button" onclick={() => sortBy(column.key)}>
                                {column.label}
                                {#if column.key !== sortColumn}
                                    <ArrowUpDown size={14} aria-hidden="true" />
                                {:else if sortDirection === "asc"}
                                    <ArrowUp size={14} aria-hidden="true" />
                                {:else}
                                    <ArrowDown size={14} aria-hidden="true" />
                                {/if}
                            </button>
                        </th>
                    {/each}
                    <th>振幅</th>
                    {#each sortableColumns.slice(4) as column (column.key)}
                        <th>
                            <button class="sort-button" onclick={() => sortBy(column.key)}>
                                {column.label}
                                {#if column.key !== sortColumn}
                                    <ArrowUpDown size={14} aria-hidden="true" />
                                {:else if sortDirection === "asc"}
                                    <ArrowUp size={14} aria-hidden="true" />
                                {:else}
                                    <ArrowDown size={14} aria-hidden="true" />
                                {/if}
                            </button>
                        </th>
                    {/each}
                    <th class="actions-column">操作</th>
                </tr>
            </thead>
            <tbody>
                {#if loading && result.data.length === 0}
                    <tr><td colspan="11" class="table-state">正在加载实时行情...</td></tr>
                {:else if result.data.length === 0}
                    <tr><td colspan="11" class="table-state">暂无行情数据</td></tr>
                {:else}
                    {#each result.data as stock (stock.symbol)}
                        <tr>
                            <td class="watch-column">
                                <button
                                    class="star-button"
                                    class:active={watchSet.has(sixDigit(stock.symbol))}
                                    onclick={() => toggleWatch(stock)}
                                    title={watchSet.has(sixDigit(stock.symbol)) ? "移出收藏" : "加入收藏"}
                                    aria-label={watchSet.has(sixDigit(stock.symbol)) ? `移出收藏 ${stock.name}` : `加入收藏 ${stock.name}`}
                                >
                                    <Star size={17} fill={watchSet.has(sixDigit(stock.symbol)) ? "currentColor" : "none"} aria-hidden="true" />
                                </button>
                            </td>
                            <td><span class="symbol">{stock.symbol}</span></td>
                            <td class="stock-name">{stock.name || "-"}</td>
                            <td>{formatDate(stock.date)}</td>
                            <td>{stock.close.toFixed(2)}</td>
                            <td>{formatVolume(stock.volume)}手</td>
                            <td>{formatVolume(stock.amount)}</td>
                            <td>{stock.amplitude.toFixed(2)}%</td>
                            <td class:price-up={stock.change > 0} class:price-down={stock.change < 0}>{stock.change > 0 ? "+" : ""}{stock.change.toFixed(2)}</td>
                            <td class:price-up={stock.change_percent > 0} class:price-down={stock.change_percent < 0}>{stock.change_percent > 0 ? "+" : ""}{stock.change_percent.toFixed(2)}%</td>
                            <td class="actions-column">
                                <button class="row-action" onclick={() => goHistory(stock)} title="查看历史数据">
                                    <History size={15} aria-hidden="true" />历史数据
                                </button>
                            </td>
                        </tr>
                    {/each}
                {/if}
            </tbody>
        </table>
        {#if loading && result.data.length > 0}<div class="table-loading">正在更新...</div>{/if}
    </div>

    <nav class="pagination" aria-label="实时行情分页">
        <button class="icon-button" onclick={() => changePage(result.page - 1)} disabled={result.page <= 1} aria-label="上一页"><ChevronLeft size={18} aria-hidden="true" /></button>
        <span>第 {result.page} / {totalPages} 页</span>
        <button class="icon-button" onclick={() => changePage(result.page + 1)} disabled={result.page >= totalPages} aria-label="下一页"><ChevronRight size={18} aria-hidden="true" /></button>
    </nav>
</div>

<style>
    .page-shell { max-width: 1320px; margin: 0 auto; }
    .result-count { color: var(--text-secondary); font-size: 0.86rem; white-space: nowrap; }
    .toolbar-row { display: flex; gap: 0.65rem; margin-bottom: 1rem; }
    .search-field { width: min(420px, 100%); }
    .table-frame { position: relative; }
    table { min-width: 1180px; }
    .watch-column { width: 46px; text-align: center; }
    .actions-column { width: 112px; text-align: right; }
    .sort-button { display: inline-flex; align-items: center; gap: 0.3rem; border: 0; padding: 0; background: transparent; color: inherit; font: inherit; font-weight: inherit; cursor: pointer; white-space: nowrap; }
    .sort-button:hover { color: var(--accent); }
    .symbol { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); }
    .stock-name { color: var(--text-primary); font-weight: 600; }
    .star-button { display: inline-grid; place-items: center; width: 32px; height: 32px; border: 0; background: transparent; color: var(--text-muted); cursor: pointer; }
    .star-button:hover, .star-button.active { color: var(--warning); }
    .row-action { display: inline-flex; align-items: center; gap: 0.3rem; border: 0; background: transparent; color: var(--accent); cursor: pointer; font: inherit; font-size: 0.8rem; }
    .price-up { color: var(--price-up); }
    .price-down { color: var(--price-down); }
    .table-loading { position: absolute; inset: 0; display: grid; place-items: center; background: rgba(17, 19, 21, 0.58); color: var(--text-secondary); }
    .pagination { display: flex; align-items: center; justify-content: flex-end; gap: 0.75rem; margin-top: 1rem; color: var(--text-secondary); font-size: 0.86rem; }
</style>
