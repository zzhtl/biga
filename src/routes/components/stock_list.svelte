<script lang="ts">
    import { onMount } from "svelte";
    import {
        ChevronLeft,
        ChevronRight,
        History,
        RefreshCw,
        Search,
        Star,
    } from "lucide-svelte";
    import {
        errorMessage,
        getStockList,
        getWatchlistSymbols,
        setWatchlistMembership,
    } from "../services";
    import type { NavTarget, PagedResponse, Stock } from "../types";

    const PAGE_SIZE = 50;

    let { onNavigate = () => {} }: { onNavigate?: (target: NavTarget) => void } =
        $props();
    let result = $state<PagedResponse<Stock>>({
        data: [],
        total: 0,
        page: 1,
        page_size: PAGE_SIZE,
    });
    let watchSet = $state(new Set<string>());
    let searchQuery = $state("");
    let loading = $state(true);
    let error = $state("");
    let searchTimer: ReturnType<typeof setTimeout> | undefined;
    let requestSequence = 0;

    const totalPages = $derived(
        Math.max(1, Math.ceil(result.total / result.page_size)),
    );

    function sixDigit(value: string): string {
        const digits = value.replace(/\D/g, "");
        return digits.length === 6 ? digits : value.trim();
    }

    async function loadStocks(page = result.page): Promise<void> {
        const sequence = ++requestSequence;
        loading = true;
        error = "";
        try {
            const next = await getStockList(searchQuery.trim(), page, PAGE_SIZE);
            if (sequence === requestSequence) result = next;
        } catch (reason) {
            if (sequence === requestSequence) {
                error = errorMessage(reason, "股票列表加载失败");
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
        searchTimer = setTimeout(() => loadStocks(1), 300);
    }

    async function changePage(page: number): Promise<void> {
        if (page < 1 || page > totalPages || page === result.page) return;
        await loadStocks(page);
    }

    async function toggleWatch(stock: Stock): Promise<void> {
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

    function goHistory(stock: Stock): void {
        onNavigate({
            view: "historical",
            symbol: stock.symbol,
            name: stock.name,
            action: "history",
        });
    }

    onMount(() => {
        void loadStocks(1);
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
            <h1>股票列表</h1>
            <p>检索沪深股票，收藏后进入风险扫描与横向对比。</p>
        </div>
        <span class="result-count">共 {result.total.toLocaleString("zh-CN")} 只</span>
    </header>

    <div class="toolbar-row">
        <label class="search-field">
            <Search size={17} aria-hidden="true" />
            <input
                bind:value={searchQuery}
                oninput={handleSearch}
                placeholder="搜索代码、名称、行业或板块"
                aria-label="搜索股票"
            />
        </label>
        <button class="icon-button" onclick={() => loadStocks()} title="刷新列表" aria-label="刷新列表">
            <RefreshCw size={17} class={loading ? "spin" : undefined} aria-hidden="true" />
        </button>
    </div>

    {#if error}
        <div class="status-panel error" role="alert">
            <span>{error}</span>
            <button onclick={() => loadStocks()}>重试</button>
        </div>
    {/if}

    <div class="table-frame" aria-busy={loading}>
        <table>
            <thead>
                <tr>
                    <th class="watch-column"><span class="sr-only">收藏</span></th>
                    <th>股票代码</th>
                    <th>名称</th>
                    <th>行业</th>
                    <th>板块</th>
                    <th>市场</th>
                    <th class="actions-column">操作</th>
                </tr>
            </thead>
            <tbody>
                {#if loading && result.data.length === 0}
                    <tr><td colspan="7" class="table-state">正在加载股票列表...</td></tr>
                {:else if result.data.length === 0}
                    <tr><td colspan="7" class="table-state">没有符合条件的股票</td></tr>
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
                            <td>{stock.industry || "未标注"}</td>
                            <td>{stock.category || "未分类"}</td>
                            <td>{stock.market || "-"}</td>
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
        {#if loading && result.data.length > 0}
            <div class="table-loading">正在更新...</div>
        {/if}
    </div>

    <nav class="pagination" aria-label="股票列表分页">
        <button class="icon-button" onclick={() => changePage(result.page - 1)} disabled={result.page <= 1} aria-label="上一页">
            <ChevronLeft size={18} aria-hidden="true" />
        </button>
        <span>第 {result.page} / {totalPages} 页</span>
        <button class="icon-button" onclick={() => changePage(result.page + 1)} disabled={result.page >= totalPages} aria-label="下一页">
            <ChevronRight size={18} aria-hidden="true" />
        </button>
    </nav>
</div>

<style>
    .page-shell { max-width: 1240px; margin: 0 auto; }
    .result-count { color: var(--text-secondary); font-size: 0.86rem; white-space: nowrap; }
    .toolbar-row { display: flex; gap: 0.65rem; margin-bottom: 1rem; }
    .search-field { width: min(440px, 100%); }
    .table-frame { position: relative; }
    .watch-column { width: 48px; text-align: center; }
    .actions-column { width: 126px; text-align: right; }
    .symbol { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--accent); }
    .stock-name { color: var(--text-primary); font-weight: 600; }
    .star-button { display: inline-grid; place-items: center; width: 32px; height: 32px; border: 0; background: transparent; color: var(--text-muted); cursor: pointer; }
    .star-button:hover, .star-button.active { color: var(--warning); }
    .row-action { display: inline-flex; align-items: center; gap: 0.35rem; border: 0; background: transparent; color: var(--accent); cursor: pointer; font: inherit; font-size: 0.82rem; }
    .table-loading { position: absolute; inset: 0; display: grid; place-items: center; background: rgba(17, 19, 21, 0.58); color: var(--text-secondary); }
    .pagination { display: flex; align-items: center; justify-content: flex-end; gap: 0.75rem; margin-top: 1rem; color: var(--text-secondary); font-size: 0.86rem; }
    @media (max-width: 1100px) {
        table { min-width: 850px; }
    }
</style>
