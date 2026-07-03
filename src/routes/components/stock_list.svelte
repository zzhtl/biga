<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";

    // 定义类型
    interface Stock {
        symbol: string;
        name: string;
        area: string;
        industry: string;
        market: string;
        ts_code: string;
        list_date: string;
        act_name: string;
        act_ent_type: string;
        category: string;
    }

    // 跨页导航：点击股票卡片跳转历史数据页
    type NavTarget = {
        view: "favorites" | "stock" | "list" | "realtime" | "historical" | "settings";
        symbol?: string;
        name?: string;
        action?: "history" | "predict";
    };
    export let onNavigate: (target: NavTarget) => void = () => {};

    // 状态管理
    let stocks: Stock[] = [];
    let loading = true;
    let error: string | null = null;
    let searchQuery = "";
    let searchDebounce: number | null = null;
    let collapsed: Record<string, boolean> = {};

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

    async function toggleWatch(stock: Stock) {
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

    function goHistory(stock: Stock) {
        onNavigate({
            view: "historical",
            symbol: stock.symbol,
            name: stock.name,
            action: "history",
        });
    }

    // 按板块分组（保持后端 ORDER BY category, symbol 的顺序）
    $: groups = (() => {
        const map = new Map<string, Stock[]>();
        for (const s of stocks) {
            const key = s.category && s.category.trim() ? s.category : "未分类";
            if (!map.has(key)) map.set(key, []);
            map.get(key)!.push(s);
        }
        return Array.from(map.entries());
    })();

    // 数据获取函数
    async function fetchData(query?: string) {
        try {
            loading = true;
            error = null;
            const response = await invoke<Stock[]>("get_stock_list", {
                search: query,
            });
            stocks = response;
        } catch (err) {
            console.error("获取数据失败:", err);
            error = "无法获取股票数据，请稍后重试";
        } finally {
            loading = false;
        }
    }

    function toggle(category: string) {
        collapsed = { ...collapsed, [category]: !collapsed[category] };
    }

    // 初始化加载
    onMount(async () => {
        await fetchData("");
        loadWatchSet();
    });

    // 监听搜索输入变化
    $: if (searchQuery) {
        if (searchDebounce) clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => {
            fetchData(searchQuery);
        }, 300);
    } else {
        fetchData("");
    }
</script>

<!-- 页面布局 -->
<div class="container">
    <!-- 搜索框 -->
    <div class="search-container">
        <input
            bind:value={searchQuery}
            placeholder="搜索股票代码 / 名称 / 行业 / 板块"
            class="search-input"
        />
    </div>

    <!-- 数据展示 -->
    {#if loading}
        <div class="loading">加载中...</div>
    {:else if error}
        <div class="error">{error}</div>
    {:else if stocks.length === 0}
        <div class="loading">暂无数据</div>
    {:else}
        {#each groups as [category, items] (category)}
            <div class="sector">
                <button class="sector-header" on:click={() => toggle(category)}>
                    <span class="caret">{collapsed[category] ? "▶" : "▼"}</span>
                    <span class="sector-name">{category}</span>
                    <span class="sector-count">{items.length} 只</span>
                </button>
                {#if !collapsed[category]}
                    <div class="grid">
                        {#each items as stock (stock.symbol)}
                            <div
                                class="card"
                                role="button"
                                tabindex="0"
                                title="点击查看历史K线"
                                on:click={() => goHistory(stock)}
                                on:keydown={(e) => {
                                    if (e.key === "Enter") goHistory(stock);
                                }}
                            >
                                <span class="symbol">{stock.symbol}</span>
                                <span class="name">{stock.name}</span>
                                <button
                                    class="star"
                                    class:on={watchSet.has(sixDigit(stock.symbol))}
                                    title={watchSet.has(sixDigit(stock.symbol))
                                        ? "移出收藏池"
                                        : "加入收藏池"}
                                    on:click|stopPropagation={() => toggleWatch(stock)}
                                >
                                    {watchSet.has(sixDigit(stock.symbol)) ? "★" : "☆"}
                                </button>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
    {/if}
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
        width: 320px;
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

    /* 板块分组 */
    .sector {
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .sector-header {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.9rem 1.2rem;
        background: rgba(99, 102, 241, 0.18);
        border: none;
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 600;
        cursor: pointer;
        text-align: left;
    }

    .sector-header:hover {
        background: rgba(99, 102, 241, 0.28);
    }

    .caret {
        font-size: 0.8rem;
        color: #a5b4fc;
    }

    .sector-name {
        flex: 0 0 auto;
    }

    .sector-count {
        margin-left: auto;
        font-size: 0.85rem;
        font-weight: 400;
        color: #94a3b8;
    }

    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 0.5rem;
        padding: 0.9rem 1.2rem 1.2rem;
    }

    .card {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        padding: 0.55rem 0.75rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        cursor: pointer;
        transition: background 0.15s ease;
    }

    .card:hover {
        background: rgba(99, 102, 241, 0.15);
    }

    .card .star {
        margin-left: auto;
        padding: 0 0.2rem;
        background: none;
        border: none;
        color: #64748b;
        font-size: 1rem;
        line-height: 1;
        cursor: pointer;
    }

    .card .star:hover {
        color: #cbd5e1;
    }

    .card .star.on {
        color: #fbbf24;
    }

    .card .symbol {
        font-family: monospace;
        color: #38bdf8;
        font-size: 0.9rem;
    }

    .card .name {
        color: #e2e8f0;
        font-size: 0.95rem;
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

    @media (max-width: 768px) {
        .search-input {
            width: 100%;
            margin: 0 1rem;
        }
    }
</style>
