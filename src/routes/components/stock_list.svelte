<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";

    // 定义类型
    interface RealtimeData {
        symbol: string;
        name: string;
        area: string;
        industry: string;
        market: string;
        ts_code: string;
        list_date: string;
        act_name: string;
        act_ent_type: string;
    }

    // 状态管理
    let stocks: RealtimeData[] = [];
    let loading = true;
    let error: string | null = null;
    let searchQuery = "";
    let searchDebounce: number | null = null;

    // 数据获取函数
    async function fetchData(query?: string) {
        try {
            loading = true;
            error = null;
            const response = await invoke<RealtimeData[]>("get_stock_list", {
                search: query,
            });
            stocks = response; // 保持引用（如果允许修改原始数据）
        } catch (err) {
            console.error("获取数据失败:", err);
            error = "无法获取股票数据，请稍后重试";
        } finally {
            loading = false;
        }
    }

    // 初始化加载（传递默认排序参数）
    onMount(async () => {
        await fetchData("");
    });

    // 监听搜索输入变化
    $: if (searchQuery) {
        if (searchDebounce) clearTimeout(searchDebounce);
        searchDebounce = setTimeout(() => {
            // 传递当前参数
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
            placeholder="搜索股票代码或名称或行业"
            class="search-input"
        />
    </div>

    <!-- 数据展示 -->
    <div class="data-table">
        <div class="table-header">
            <div>股票代码</div>
            <div>股票简称</div>
            <div>地域</div>
            <div>所属行业</div>
            <div>市场类型</div>
            <div>交易所</div>
            <div>上市日期</div>
            <div>实控人名称</div>
            <div>企业性质</div>
        </div>

        {#if loading}
            <div class="loading">加载中...</div>
        {:else if error}
            <div class="error">{error}</div>
        {:else}
            {#each stocks as stock}
                <div class="table-row">
                    <div class="symbol">{stock.symbol}</div>
                    <div class="name">{stock.name}</div>
                    <div class="region">{stock.area}</div>
                    <div class="industry">{stock.industry}</div>
                    <div class="market">{stock.market}</div>
                    <div class="ts_code">{stock.ts_code}</div>
                    <div class="list_date">{stock.list_date}</div>
                    <div class="act_name">{stock.act_name}</div>
                    <div class="act_ent_type">{stock.act_ent_type}</div>
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
        grid-template-columns: repeat(9, 1fr);
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
