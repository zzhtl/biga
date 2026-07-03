<script lang="ts">
    import StockPrediction from "./components/stock_prediction.svelte";
    import Settings from "./components/sys_settings.svelte";
    import RealTimeData from "./components/stock_realtime.svelte";
    import HistoricalData from "./components/stock_historical.svelte";
    import StockList from "./components/stock_list.svelte";
    import StockFavorites from "./components/stock_favorites.svelte";

    type View =
        | "favorites"
        | "stock"
        | "list"
        | "realtime"
        | "historical"
        | "settings";
    // 跨页跳转载荷：由来源页发起，目标页挂载时消费（消费后回调清空，防止切页复放）
    type NavTarget = {
        view: View;
        symbol?: string;
        name?: string;
        action?: "history" | "predict";
    };

    let activeView = $state<View>("favorites");
    let navTarget = $state<NavTarget | null>(null);

    function navigate(target: NavTarget) {
        navTarget = target;
        activeView = target.view;
    }
</script>

<div class="main-container">
    <nav class="sidebar">
        <div class="logo">StockAI</div>
        <ul>
            <li
                class:active={activeView === "favorites"}
                onclick={() => (activeView = "favorites")}
            >
                <span>⭐ 我的收藏</span>
            </li>
            <li
                class:active={activeView === "stock"}
                onclick={() => (activeView = "stock")}
            >
                <span>📈 股票预测</span>
            </li>
            <li
                class:active={activeView === "realtime"}
                onclick={() => (activeView = "realtime")}
            >
                <span>⌚ 实时行情</span>
            </li>
            <li
                class:active={activeView === "historical"}
                onclick={() => (activeView = "historical")}
            >
                <span>📅 历史数据</span>
            </li>
            <li
                class:active={activeView === "list"}
                onclick={() => (activeView = "list")}
            >
                <span>📋 股票列表</span>
            </li>
            <li
                class:active={activeView === "settings"}
                onclick={() => (activeView = "settings")}
            >
                <span>⚙️ 系统设置</span>
            </li>
        </ul>
    </nav>

    <main class="content">
        {#if activeView === "favorites"}
            <StockFavorites onNavigate={navigate} />
        {:else if activeView === "stock"}
            <StockPrediction
                navSymbol={navTarget?.view === "stock"
                    ? (navTarget.symbol ?? null)
                    : null}
                navAction={navTarget?.view === "stock" &&
                navTarget.action === "predict"
                    ? "predict"
                    : null}
                onNavConsumed={() => (navTarget = null)}
            />
        {:else if activeView === "list"}
            <StockList onNavigate={navigate} />
        {:else if activeView === "realtime"}
            <RealTimeData onNavigate={navigate} />
        {:else if activeView === "historical"}
            <HistoricalData
                navTarget={navTarget?.view === "historical" && navTarget.symbol
                    ? { symbol: navTarget.symbol, name: navTarget.name }
                    : null}
                onNavConsumed={() => (navTarget = null)}
            />
        {:else if activeView === "settings"}
            <Settings />
        {/if}
    </main>
</div>

<!-- 保持原有样式不变 -->
<style>
    :root {
        --sidebar-width: 240px;
        --primary-color: #1a1d24;
        --active-color: #6366f1;
        --hover-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #f8fafc;
    }

    .main-container {
        display: flex;
        min-height: 100vh;
        background: #0f172a;
        color: var(--text-primary);
    }

    .sidebar {
        width: var(--sidebar-width);
        background: var(--primary-color);
        padding: 1.5rem;
        position: relative;
    }

    .logo {
        font-size: 1.5rem;
        font-weight: 600;
        padding: 1rem;
        margin-bottom: 2rem;
        color: var(--active-color);
        text-align: center;
    }

    ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    li {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    li:hover {
        background: var(--hover-bg);
    }

    li.active {
        background: var(--active-color);
        font-weight: 500;
    }

    .content {
        flex: 1;
        padding: 2rem;
        background: #0f172a;
        overflow-y: auto;
    }

    @media (max-width: 768px) {
        :root {
            --sidebar-width: 100%;
        }

        .main-container {
            flex-direction: column;
        }

        .sidebar {
            width: 100%;
            height: auto;
        }

        .content {
            min-height: calc(100vh - 160px);
        }
    }
</style>
