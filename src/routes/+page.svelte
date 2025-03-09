<script lang="ts">
    import StockPrediction from "./components/stock_prediction.svelte";
    import Settings from "./components/sys_settings.svelte";
    import RealTimeData from "./components/stock_realtime.svelte";
    import HistoricalData from "./components/stock_historical.svelte";

    let activeView: "stock" | "realtime" | "historical" | "settings" = "stock";
</script>

<div class="main-container">
    <nav class="sidebar">
        <div class="logo">StockAI</div>
        <ul>
            <li
                class:active={activeView === "stock"}
                on:click={() => (activeView = "stock")}
            >
                <span>📈 股票预测</span>
            </li>
            <li
                class:active={activeView === "realtime"}
                on:click={() => (activeView = "realtime")}
            >
                <span>⌚ 实时行情</span>
            </li>
            <li
                class:active={activeView === "historical"}
                on:click={() => (activeView = "historical")}
            >
                <span>📅 历史数据</span>
            </li>
            <li
                class:active={activeView === "settings"}
                on:click={() => (activeView = "settings")}
            >
                <span>⚙️ 系统设置</span>
            </li>
        </ul>
    </nav>

    <main class="content">
        {#if activeView === "stock"}
            <StockPrediction />
        {:else if activeView === "realtime"}
            <RealTimeData />
        {:else if activeView === "historical"}
            <HistoricalData />
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
