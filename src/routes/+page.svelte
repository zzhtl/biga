<script lang="ts">
    import "./global.css";
    import type { NavTarget, View } from "./types";
    import {
        CalendarDays,
        Clock3,
        List,
        Settings as SettingsIcon,
        Star,
        TrendingUp,
    } from "lucide-svelte";

    let activeView = $state<View>("favorites");
    let navTarget = $state<NavTarget | null>(null);

    let favoritesModule: ReturnType<typeof importFavorites> | null = null;
    let predictionModule: ReturnType<typeof importPrediction> | null = null;
    let realtimeModule: ReturnType<typeof importRealtime> | null = null;
    let historicalModule: ReturnType<typeof importHistorical> | null = null;
    let listModule: ReturnType<typeof importList> | null = null;
    let settingsModule: ReturnType<typeof importSettings> | null = null;

    function importFavorites() { return import("./components/stock_favorites.svelte"); }
    function importPrediction() { return import("./components/stock_prediction.svelte"); }
    function importRealtime() { return import("./components/stock_realtime.svelte"); }
    function importHistorical() { return import("./components/stock_historical.svelte"); }
    function importList() { return import("./components/stock_list.svelte"); }
    function importSettings() { return import("./components/sys_settings.svelte"); }

    function loadFavorites() { return favoritesModule ??= importFavorites(); }
    function loadPrediction() { return predictionModule ??= importPrediction(); }
    function loadRealtime() { return realtimeModule ??= importRealtime(); }
    function loadHistorical() { return historicalModule ??= importHistorical(); }
    function loadList() { return listModule ??= importList(); }
    function loadSettings() { return settingsModule ??= importSettings(); }

    function navigate(target: NavTarget) {
        navTarget = target;
        activeView = target.view;
    }

    function selectView(view: View) {
        navTarget = null;
        activeView = view;
    }
</script>

<div class="main-container">
    <nav class="sidebar">
        <div class="logo">BigA</div>
        <ul>
            <li>
                <button class="nav-button" class:active={activeView === "favorites"} onclick={() => selectView("favorites")} aria-current={activeView === "favorites" ? "page" : undefined}>
                    <Star size={18} aria-hidden="true" />我的收藏
                </button>
            </li>
            <li>
                <button class="nav-button" class:active={activeView === "stock"} onclick={() => selectView("stock")} aria-current={activeView === "stock" ? "page" : undefined}>
                    <TrendingUp size={18} aria-hidden="true" />股票预测
                </button>
            </li>
            <li>
                <button class="nav-button" class:active={activeView === "realtime"} onclick={() => selectView("realtime")} aria-current={activeView === "realtime" ? "page" : undefined}>
                    <Clock3 size={18} aria-hidden="true" />实时行情
                </button>
            </li>
            <li>
                <button class="nav-button" class:active={activeView === "historical"} onclick={() => selectView("historical")} aria-current={activeView === "historical" ? "page" : undefined}>
                    <CalendarDays size={18} aria-hidden="true" />历史数据
                </button>
            </li>
            <li>
                <button class="nav-button" class:active={activeView === "list"} onclick={() => selectView("list")} aria-current={activeView === "list" ? "page" : undefined}>
                    <List size={18} aria-hidden="true" />股票列表
                </button>
            </li>
            <li>
                <button class="nav-button" class:active={activeView === "settings"} onclick={() => selectView("settings")} aria-current={activeView === "settings" ? "page" : undefined}>
                    <SettingsIcon size={18} aria-hidden="true" />系统设置
                </button>
            </li>
        </ul>
    </nav>

    <main class="content">
        {#if activeView === "favorites"}
            {#await loadFavorites() then module}
                {@const Component = module.default}
                <Component onNavigate={navigate} />
            {/await}
        {:else if activeView === "stock"}
            {#await loadPrediction() then module}
                {@const Component = module.default}
                <Component
                    navSymbol={navTarget?.view === "stock" ? (navTarget.symbol ?? null) : null}
                    navAction={navTarget?.view === "stock" && navTarget.action === "predict" ? "predict" : null}
                    onNavConsumed={() => (navTarget = null)}
                />
            {/await}
        {:else if activeView === "list"}
            {#await loadList() then module}
                {@const Component = module.default}
                <Component onNavigate={navigate} />
            {/await}
        {:else if activeView === "realtime"}
            {#await loadRealtime() then module}
                {@const Component = module.default}
                <Component onNavigate={navigate} />
            {/await}
        {:else if activeView === "historical"}
            {#await loadHistorical() then module}
                {@const Component = module.default}
                <Component
                    navTarget={navTarget?.view === "historical" && navTarget.symbol ? { symbol: navTarget.symbol, name: navTarget.name } : null}
                    onNavConsumed={() => (navTarget = null)}
                />
            {/await}
        {:else if activeView === "settings"}
            {#await loadSettings() then module}
                {@const Component = module.default}
                <Component />
            {/await}
        {/if}
    </main>
</div>

<style>
    .main-container {
        display: flex;
        min-height: 100vh;
        background: var(--app-bg);
        color: var(--text-primary);
    }

    .sidebar {
        position: sticky;
        top: 0;
        width: 220px;
        height: 100vh;
        box-sizing: border-box;
        flex: 0 0 220px;
        padding: 1.25rem 1rem;
        border-right: 1px solid var(--border);
        background: var(--sidebar-bg);
    }

    .logo {
        height: 52px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 1.25rem;
        color: var(--text-primary);
        font-size: 1.25rem;
        font-weight: 760;
        letter-spacing: 0;
    }

    ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    li {
        margin: 0.25rem 0;
    }

    .nav-button {
        width: 100%;
        min-height: 42px;
        padding: 0.65rem 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        border: 1px solid transparent;
        border-radius: 6px;
        background: transparent;
        color: var(--text-secondary);
        cursor: pointer;
        font: inherit;
        font-size: 0.88rem;
        text-align: left;
        white-space: nowrap;
        transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }

    .nav-button:hover {
        background: var(--surface-2);
        color: var(--text-primary);
    }

    .nav-button.active {
        border-color: rgba(49, 181, 200, 0.25);
        background: var(--accent-muted);
        color: var(--accent-hover);
        font-weight: 650;
    }

    .content {
        min-width: 0;
        flex: 1;
        padding: 1.75rem 2rem 3rem;
        background: var(--app-bg);
    }

    @media (max-width: 1100px) {
        .sidebar {
            width: 196px;
            flex-basis: 196px;
        }

        .content { padding-inline: 1.25rem; }
    }

    @media (max-width: 760px) {
        .main-container {
            flex-direction: column;
        }

        .sidebar {
            position: sticky;
            top: 0;
            width: 100%;
            height: auto;
            flex-basis: auto;
            padding: 0.6rem 0.75rem;
            border-right: 0;
            border-bottom: 1px solid var(--border);
            z-index: 20;
        }

        .logo {
            height: auto;
            margin: 0 0 0.35rem;
            padding: 0.2rem 0.4rem;
            font-size: 1rem;
        }

        ul {
            display: flex;
            gap: 0.4rem;
            overflow-x: auto;
            padding-bottom: 0.2rem;
        }

        li {
            flex: 0 0 auto;
            margin: 0;
        }

        .nav-button {
            width: auto;
            min-height: 40px;
            padding: 0.55rem 0.7rem;
        }

        .content {
            min-width: 0;
            padding: 1rem 0.85rem 2rem;
        }
    }
</style>
