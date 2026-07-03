<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { confirm } from "@tauri-apps/plugin-dialog";
    import { onMount } from "svelte";

    // ---------- 类型（内联，snake_case 对齐后端 serde） ----------
    interface WatchlistItem {
        symbol: string;
        name: string;
        added_at: string;
        sort_order: number;
        latest_date: string | null;
        staleness_days: number | null;
        close: number | null;
        change_percent: number | null;
        turnover_rate: number | null;
        volume_ratio: number | null;
        change_5d: number | null;
        change_10d: number | null;
        change_20d: number | null;
        change_ytd: number | null;
        change_1y: number | null;
        week52_high: number | null;
        week52_low: number | null;
        week52_position: number | null;
        pe: number | null;
        pb: number | null;
        circulating_market_cap_yi: number | null;
        roe: number | null;
        revenue_growth: number | null;
        report_date: string | null;
    }

    interface PredictionInterval {
        confidence: number;
        lower_change_percent: number;
        upper_change_percent: number;
        lower_price: number;
        upper_price: number;
    }

    // 一键综合预测报告（只声明对比表用到的字段）
    interface ComprehensiveReport {
        symbol: string;
        name: string;
        generated_at: string;
        latest_date: string;
        staleness_days: number;
        current_price: number;
        prediction_days: number;
        direction: string;
        signal_strength: number;
        expected_change_percent: number;
        interval: PredictionInterval | null;
        current_advice: string;
        risk_level: string;
        adaptive_score: number;
        buy_point_count: number;
        sell_point_count: number;
        nearest_support: number | null;
        nearest_resistance: number | null;
        key_factors: string[];
        momentum_5d: number | null;
        momentum_20d: number | null;
        momentum_60d: number | null;
        week52_position: number | null;
        up_ratio_20d: number | null;
        up_ratio_60d: number | null;
        up_ratio_250d: number | null;
        avg_daily_change_250d: number | null;
        disclaimer: string;
    }

    interface SearchStock {
        symbol: string;
        name: string;
        industry: string;
        category: string;
    }

    type View =
        | "favorites"
        | "stock"
        | "list"
        | "realtime"
        | "historical"
        | "settings";
    type NavTarget = {
        view: View;
        symbol?: string;
        name?: string;
        action?: "history" | "predict";
    };
    type Props = { onNavigate?: (target: NavTarget) => void };

    let { onNavigate = () => {} }: Props = $props();

    // ---------- 状态 ----------
    let items = $state<WatchlistItem[]>([]);
    let loading = $state(true);
    let errorMessage = $state("");

    // 搜索添加
    let searchQuery = $state("");
    let searchResults = $state<SearchStock[]>([]);
    let isDropdownOpen = $state(false);
    let searchDebounce: ReturnType<typeof setTimeout> | null = null;

    // 批量刷新 / 单票刷新
    let refreshing = $state<{
        done: number;
        total: number;
        current: string;
        failures: string[];
    } | null>(null);
    let rowRefreshing = $state<Record<string, boolean>>({});

    // 一键预测
    let predicting = $state<{
        done: number;
        total: number;
        current: string;
    } | null>(null);
    let reports = $state<Record<string, ComprehensiveReport>>({});
    let reportErrors = $state<Record<string, string>>({});
    let showComparison = $state(false);

    // 对比表排序（客户端排序；仅为相对强弱浏览便利，不合成任何新综合分）
    type SortKey =
        | "signal_strength"
        | "adaptive_score"
        | "expected_change_percent"
        | "momentum_5d"
        | "momentum_20d"
        | "momentum_60d"
        | "week52_position";
    let sortKey = $state<SortKey>("signal_strength");
    let sortDir = $state<"asc" | "desc">("desc");

    const busy = $derived(refreshing !== null || predicting !== null);

    // ---------- 数据加载 ----------
    async function loadOverview(silent = false) {
        try {
            if (!silent) loading = true;
            items = await invoke<WatchlistItem[]>("get_watchlist_overview");
        } catch (e) {
            errorMessage = `加载收藏失败: ${e}`;
        } finally {
            loading = false;
        }
    }

    onMount(() => {
        loadOverview();
    });

    // ---------- 搜索添加收藏 ----------
    function onSearchInput() {
        if (searchDebounce) clearTimeout(searchDebounce);
        const q = searchQuery.trim();
        if (!q) {
            searchResults = [];
            isDropdownOpen = false;
            return;
        }
        searchDebounce = setTimeout(async () => {
            try {
                const res = await invoke<SearchStock[]>("get_stock_list", {
                    search: q,
                });
                searchResults = res.slice(0, 30);
                isDropdownOpen = true;
            } catch {
                searchResults = [];
            }
        }, 300);
    }

    function handleDocumentClick(event: MouseEvent) {
        const target = event.target as HTMLElement;
        if (!target.closest(".add-select") && isDropdownOpen) {
            isDropdownOpen = false;
        }
    }

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

    async function addStock(stock: SearchStock) {
        try {
            await invoke("add_to_watchlist", { symbol: stock.symbol });
            searchQuery = "";
            searchResults = [];
            isDropdownOpen = false;
            await loadOverview(true);
        } catch (e) {
            errorMessage = `添加收藏失败: ${e}`;
        }
    }

    async function removeStock(item: WatchlistItem) {
        const ok = await confirm(
            `确定将 ${item.symbol} ${item.name} 移出收藏池吗？`,
            { title: "移除收藏" },
        );
        if (!ok) return;
        try {
            await invoke("remove_from_watchlist", { symbol: item.symbol });
            const next = { ...reports };
            delete next[item.symbol];
            reports = next;
            await loadOverview(true);
        } catch (e) {
            errorMessage = `移除失败: ${e}`;
        }
    }

    // ---------- 刷新（走 zhitu API，受每日额度限制） ----------
    async function refreshOne(symbol: string) {
        rowRefreshing = { ...rowRefreshing, [symbol]: true };
        try {
            await invoke("refresh_historical_data", { symbol });
            await loadOverview(true);
        } catch (e) {
            errorMessage = `刷新 ${symbol} 失败: ${e}（不影响已有数据；可能是当日 API 额度耗尽）`;
        } finally {
            rowRefreshing = { ...rowRefreshing, [symbol]: false };
        }
    }

    async function refreshAll() {
        if (!items.length || busy) return;
        errorMessage = "";
        const symbols = items.map((i) => i.symbol);
        refreshing = {
            done: 0,
            total: symbols.length,
            current: "",
            failures: [],
        };
        let consecutiveFailures = 0;
        for (const symbol of symbols) {
            refreshing = { ...refreshing!, current: symbol };
            try {
                await invoke("refresh_historical_data", { symbol });
                consecutiveFailures = 0;
                await loadOverview(true); // 每票成功后即时可见
            } catch (e) {
                consecutiveFailures += 1;
                refreshing = {
                    ...refreshing!,
                    failures: [...refreshing!.failures, `${symbol}: ${e}`],
                };
            }
            refreshing = { ...refreshing!, done: refreshing!.done + 1 };
            // 连续失败早停：探测当日 API 额度耗尽，避免浪费剩余请求
            if (consecutiveFailures >= 3) {
                errorMessage = `连续 ${consecutiveFailures} 只刷新失败，疑似当日 API 额度耗尽，已停止批量刷新（已处理 ${refreshing!.done}/${refreshing!.total}）。一键预测为纯本地计算不受影响，但请留意数据新鲜度。`;
                break;
            }
        }
        refreshing = null;
        await loadOverview(true);
    }

    // ---------- 一键综合预测（纯本地计算，不消耗 API 额度） ----------
    async function predictOne(symbol: string) {
        try {
            const report = await invoke<ComprehensiveReport>(
                "comprehensive_predict",
                { symbol, days: 5 },
            );
            reports = { ...reports, [symbol]: report };
            const next = { ...reportErrors };
            delete next[symbol];
            reportErrors = next;
        } catch (e) {
            reportErrors = { ...reportErrors, [symbol]: String(e) };
        }
    }

    async function predictAll() {
        if (!items.length || busy) return;
        errorMessage = "";
        const symbols = items.map((i) => i.symbol);
        predicting = { done: 0, total: symbols.length, current: "" };
        for (const symbol of symbols) {
            predicting = { ...predicting!, current: symbol };
            await predictOne(symbol);
            predicting = { ...predicting!, done: predicting!.done + 1 };
        }
        predicting = null;
        showComparison = true;
    }

    // ---------- 跳转 ----------
    function goHistory(symbol: string, name: string) {
        onNavigate({ view: "historical", symbol, name, action: "history" });
    }

    function goPredict(symbol: string) {
        onNavigate({ view: "stock", symbol, action: "predict" });
    }

    // ---------- 对比表 ----------
    const comparisonRows = $derived.by(() => {
        const rows = items
            .map((it) => ({
                item: it,
                report: reports[it.symbol] ?? null,
                error: reportErrors[it.symbol] ?? null,
            }))
            .filter((r) => r.report || r.error);
        const dir = sortDir === "asc" ? 1 : -1;
        const val = (r: (typeof rows)[number]) => {
            const v = r.report?.[sortKey];
            return typeof v === "number" && Number.isFinite(v)
                ? v
                : Number.NEGATIVE_INFINITY;
        };
        return rows.sort((a, b) => {
            if (!a.report && !b.report) return 0;
            if (!a.report) return 1; // 失败行沉底
            if (!b.report) return -1;
            return (val(a) - val(b)) * dir;
        });
    });

    function sortBy(key: SortKey) {
        if (key === sortKey) {
            sortDir = sortDir === "asc" ? "desc" : "asc";
        } else {
            sortKey = key;
            sortDir = "desc";
        }
    }

    const firstDisclaimer = $derived(
        Object.values(reports)[0]?.disclaimer ?? "",
    );

    // ---------- 格式化 ----------
    function fmt(
        v: number | null | undefined,
        digits = 2,
        suffix = "",
    ): string {
        if (v == null || !Number.isFinite(v)) return "—";
        return `${v.toFixed(digits)}${suffix}`;
    }

    function fmtSigned(v: number | null | undefined): string {
        if (v == null || !Number.isFinite(v)) return "—";
        return `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
    }

    function fmtRatio(v: number | null | undefined): string {
        if (v == null || !Number.isFinite(v)) return "—";
        return `${(v * 100).toFixed(0)}%`;
    }

    // A股惯例红涨绿跌（与实时行情页一致）
    function updown(v: number | null | undefined): string {
        if (v == null || !Number.isFinite(v) || v === 0) return "";
        return v > 0 ? "up" : "down";
    }

    function staleInfo(days: number | null): { cls: string; title: string } {
        if (days == null) return { cls: "stale-old", title: "无数据，请刷新" };
        if (days <= 1) return { cls: "stale-fresh", title: "数据最新" };
        if (days <= 4)
            return {
                cls: "stale-warn",
                title: "可能非最新（周末/假期或未刷新）",
            };
        return { cls: "stale-old", title: "数据陈旧，请刷新" };
    }

    function directionClass(direction: string): string {
        if (/买|多|涨/.test(direction)) return "up";
        if (/卖|空|跌/.test(direction)) return "down";
        return "";
    }

    function rowKeydown(e: KeyboardEvent, fn: () => void) {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fn();
        }
    }
</script>

<div class="container">
    <h1>⭐ 我的收藏（选票池）</h1>

    <!-- 工具条 -->
    <div class="toolbar">
        <div class="custom-select add-select">
            <input
                type="text"
                class="search-input"
                bind:value={searchQuery}
                oninput={onSearchInput}
                onfocus={() => {
                    if (searchResults.length) isDropdownOpen = true;
                }}
                placeholder="搜索代码/名称/行业，添加到收藏..."
            />
            {#if isDropdownOpen}
                <div class="dropdown-list">
                    {#each searchResults as stock (stock.symbol)}
                        <div
                            class="dropdown-item"
                            role="button"
                            tabindex="0"
                            onclick={() => addStock(stock)}
                            onkeydown={(e) =>
                                rowKeydown(e, () => addStock(stock))}
                        >
                            <span class="symbol">{stock.symbol}</span>
                            <span class="name">{stock.name}</span>
                            <span class="industry">{stock.industry}</span>
                        </div>
                    {:else}
                        <div class="dropdown-empty">未找到匹配的股票</div>
                    {/each}
                </div>
            {/if}
        </div>

        <div class="action-buttons">
            <button
                class="btn"
                onclick={refreshAll}
                disabled={!items.length || busy}
                title="逐票拉取最新行情/估值/基本面（消耗 API 额度）"
            >
                🔄 全部刷新
            </button>
            <button
                class="btn primary"
                onclick={predictAll}
                disabled={!items.length || busy}
                title="对收藏池全部股票做综合分析并生成对比表（纯本地计算）"
            >
                🎯 一键预测全部
            </button>
        </div>
    </div>

    <!-- 进度条 -->
    {#if refreshing}
        <div class="progress">
            <div class="progress-text">
                刷新中 {refreshing.done}/{refreshing.total} · {refreshing.current}
                {#if refreshing.failures.length}
                    <span class="fail-count"
                        >失败 {refreshing.failures.length}</span
                    >
                {/if}
            </div>
            <div class="progress-track">
                <div
                    class="progress-fill"
                    style="width: {refreshing.total
                        ? (refreshing.done / refreshing.total) * 100
                        : 0}%"
                ></div>
            </div>
            {#if refreshing.failures.length}
                <div class="failures">
                    {refreshing.failures.join("；")}
                </div>
            {/if}
        </div>
    {/if}
    {#if predicting}
        <div class="progress">
            <div class="progress-text">
                分析中 {predicting.done}/{predicting.total} · {predicting.current}
            </div>
            <div class="progress-track">
                <div
                    class="progress-fill"
                    style="width: {predicting.total
                        ? (predicting.done / predicting.total) * 100
                        : 0}%"
                ></div>
            </div>
        </div>
    {/if}

    {#if errorMessage}
        <div class="error-banner">
            {errorMessage}
            <button class="dismiss" onclick={() => (errorMessage = "")}
                >✕</button
            >
        </div>
    {/if}

    <!-- 主内容 -->
    {#if loading}
        <div class="loading">加载中...</div>
    {:else if items.length === 0}
        <div class="empty-card">
            <p class="empty-title">还没有收藏股票</p>
            <p class="empty-hint">
                在上方搜索添加，或到股票列表 / 实时行情页点击 ☆ 星标收藏。
                收藏池将成为你的选票池：可一键刷新全量数据、对比多周期表现与估值，
                并对整池做综合分析辅助判断。
            </p>
            <div class="empty-actions">
                <button
                    class="btn"
                    onclick={() => onNavigate({ view: "list" })}
                >
                    📋 去股票列表
                </button>
                <button
                    class="btn"
                    onclick={() => onNavigate({ view: "realtime" })}
                >
                    ⌚ 去实时行情
                </button>
            </div>
        </div>
    {:else}
        <div class="table-wrap">
            <table class="fav-table">
                <thead>
                    <tr>
                        <th>代码</th>
                        <th>名称</th>
                        <th>现价</th>
                        <th>当日</th>
                        <th>5日</th>
                        <th>10日</th>
                        <th>20日</th>
                        <th>年初至今</th>
                        <th>一年</th>
                        <th>量比</th>
                        <th>换手%</th>
                        <th>PE</th>
                        <th>PB</th>
                        <th>市值(亿)</th>
                        <th title="现价在近一年最高/最低区间的位置：0=年内低点，100=年内高点"
                            >52周位</th
                        >
                        <th>ROE%</th>
                        <th>营收增%</th>
                        <th>数据日期</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody>
                    {#each items as item (item.symbol)}
                        <tr
                            class="row-click"
                            tabindex="0"
                            onclick={() => goHistory(item.symbol, item.name)}
                            onkeydown={(e) =>
                                rowKeydown(e, () =>
                                    goHistory(item.symbol, item.name),
                                )}
                            title="点击查看历史K线"
                        >
                            <td class="symbol">{item.symbol}</td>
                            <td class="name">{item.name || "—"}</td>
                            <td>{fmt(item.close)}</td>
                            <td class={updown(item.change_percent)}
                                >{fmtSigned(item.change_percent)}</td
                            >
                            <td class={updown(item.change_5d)}
                                >{fmtSigned(item.change_5d)}</td
                            >
                            <td class={updown(item.change_10d)}
                                >{fmtSigned(item.change_10d)}</td
                            >
                            <td class={updown(item.change_20d)}
                                >{fmtSigned(item.change_20d)}</td
                            >
                            <td class={updown(item.change_ytd)}
                                >{fmtSigned(item.change_ytd)}</td
                            >
                            <td class={updown(item.change_1y)}
                                >{fmtSigned(item.change_1y)}</td
                            >
                            <td>{fmt(item.volume_ratio)}</td>
                            <td>{fmt(item.turnover_rate)}</td>
                            <td>{fmt(item.pe, 1)}</td>
                            <td>{fmt(item.pb, 2)}</td>
                            <td>{fmt(item.circulating_market_cap_yi, 0)}</td>
                            <td>{fmt(item.week52_position, 0)}</td>
                            <td>{fmt(item.roe, 2)}</td>
                            <td class={updown(item.revenue_growth)}
                                >{fmt(item.revenue_growth, 2)}</td
                            >
                            <td>
                                {#if item.latest_date}
                                    {@const s = staleInfo(item.staleness_days)}
                                    <span
                                        class="stale-dot {s.cls}"
                                        title={s.title}
                                    ></span>{item.latest_date}
                                {:else}
                                    <span
                                        class="stale-dot stale-old"
                                        title="无本地数据，请刷新"
                                    ></span>无数据
                                {/if}
                            </td>
                            <td class="ops">
                                <button
                                    class="mini"
                                    title="刷新该股全部数据"
                                    disabled={busy ||
                                        rowRefreshing[item.symbol]}
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        refreshOne(item.symbol);
                                    }}
                                >
                                    {rowRefreshing[item.symbol] ? "…" : "🔄"}
                                </button>
                                <button
                                    class="mini"
                                    title="跳转预测页生成综合报告"
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        goPredict(item.symbol);
                                    }}
                                >
                                    🎯
                                </button>
                                <button
                                    class="mini danger"
                                    title="移出收藏"
                                    disabled={busy}
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        removeStock(item);
                                    }}
                                >
                                    ✕
                                </button>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        <!-- 一键预测对比表 -->
        {#if showComparison && comparisonRows.length}
            <div class="compare-card">
                <div class="compare-head">
                    <h2>🎯 综合分析对比</h2>
                    <span class="compare-sub"
                        >信号强度非命中概率；排序仅为相对强弱浏览便利，非收益预测排名</span
                    >
                </div>
                <div class="table-wrap">
                    <table class="fav-table compare-table">
                        <thead>
                            <tr>
                                <th>代码</th>
                                <th>名称</th>
                                <th>信号方向</th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("signal_strength")}
                                    title="引擎信号强度 0.25-0.92（非方向命中概率）"
                                >
                                    信号强度{sortKey === "signal_strength"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("adaptive_score")}
                                    title="自适应多因子得分（0-100）"
                                >
                                    多因子分{sortKey === "adaptive_score"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    class="sortable"
                                    onclick={() =>
                                        sortBy("expected_change_percent")}
                                    title="点预测为历史无条件漂移锚，仅供参考"
                                >
                                    预期5日{sortKey ===
                                    "expected_change_percent"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th title="真实不确定性的诚实表达：约80%概率落在带内"
                                    >80%区间</th
                                >
                                <th>风险</th>
                                <th>建议</th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("momentum_5d")}
                                >
                                    动量5日{sortKey === "momentum_5d"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("momentum_20d")}
                                >
                                    20日{sortKey === "momentum_20d"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("momentum_60d")}
                                >
                                    60日{sortKey === "momentum_60d"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    class="sortable"
                                    onclick={() => sortBy("week52_position")}
                                >
                                    52周位{sortKey === "week52_position"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th
                                    title="近20/60/250个交易日日线上涨占比——无技能基准，供对照引擎信号"
                                    >上涨占比 20/60/250日</th
                                >
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each comparisonRows as row (row.item.symbol)}
                                {#if row.report}
                                    {@const r = row.report}
                                    <tr>
                                        <td class="symbol">{r.symbol}</td>
                                        <td class="name"
                                            >{r.name ||
                                                row.item.name ||
                                                "—"}</td
                                        >
                                        <td class={directionClass(r.direction)}
                                            >{r.direction}</td
                                        >
                                        <td>{fmt(r.signal_strength, 2)}</td>
                                        <td>{fmt(r.adaptive_score, 0)}</td>
                                        <td
                                            class={updown(
                                                r.expected_change_percent,
                                            )}
                                            >{fmtSigned(
                                                r.expected_change_percent,
                                            )}</td
                                        >
                                        <td class="interval">
                                            {#if r.interval}
                                                {fmtSigned(
                                                    r.interval
                                                        .lower_change_percent,
                                                )} ~ {fmtSigned(
                                                    r.interval
                                                        .upper_change_percent,
                                                )}
                                            {:else}
                                                —
                                            {/if}
                                        </td>
                                        <td>{r.risk_level || "—"}</td>
                                        <td
                                            class="advice"
                                            title={r.current_advice}
                                            >{r.current_advice || "—"}</td
                                        >
                                        <td class={updown(r.momentum_5d)}
                                            >{fmtSigned(r.momentum_5d)}</td
                                        >
                                        <td class={updown(r.momentum_20d)}
                                            >{fmtSigned(r.momentum_20d)}</td
                                        >
                                        <td class={updown(r.momentum_60d)}
                                            >{fmtSigned(r.momentum_60d)}</td
                                        >
                                        <td>{fmt(r.week52_position, 0)}</td>
                                        <td class="base-rates">
                                            {fmtRatio(r.up_ratio_20d)} / {fmtRatio(
                                                r.up_ratio_60d,
                                            )} / {fmtRatio(r.up_ratio_250d)}
                                        </td>
                                        <td class="ops">
                                            <button
                                                class="mini"
                                                title="跳转预测页看完整报告"
                                                onclick={() =>
                                                    goPredict(r.symbol)}
                                            >
                                                🎯
                                            </button>
                                            <button
                                                class="mini"
                                                title="查看历史K线"
                                                onclick={() =>
                                                    goHistory(
                                                        r.symbol,
                                                        r.name ||
                                                            row.item.name,
                                                    )}
                                            >
                                                📅
                                            </button>
                                        </td>
                                    </tr>
                                {:else}
                                    <tr class="failed-row">
                                        <td class="symbol"
                                            >{row.item.symbol}</td
                                        >
                                        <td class="name"
                                            >{row.item.name || "—"}</td
                                        >
                                        <td colspan="11" class="row-error"
                                            >分析失败：{row.error}</td
                                        >
                                        <td></td>
                                        <td class="ops">
                                            <button
                                                class="mini"
                                                title="重试"
                                                onclick={() =>
                                                    predictOne(
                                                        row.item.symbol,
                                                    )}
                                            >
                                                ↻
                                            </button>
                                        </td>
                                    </tr>
                                {/if}
                            {/each}
                        </tbody>
                    </table>
                </div>
                {#if firstDisclaimer}
                    <p class="disclaimer">⚠️ {firstDisclaimer}</p>
                {/if}
            </div>
        {/if}
    {/if}
</div>

<style>
    .container {
        max-width: 1400px;
        margin: 0 auto;
    }

    h1 {
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }

    /* 工具条 */
    .toolbar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .custom-select {
        position: relative;
        flex: 0 0 auto;
        min-width: 260px;
        max-width: 360px;
    }

    .search-input {
        width: 100%;
        padding: 0.7rem 1.2rem;
        border: 2px solid #3b82f6;
        border-radius: 24px;
        background: #2d2d30;
        color: #ffffff;
        font-size: 0.95rem;
        transition: border-color 0.3s ease;
    }

    .search-input:focus {
        outline: none;
        border-color: #0ea5e9;
    }

    .search-input::placeholder {
        color: #666666;
    }

    .dropdown-list {
        position: absolute;
        top: calc(100% + 4px);
        left: 0;
        right: 0;
        max-height: 320px;
        overflow-y: auto;
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 0.5rem;
        z-index: 30;
    }

    .dropdown-item {
        display: flex;
        gap: 0.6rem;
        align-items: baseline;
        padding: 0.55rem 0.9rem;
        cursor: pointer;
    }

    .dropdown-item:hover {
        background: rgba(99, 102, 241, 0.25);
    }

    .dropdown-item .symbol {
        font-family: monospace;
        color: #38bdf8;
    }

    .dropdown-item .name {
        color: #e2e8f0;
    }

    .dropdown-item .industry {
        margin-left: auto;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .dropdown-empty {
        padding: 0.8rem;
        color: #94a3b8;
        text-align: center;
    }

    .action-buttons {
        display: flex;
        gap: 0.6rem;
    }

    .btn {
        padding: 0.6rem 1.1rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 0.5rem;
        background: rgba(255, 255, 255, 0.06);
        color: #f8fafc;
        font-size: 0.95rem;
        cursor: pointer;
        transition: background 0.2s ease;
    }

    .btn:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.12);
    }

    .btn.primary {
        background: #6366f1;
        border-color: #6366f1;
    }

    .btn.primary:hover:not(:disabled) {
        background: #818cf8;
    }

    .btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }

    /* 进度条 */
    .progress {
        margin: 0.6rem 0 1rem;
        padding: 0.7rem 1rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.5rem;
    }

    .progress-text {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-bottom: 0.45rem;
        font-variant-numeric: tabular-nums;
    }

    .fail-count {
        margin-left: 0.6rem;
        color: #f59e0b;
    }

    .progress-track {
        height: 6px;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #6366f1;
        transition: width 0.25s ease;
    }

    .failures {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #f59e0b;
        word-break: break-all;
    }

    /* 错误横幅 */
    .error-banner {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        margin: 0.6rem 0 1rem;
        padding: 0.7rem 1rem;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 0.5rem;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    .error-banner .dismiss {
        margin-left: auto;
        background: none;
        border: none;
        color: #fca5a5;
        cursor: pointer;
        font-size: 0.9rem;
    }

    /* 空态 */
    .empty-card {
        margin-top: 2rem;
        padding: 2.5rem 2rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 0.75rem;
        text-align: center;
    }

    .empty-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .empty-hint {
        color: #94a3b8;
        max-width: 560px;
        margin: 0 auto 1.2rem;
        line-height: 1.7;
    }

    .empty-actions {
        display: flex;
        gap: 0.8rem;
        justify-content: center;
    }

    /* 表格 */
    .table-wrap {
        overflow-x: auto;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
    }

    .fav-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }

    .fav-table th {
        position: sticky;
        top: 0;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.7rem 0.6rem;
        text-align: left;
        font-weight: 600;
        color: #e2e8f0;
    }

    .fav-table td {
        padding: 0.55rem 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
    }

    .row-click {
        cursor: pointer;
    }

    .row-click:hover td {
        background: rgba(99, 102, 241, 0.1);
    }

    .fav-table .symbol {
        font-family: monospace;
        color: #38bdf8;
    }

    .fav-table .name {
        color: #f8fafc;
    }

    /* A股惯例：红涨绿跌 */
    .up {
        color: #ef4444;
    }

    .down {
        color: #10b981;
    }

    /* 新鲜度徽标 */
    .stale-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
        vertical-align: middle;
    }

    .stale-fresh {
        background: #10b981;
    }

    .stale-warn {
        background: #f59e0b;
    }

    .stale-old {
        background: #ef4444;
    }

    /* 操作按钮 */
    .ops {
        white-space: nowrap;
    }

    .mini {
        padding: 0.25rem 0.45rem;
        margin-right: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 0.35rem;
        background: rgba(255, 255, 255, 0.06);
        color: #f8fafc;
        font-size: 0.8rem;
        cursor: pointer;
    }

    .mini:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.15);
    }

    .mini:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }

    .mini.danger:hover:not(:disabled) {
        background: rgba(239, 68, 68, 0.3);
        border-color: rgba(239, 68, 68, 0.5);
    }

    /* 对比卡 */
    .compare-card {
        margin-top: 1.5rem;
        padding: 1rem 1.2rem 1.2rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.75rem;
    }

    .compare-head {
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .compare-head h2 {
        font-size: 1.1rem;
    }

    .compare-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .sortable {
        cursor: pointer;
        user-select: none;
    }

    .sortable:hover {
        color: #a5b4fc;
    }

    .interval {
        font-size: 0.82rem;
    }

    .advice {
        max-width: 180px;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .base-rates {
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    .failed-row .row-error {
        color: #f59e0b;
        font-size: 0.82rem;
        white-space: normal;
    }

    .disclaimer {
        margin-top: 0.9rem;
        font-size: 0.78rem;
        line-height: 1.6;
        color: #94a3b8;
    }

    .loading {
        padding: 2rem;
        text-align: center;
        color: #ffffff;
    }

    @media (max-width: 768px) {
        .custom-select {
            min-width: 100%;
            max-width: 100%;
        }

        .toolbar {
            flex-direction: column;
            align-items: stretch;
        }

        .action-buttons {
            justify-content: stretch;
        }

        .action-buttons .btn {
            flex: 1;
        }
    }
</style>
