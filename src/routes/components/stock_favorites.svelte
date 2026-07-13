<script lang="ts">
    import { confirm } from "@tauri-apps/plugin-dialog";
    import { onMount } from "svelte";
    import { CalendarDays, Clock3, List, RefreshCw, RotateCw, Search, ShieldAlert, Target, Trophy, X } from "lucide-svelte";
    import RiskAlertPanel from "./risk_alert_panel.svelte";
    import { errorMessage as readableError, getStockList, invokeCommand, refreshHistoricalData, setWatchlistMembership } from "../services";
    import type { NavTarget, RiskLevel, RiskSummary } from "../types";

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
        risk_summary: RiskSummary;
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
    let searchSequence = 0;

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
    // 一键优选：从综合报告中选出相对强弱靠前的几只（描述性参考，非上涨概率）
    let showPicks = $state(false);
    let riskFilter = $state<"all" | RiskLevel>("all");
    const TOP_PICK_COUNT = 3;

    // 对比表排序（客户端排序；仅为相对强弱浏览便利，不合成任何新综合分）
    type SortKey =
        | "signal_strength"
        | "adaptive_score"
        | "expected_change_percent"
        | "momentum_5d"
        | "momentum_20d"
        | "momentum_60d"
        | "week52_position"
        | "risk_level";
    let sortKey = $state<SortKey>("signal_strength");
    let sortDir = $state<"asc" | "desc">("desc");

    const busy = $derived(refreshing !== null || predicting !== null);

    // 与后端 canonical_symbol 同口径：提取到恰好 6 位数字则用之，否则原样 trim
    function sixDigit(value: string): string {
        const digits = value.replace(/\D/g, "");
        return digits.length === 6 ? digits : value.trim();
    }

    // 已在收藏池中的代码集合（下拉搜索里标识"已收藏"）
    const pooledSymbols = $derived(new Set(items.map((i) => i.symbol)));

    // ---------- 数据加载 ----------
    async function loadOverview(silent = false) {
        try {
            if (!silent) loading = true;
            items = await invokeCommand<WatchlistItem[]>("get_watchlist_overview");
        } catch (e) {
            errorMessage = readableError(e, "收藏池加载失败");
        } finally {
            loading = false;
        }
    }

    onMount(() => {
        void loadOverview();
        return () => {
            searchSequence += 1;
            if (searchDebounce) clearTimeout(searchDebounce);
        };
    });

    // ---------- 搜索添加收藏 ----------
    function onSearchInput() {
        const sequence = ++searchSequence;
        if (searchDebounce) clearTimeout(searchDebounce);
        const q = searchQuery.trim();
        if (!q) {
            searchResults = [];
            isDropdownOpen = false;
            return;
        }
        searchDebounce = setTimeout(async () => {
            try {
                const res = await getStockList(q, 1, 30);
                if (sequence !== searchSequence) return;
                searchResults = res.data.slice(0, 30);
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
            await setWatchlistMembership(stock.symbol, false);
            searchQuery = "";
            searchResults = [];
            isDropdownOpen = false;
            await loadOverview(true);
        } catch (e) {
            errorMessage = readableError(e, "添加收藏失败");
        }
    }

    async function removeStock(item: WatchlistItem) {
        const ok = await confirm(
            `确定将 ${item.symbol} ${item.name} 移出收藏池吗？`,
            { title: "移除收藏" },
        );
        if (!ok) return;
        try {
            await setWatchlistMembership(item.symbol, true);
            invalidateReport(item.symbol);
            await loadOverview(true);
        } catch (e) {
            errorMessage = readableError(e, "移除收藏失败");
        }
    }

    function invalidateReport(symbol: string) {
        const nextReports = { ...reports };
        delete nextReports[symbol];
        reports = nextReports;

        const nextErrors = { ...reportErrors };
        delete nextErrors[symbol];
        reportErrors = nextErrors;
    }

    // ---------- 刷新（走 zhitu API，受每日额度限制） ----------
    async function refreshOne(symbol: string) {
        rowRefreshing = { ...rowRefreshing, [symbol]: true };
        errorMessage = "";
        try {
            await refreshHistoricalData(symbol);
            invalidateReport(symbol);
            await loadOverview(true);
        } catch (e) {
            errorMessage = `刷新 ${symbol} 失败：${readableError(e, "数据服务暂不可用")}（已有数据仍可使用）`;
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
                await refreshHistoricalData(symbol);
                invalidateReport(symbol);
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
        // 进度条消失后失败信息不能静默丢失：汇总到错误横幅
        const failures = refreshing?.failures ?? [];
        refreshing = null;
        await loadOverview(true);
        if (failures.length && !errorMessage) {
            const names = failures
                .slice(0, 5)
                .map((f) => f.split(":")[0])
                .join("、");
            errorMessage = `批量刷新完成，${failures.length} 只失败（${names}${failures.length > 5 ? " 等" : ""}），其余成功。失败可能因当日额度耗尽或代码无效，可稍后单票重试。`;
        }
    }

    // ---------- 一键综合预测（纯本地计算，不消耗 API 额度） ----------
    async function predictOne(symbol: string) {
        try {
            const report = await invokeCommand<ComprehensiveReport>(
                "comprehensive_predict",
                { symbol, days: 5 },
            );
            reports = { ...reports, [symbol]: report };
            const next = { ...reportErrors };
            delete next[symbol];
            reportErrors = next;
        } catch (e) {
            reportErrors = { ...reportErrors, [symbol]: readableError(e, "综合分析失败") };
        }
    }

    async function predictAll(onlySymbols?: string[]) {
        if (!items.length || busy) return;
        errorMessage = "";
        const symbols = onlySymbols ?? items.map((i) => i.symbol);
        if (!symbols.length) return;
        predicting = { done: 0, total: symbols.length, current: "" };
        for (const symbol of symbols) {
            predicting = { ...predicting!, current: symbol };
            await predictOne(symbol);
            predicting = { ...predicting!, done: predicting!.done + 1 };
        }
        predicting = null;
        showComparison = true;
    }

    async function scanRisks() {
        await predictAll();
        sortKey = "risk_level";
        sortDir = "desc";
        riskFilter = "all";
    }

    // 一键优选：报告不全时先补齐缺失的分析，再展示优选卡
    async function pickTop() {
        if (!items.length || busy) return;
        const missing = items
            .filter((i) => !reports[i.symbol])
            .map((i) => i.symbol);
        if (missing.length) {
            await predictAll(missing);
        }
        showPicks = true;
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
            .filter((r) => r.report || r.error)
            .filter((r) =>
                riskFilter === "all"
                    ? true
                    : r.report?.risk_summary?.level === riskFilter,
            );
        const dir = sortDir === "asc" ? 1 : -1;
        const val = (r: (typeof rows)[number]) => {
            if (sortKey === "risk_level") {
                return riskRank(r.report?.risk_summary?.level);
            }
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

    function riskRank(level: RiskLevel | undefined): number {
        if (level === "high") return 3;
        if (level === "medium") return 2;
        return 1;
    }

    const riskReports = $derived.by(() =>
        items
            .map((item) => reports[item.symbol])
            .filter(
                (report): report is ComprehensiveReport =>
                    !!report &&
                    (riskFilter === "all" ||
                        report.risk_summary?.level === riskFilter),
            )
            .sort(
                (a, b) =>
                    riskRank(b.risk_summary.level) - riskRank(a.risk_summary.level) ||
                    b.risk_summary.warnings.length - a.risk_summary.warnings.length,
            ),
    );

    const riskCounts = $derived.by(() => {
        const values = Object.values(reports);
        return {
            high: values.filter((report) => report.risk_summary?.level === "high").length,
            medium: values.filter((report) => report.risk_summary?.level === "medium").length,
            low: values.filter((report) => report.risk_summary?.level === "low").length,
        };
    });

    // 优选排序：仅对引擎既有输出做透明聚合排序（多因子分 → 信号强度 → 20日动量），
    // 不引入任何新方向规则/阈值。优先在"引擎看涨"的票中选；全池无看涨时退化为
    // 全池相对强弱排序并在界面明示。这是描述性相对强弱，不是上涨概率（实证结论1/8）。
    const topPicks = $derived.by(() => {
        const rs = items
            .map((it) => reports[it.symbol])
            .filter((r): r is ComprehensiveReport => !!r);
        if (!rs.length) return { picks: [] as ComprehensiveReport[], bullishOnly: false };
        const isBullish = (r: ComprehensiveReport) =>
            /买|多|涨/.test(r.direction) && !/卖|空|跌/.test(r.direction);
        const bullish = rs.filter(isBullish);
        const pool = bullish.length ? bullish : rs;
        const picks = [...pool]
            .sort(
                (a, b) =>
                    b.adaptive_score - a.adaptive_score ||
                    b.signal_strength - a.signal_strength ||
                    (b.momentum_20d ?? Number.NEGATIVE_INFINITY) -
                        (a.momentum_20d ?? Number.NEGATIVE_INFINITY),
            )
            .slice(0, TOP_PICK_COUNT);
        return { picks, bullishOnly: bullish.length > 0 };
    });

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
    <header class="page-header">
        <div>
            <h1>我的收藏与风险扫描</h1>
            <p>优先核对数据新鲜度、风险告警和校准区间，再比较相对强弱。</p>
        </div>
    </header>

    <!-- 工具条 -->
    <div class="toolbar">
        <div class="custom-select add-select">
            <span class="search-icon"><Search size={17} aria-hidden="true" /></span>
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
                            {#if pooledSymbols.has(sixDigit(stock.symbol))}
                                <span class="added">★ 已收藏</span>
                            {:else}
                                <span class="industry">{stock.industry}</span>
                            {/if}
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
                <RefreshCw size={16} aria-hidden="true" />全部刷新
            </button>
            <button
                class="btn primary"
                onclick={scanRisks}
                disabled={!items.length || busy}
                title="对收藏池全部股票生成结构化风险告警（纯本地计算）"
            >
                <ShieldAlert size={16} aria-hidden="true" />风险扫描
            </button>
            <button
                class="btn gold"
                onclick={pickTop}
                disabled={!items.length || busy}
                title="按引擎多因子分/信号强度/动量选出相对强弱靠前的几只（描述性参考，非上涨概率）"
            >
                <Trophy size={16} aria-hidden="true" />相对强弱参考
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
            <button class="dismiss" onclick={() => (errorMessage = "")} aria-label="关闭错误提示"><X size={15} aria-hidden="true" /></button>
        </div>
    {/if}

    {#if Object.keys(reports).length}
        <section class="risk-overview" aria-label="收藏池风险概览">
            <div class="risk-overview-head">
                <div>
                    <h2>风险概览</h2>
                    <span>高风险 {riskCounts.high} · 中风险 {riskCounts.medium} · 低风险 {riskCounts.low}</span>
                </div>
                <div class="risk-filters" aria-label="风险等级筛选">
                    <button class:active={riskFilter === "all"} onclick={() => riskFilter = "all"}>全部</button>
                    <button class:active={riskFilter === "high"} onclick={() => riskFilter = "high"}>高风险</button>
                    <button class:active={riskFilter === "medium"} onclick={() => riskFilter = "medium"}>中风险</button>
                    <button class:active={riskFilter === "low"} onclick={() => riskFilter = "low"}>低风险</button>
                </div>
            </div>
            {#if riskReports.length}
                <div class="risk-scan-list">
                    {#each riskReports.slice(0, 5) as report (report.symbol)}
                        <div class="risk-scan-row">
                            <button class="risk-symbol" onclick={() => goPredict(report.symbol)} title="打开完整风险报告">
                                <span>{report.symbol}</span>
                                <small>{report.name || "—"}</small>
                            </button>
                            <RiskAlertPanel summary={report.risk_summary} compact />
                        </div>
                    {/each}
                </div>
            {:else}
                <p class="risk-empty">当前筛选条件下没有风险报告。</p>
            {/if}
        </section>
    {/if}

    <!-- 主内容 -->
    {#if loading}
        <div class="loading">加载中...</div>
    {:else if items.length === 0}
        <div class="empty-card">
            <p class="empty-title">还没有收藏股票</p>
            <p class="empty-hint">
                收藏股票后，可在这里统一核对风险、数据新鲜度和相对强弱。
            </p>
            <div class="empty-actions">
                <button
                    class="btn"
                    onclick={() => onNavigate({ view: "list" })}
                >
                    <List size={16} aria-hidden="true" />去股票列表
                </button>
                <button
                    class="btn"
                    onclick={() => onNavigate({ view: "realtime" })}
                >
                    <Clock3 size={16} aria-hidden="true" />去实时行情
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
                                    {#if rowRefreshing[item.symbol]}
                                        <RotateCw size={15} class="spin" aria-hidden="true" />
                                    {:else}
                                        <RefreshCw size={15} aria-hidden="true" />
                                    {/if}
                                </button>
                                <button
                                    class="mini"
                                    title="跳转预测页生成综合报告"
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        goPredict(item.symbol);
                                    }}
                                >
                                    <Target size={15} aria-hidden="true" />
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
                                    <X size={15} aria-hidden="true" />
                                </button>
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        <!-- 一键优选：相对强弱靠前的几只（描述性参考，非上涨概率） -->
        {#if showPicks && topPicks.picks.length}
            <div class="picks-card">
                <div class="picks-head">
                    <h2><Trophy size={18} aria-hidden="true" />相对强弱参考（前 {topPicks.picks.length} 只）</h2>
                    <span class="picks-sub">
                        {topPicks.bullishOnly
                            ? "从引擎判断看涨的股票中"
                            : "⚠️ 当前引擎对全池均无看涨信号，以下仅为全池中"}按
                        多因子分 → 信号强度 → 20日动量 排序选出
                    </span>
                </div>
                <div class="picks-grid">
                    {#each topPicks.picks as r, i (r.symbol)}
                        <div class="pick-item">
                            <div class="pick-head">
                                <span class="pick-rank">#{i + 1}</span>
                                <span class="symbol">{r.symbol}</span>
                                <span class="pick-name">{r.name || "—"}</span>
                                <span class="pick-direction {directionClass(r.direction)}"
                                    >{r.direction}</span
                                >
                            </div>
                            <div class="pick-stats">
                                <span>多因子 <b>{fmt(r.adaptive_score, 0)}</b></span>
                                <span title="信号强度非方向命中概率"
                                    >信号强度 <b>{fmt(r.signal_strength, 2)}</b></span
                                >
                                <span class={updown(r.momentum_20d)}
                                    >20日动量 {fmtSigned(r.momentum_20d)}</span
                                >
                                <span
                                    >预期{r.prediction_days}日 {fmtSigned(
                                        r.expected_change_percent,
                                    )}{#if r.interval}
                                        <em class="pick-interval"
                                            >（80%区间 {fmtSigned(
                                                r.interval.lower_change_percent,
                                            )} ~ {fmtSigned(
                                                r.interval.upper_change_percent,
                                            )}）</em
                                        >{/if}</span
                                >
                                <span>风险 {r.risk_level || "—"}</span>
                            </div>
                            <div class="pick-advice">{r.current_advice}</div>
                            <div class="pick-ops">
                                <button class="mini" onclick={() => goPredict(r.symbol)}
                                    ><Target size={14} aria-hidden="true" />完整报告</button
                                >
                                <button
                                    class="mini"
                                    onclick={() => goHistory(r.symbol, r.name)}
                                    ><CalendarDays size={14} aria-hidden="true" />历史K线</button
                                >
                            </div>
                        </div>
                    {/each}
                </div>
                <p class="picks-note">
                    ⚠️ 本项目实证：技术信号对单股方向无预测力（引擎方向准确率不高于朴素基准），此优选只是把引擎信号与近期动量的<b>相对强弱</b>排序，<b>不是"最可能上涨"的概率保证</b>；请结合
                    80% 区间带、估值与自身判断使用。
                </p>
            </div>
        {/if}

        <!-- 一键预测对比表 -->
        {#if showComparison && comparisonRows.length}
            <div class="compare-card">
                <div class="compare-head">
                    <h2><ShieldAlert size={18} aria-hidden="true" />综合分析对比</h2>
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
                                    漂移中枢{sortKey ===
                                    "expected_change_percent"
                                        ? sortDir === "asc"
                                            ? " ↑"
                                            : " ↓"
                                        : ""}
                                </th>
                                <th title="近20日实现波动率校准的80%区间"
                                    >80%校准区间</th
                                >
                                <th class="sortable" onclick={() => sortBy("risk_level")}>风险{sortKey === "risk_level" ? sortDir === "asc" ? " ↑" : " ↓" : ""}</th>
                                <th>技术状态解读</th>
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
                                        <td>
                                            <span
                                                class="risk-chip level-{r.risk_summary?.level ?? 'low'}"
                                                title={r.risk_summary?.warnings?.[0]?.title ?? "未触发中高风险告警"}
                                            >
                                                {r.risk_summary?.level_label || r.risk_level || "—"}
                                                {#if r.risk_summary?.warnings?.length}
                                                    <small>{r.risk_summary.warnings.length}</small>
                                                {/if}
                                            </span>
                                        </td>
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
                                                <Target size={15} aria-hidden="true" />
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
                                                <CalendarDays size={15} aria-hidden="true" />
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
                                                disabled={busy}
                                                onclick={() =>
                                                    predictOne(
                                                        row.item.symbol,
                                                    )}
                                            >
                                                <RotateCw size={15} aria-hidden="true" />
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
        /* 明确 flex 基准宽度：容器宽度不依赖子元素百分比解析，
           避免 WebKitGTK（Tauri Linux 渲染引擎）下输入框溢出遮挡右侧按钮 */
        flex: 0 0 320px;
        min-width: 0;
    }

    .search-input {
        /* 项目未引入全局 border-box 重置（global.css 是死文件），必须显式声明，
           否则 width:100% + 内边距会溢出容器 ~42px，遮挡右侧按钮 */
        box-sizing: border-box;
        width: 100%;
        padding: 0.7rem 1.2rem;
        border: 2px solid #3b82f6;
        border-radius: 8px;
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

    .dropdown-item .added {
        margin-left: auto;
        font-size: 0.8rem;
        color: #fbbf24;
    }

    .dropdown-empty {
        padding: 0.8rem;
        color: #94a3b8;
        text-align: center;
    }

    .action-buttons {
        display: flex;
        gap: 0.6rem;
        flex-shrink: 0;
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
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
    }

    .btn:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.12);
    }

    .btn.primary {
        background: #0e7490;
        border-color: #0891b2;
    }

    .btn.primary:hover:not(:disabled) {
        background: #0891b2;
    }

    .btn.gold {
        background: rgba(251, 191, 36, 0.15);
        border-color: rgba(251, 191, 36, 0.45);
        color: #fbbf24;
    }

    .btn.gold:hover:not(:disabled) {
        background: rgba(251, 191, 36, 0.28);
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
        background: #0891b2;
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
        border-radius: 8px;
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
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
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

    /* 优选卡 */
    .picks-card {
        margin-top: 1.5rem;
        padding: 1rem 1.2rem 1.2rem;
        background: rgba(251, 191, 36, 0.05);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 0.75rem;
    }

    .picks-head {
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .picks-head h2 {
        font-size: 1.1rem;
        color: #fbbf24;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .picks-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .picks-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 0.8rem;
    }

    .pick-item {
        padding: 0.8rem 0.9rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
    }

    .pick-head {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .pick-rank {
        font-weight: 700;
        color: #fbbf24;
    }

    .pick-name {
        font-weight: 600;
        color: #f8fafc;
    }

    .pick-direction {
        margin-left: auto;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .pick-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1rem;
        font-size: 0.82rem;
        color: #cbd5e1;
        font-variant-numeric: tabular-nums;
    }

    .pick-interval {
        font-style: normal;
        color: #94a3b8;
    }

    .pick-advice {
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    .pick-ops {
        display: flex;
        gap: 0.4rem;
    }

    .picks-note {
        margin-top: 0.9rem;
        font-size: 0.78rem;
        line-height: 1.6;
        color: #b8a05e;
    }

    /* 对比卡 */
    .compare-card {
        margin-top: 1.5rem;
        padding: 1rem 1.2rem 1.2rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
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
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .compare-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .risk-overview {
        margin: 1rem 0 1.25rem;
        border-top: 1px solid #374151;
        border-bottom: 1px solid #374151;
        background: rgba(17, 21, 27, 0.55);
    }

    .risk-overview-head {
        padding: 0.8rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .risk-overview-head h2 {
        margin: 0 0 0.2rem;
        font-size: 1rem;
    }

    .risk-overview-head span {
        color: #9ca3af;
        font-size: 0.78rem;
    }

    .risk-filters {
        display: flex;
        overflow: hidden;
        border: 1px solid #4b5563;
        border-radius: 6px;
    }

    .risk-filters button {
        padding: 0.4rem 0.65rem;
        border: none;
        border-right: 1px solid #4b5563;
        background: transparent;
        color: #cbd5e1;
        cursor: pointer;
        font-size: 0.78rem;
    }

    .risk-filters button:last-child {
        border-right: none;
    }

    .risk-filters button.active {
        background: #374151;
        color: #ffffff;
    }

    .risk-scan-list {
        display: flex;
        flex-direction: column;
        border-top: 1px solid #2b3039;
    }

    .risk-empty {
        margin: 0;
        padding: 0.8rem 0;
        border-top: 1px solid #2b3039;
        color: #9ca3af;
        font-size: 0.82rem;
    }

    .risk-scan-row {
        padding: 0.65rem 0;
        display: grid;
        grid-template-columns: 150px minmax(0, 1fr);
        gap: 0.75rem;
        border-bottom: 1px solid #2b3039;
    }

    .risk-scan-row:last-child {
        border-bottom: none;
    }

    .risk-symbol {
        padding: 0.5rem;
        display: flex;
        align-items: flex-start;
        flex-direction: column;
        border: none;
        background: transparent;
        color: #67e8f9;
        cursor: pointer;
        text-align: left;
    }

    .risk-symbol small {
        margin-top: 0.15rem;
        color: #9ca3af;
    }

    .risk-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.45rem;
        border-radius: 4px;
        background: rgba(34, 197, 94, 0.12);
        color: #86efac;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .risk-chip.level-medium {
        background: rgba(245, 158, 11, 0.13);
        color: #fde68a;
    }

    .risk-chip.level-high {
        background: rgba(239, 68, 68, 0.13);
        color: #fecaca;
    }

    .risk-chip small {
        min-width: 16px;
        height: 16px;
        display: inline-grid;
        place-items: center;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
        color: inherit;
        font-size: 0.65rem;
    }

    :global(.spin) {
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
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

    /* 与应用工作台统一的视觉基线 */
    .container {
        width: 100%;
        max-width: 1320px;
        margin: 0 auto;
    }

    .search-icon {
        position: absolute;
        left: 0.75rem;
        top: 50%;
        z-index: 2;
        transform: translateY(-50%);
        color: var(--text-muted);
        pointer-events: none;
    }

    .search-input {
        min-height: 40px;
        padding: 0.55rem 0.75rem 0.55rem 2.25rem;
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        background: var(--surface-2);
        color: var(--text-primary);
    }

    .search-input:focus { border-color: var(--accent); }
    .search-input::placeholder { color: var(--text-muted); }
    .dropdown-list { background: var(--surface-2); border-color: var(--border-strong); border-radius: 6px; }
    .dropdown-item:hover { background: var(--accent-muted); }
    .dropdown-item .symbol { color: var(--accent); }
    .dropdown-item .name { color: var(--text-primary); }
    .dropdown-item .industry, .dropdown-empty { color: var(--text-muted); }

    .btn {
        min-height: 40px;
        padding: 0.55rem 0.85rem;
        border-color: var(--border-strong);
        border-radius: 6px;
        background: var(--surface-2);
        color: var(--text-primary);
        font-size: 0.84rem;
    }

    .btn.primary { background: #16899a; border-color: #16899a; }
    .btn.primary:hover:not(:disabled) { background: #20a4b7; }
    .btn.gold { background: rgba(232, 174, 74, 0.08); border-color: rgba(232, 174, 74, 0.42); color: var(--warning); }
    .btn.gold:hover:not(:disabled) { background: rgba(232, 174, 74, 0.15); }
    .progress, .empty-card, .picks-card, .compare-card { background: var(--surface-1); border-color: var(--border); border-radius: 8px; }
    .progress-text, .pick-stats, .base-rates { color: var(--text-secondary); }
    .progress-fill { background: var(--accent); }
    .table-wrap { border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); }
    .fav-table th { background: var(--surface-3); color: #d9dfe5; }
    .fav-table td { border-bottom-color: var(--border); color: var(--text-secondary); }
    .row-click:hover td { background: rgba(49, 181, 200, 0.05); }
    .fav-table .symbol { color: var(--accent); }
    .fav-table .name { color: var(--text-primary); }
    .up { color: var(--price-up); }
    .down { color: var(--price-down); }
    .mini { border-color: var(--border-strong); background: var(--surface-2); color: var(--text-secondary); }
    .pick-item { background: var(--surface-2); border-color: var(--border); }
    .pick-name, .empty-title { color: var(--text-primary); }
    .empty-hint, .compare-sub, .disclaimer { color: var(--text-secondary); }
    .risk-overview { border-color: var(--border); }
    .sortable:hover { color: var(--accent); }

    @media (max-width: 768px) {
        .custom-select {
            flex: 1 1 100%;
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
            padding: 0.55rem 0.4rem;
            font-size: 0.82rem;
            white-space: nowrap;
        }

        .empty-card {
            padding: 2rem 1rem;
        }

        .empty-actions {
            gap: 0.5rem;
        }

        .empty-actions .btn {
            padding: 0.55rem 0.65rem;
            font-size: 0.85rem;
            white-space: nowrap;
        }

        .risk-overview-head {
            align-items: flex-start;
            flex-direction: column;
        }

        .risk-filters {
            width: 100%;
        }

        .risk-filters button {
            flex: 1;
        }

        .risk-scan-row {
            grid-template-columns: 1fr;
        }
    }
</style>
