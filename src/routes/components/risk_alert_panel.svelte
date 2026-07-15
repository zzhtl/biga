<script lang="ts">
    import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-svelte";
    import type { RiskCategory, RiskLevel, RiskSummary } from "../types";

    export let summary: RiskSummary | null = null;
    export let compact = false;
    let expanded = false;

    const categoryLabels: Record<RiskCategory, string> = {
        data: "数据",
        uncertainty: "不确定性",
        volatility: "波动",
        trend: "趋势",
        signal: "信号",
        liquidity: "量价",
        model: "模型",
    };

    const severityLabels: Record<RiskLevel, string> = {
        low: "低",
        medium: "中",
        high: "高",
    };

    $: warnings = summary?.warnings ?? [];
    $: metrics = summary?.metrics ?? null;
    $: collapsed = compact && !expanded;
</script>

{#if summary}
    <section class:compact={collapsed} class="risk-panel level-{summary.level}" aria-label="风险警示">
        <header>
            <div class="title-row">
                {#if summary.level === "low"}
                    <CheckCircle2 size={20} aria-hidden="true" />
                {:else if summary.level === "high"}
                    <ShieldAlert size={20} aria-hidden="true" />
                {:else}
                    <AlertTriangle size={20} aria-hidden="true" />
                {/if}
                <h3>风险警示</h3>
                <span class="level-badge">{summary.level_label || "低风险"}</span>
            </div>
            <div class="header-actions">
                <span class="count">{warnings.length} 项事实告警</span>
                {#if compact}
                    <button class="detail-toggle" type="button" aria-expanded={expanded} onclick={() => (expanded = !expanded)}>
                        {expanded ? "收起" : "完整明细"}
                    </button>
                {/if}
            </div>
        </header>

        {#if !collapsed && metrics}
            <div class="metrics">
                <div>
                    <span>日波动</span>
                    <strong>{metrics.daily_volatility_percent.toFixed(2)}%</strong>
                </div>
                <div>
                    <span>波动百分位</span>
                    <strong>{metrics.volatility_percentile.toFixed(0)}%</strong>
                </div>
                <div>
                    <span>80%区间下沿</span>
                    <strong class="downside">
                        {metrics.interval_80_lower_percent == null
                            ? "—"
                            : `${metrics.interval_80_lower_percent > 0 ? "+" : ""}${metrics.interval_80_lower_percent.toFixed(2)}%`}
                    </strong>
                </div>
                <div>
                    <span>80%区间宽度</span>
                    <strong>
                        {metrics.interval_80_width_percent == null
                            ? "—"
                            : `${metrics.interval_80_width_percent.toFixed(2)}%`}
                    </strong>
                </div>
                <div>
                    <span>95%压力下沿</span>
                    <strong class="downside">
                        {metrics.stress_95_lower_percent == null
                            ? "—"
                            : `${metrics.stress_95_lower_percent > 0 ? "+" : ""}${metrics.stress_95_lower_percent.toFixed(2)}%`}
                    </strong>
                </div>
                <div>
                    <span>ATR / 现价</span>
                    <strong>{metrics.atr_percent == null ? "—" : `${metrics.atr_percent.toFixed(2)}%`}</strong>
                </div>
                <div>
                    <span>最近支撑距离</span>
                    <strong>{metrics.support_distance_percent == null ? "未识别" : `${metrics.support_distance_percent.toFixed(2)}%`}</strong>
                </div>
                <div>
                    <span>最近压力距离</span>
                    <strong>{metrics.resistance_distance_percent == null ? "未识别" : `${metrics.resistance_distance_percent.toFixed(2)}%`}</strong>
                </div>
                <div>
                    <span>数据新鲜度</span>
                    <strong class:stale={metrics.data_staleness_days != null && metrics.data_staleness_days > 4}>
                        {metrics.data_staleness_days == null
                            ? "未评估"
                            : metrics.data_staleness_days === 0
                              ? "最新"
                              : `${metrics.data_staleness_days} 个自然日`}
                    </strong>
                </div>
                <div>
                    <span>历史样本</span>
                    <strong>{metrics.history_samples} 根K线</strong>
                </div>
            </div>
            <p class="grading-note">总等级优先依据 80% 主区间与已触发的趋势、波动、量价等事实；95%压力区间用于补充低概率尾部情景。</p>
        {/if}

        {#if warnings.length}
            <div class="warning-list">
                {#each collapsed ? warnings.slice(0, 2) : warnings as warning (warning.code)}
                    <article class="warning-row severity-{warning.severity}">
                        <div class="warning-head">
                            <span class="category">{categoryLabels[warning.category]}</span>
                            <strong>{warning.title}</strong>
                            <span class="severity severity-{warning.severity}">{severityLabels[warning.severity]}等级</span>
                        </div>
                        {#if !collapsed}
                            <p>{warning.detail}</p>
                            <div class="evidence">
                                {#each warning.evidence as item}
                                    <span>{item}</span>
                                {/each}
                            </div>
                        {:else if warning.evidence[0]}
                            <span class="compact-evidence">{warning.evidence[0]}</span>
                        {/if}
                    </article>
                {/each}
                {#if collapsed && warnings.length > 2}
                    <span class="more">另有 {warnings.length - 2} 项</span>
                {/if}
            </div>
        {:else}
            <p class="clear-state">当前未触发中高等级事实告警，仍需结合校准区间判断不确定性。</p>
        {/if}
    </section>
{/if}

<style>
    .risk-panel {
        margin: 1rem 0;
        border: 1px solid #374151;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        background: #171b22;
        overflow: hidden;
    }

    .risk-panel.level-medium {
        border-left-color: #f59e0b;
    }

    .risk-panel.level-high {
        border-left-color: #ef4444;
    }

    header {
        min-height: 48px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid #2b3039;
    }

    .title-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        min-width: 0;
    }

    h3 {
        margin: 0;
        font-size: 1rem;
        letter-spacing: 0;
    }

    .level-badge,
    .category {
        border-radius: 4px;
        padding: 0.15rem 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .level-badge {
        color: #bbf7d0;
        background: rgba(34, 197, 94, 0.13);
    }

    .level-medium .level-badge {
        color: #fde68a;
        background: rgba(245, 158, 11, 0.14);
    }

    .level-high .level-badge {
        color: #fecaca;
        background: rgba(239, 68, 68, 0.14);
    }

    .count,
    .compact-evidence,
    .more {
        color: #9ca3af;
        font-size: 0.78rem;
    }

    .header-actions {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-shrink: 0;
    }

    .detail-toggle {
        padding: 0.25rem 0.5rem;
        border: 1px solid #4b5563;
        border-radius: 4px;
        background: #252b34;
        color: #d1d5db;
        font-size: 0.72rem;
        cursor: pointer;
    }

    .detail-toggle:hover {
        border-color: #6b7280;
        background: #303640;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        border-bottom: 1px solid #2b3039;
    }

    .metrics > div {
        padding: 0.75rem 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        border-right: 1px solid #2b3039;
        border-bottom: 1px solid #2b3039;
    }

    .metrics > div:nth-child(4n),
    .metrics > div:last-child {
        border-right: none;
    }

    .metrics > div:nth-last-child(-n + 2) {
        border-bottom: none;
    }

    .metrics span {
        color: #9ca3af;
        font-size: 0.72rem;
    }

    .metrics strong {
        font-size: 0.95rem;
        font-variant-numeric: tabular-nums;
    }

    .metrics .downside {
        color: #fca5a5;
    }

    .metrics .stale {
        color: #fbbf24;
    }

    .grading-note {
        margin: 0;
        padding: 0.65rem 1rem;
        border-bottom: 1px solid #2b3039;
        color: #9ca3af;
        font-size: 0.75rem;
    }

    .warning-list {
        display: flex;
        flex-direction: column;
    }

    .warning-row {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #2b3039;
    }

    .warning-row:last-child {
        border-bottom: none;
    }

    .warning-row.severity-high {
        background: rgba(239, 68, 68, 0.05);
    }

    .warning-head {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }

    .warning-head strong {
        font-size: 0.88rem;
    }

    .severity {
        margin-left: auto;
        border-radius: 999px;
        padding: 0.12rem 0.42rem;
        color: #bbf7d0;
        background: rgba(34, 197, 94, 0.12);
        font-size: 0.68rem;
        white-space: nowrap;
    }

    .severity-medium {
        color: #fde68a;
        background: rgba(245, 158, 11, 0.14);
    }

    .severity-high {
        color: #fecaca;
        background: rgba(239, 68, 68, 0.14);
    }

    .category {
        color: #d1d5db;
        background: #303640;
    }

    p {
        margin: 0.4rem 0 0;
        color: #cbd5e1;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .evidence {
        margin-top: 0.45rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }

    .evidence span {
        padding: 0.2rem 0.45rem;
        border: 1px solid #3b424e;
        border-radius: 4px;
        color: #aeb7c5;
        font-size: 0.72rem;
        font-variant-numeric: tabular-nums;
    }

    .clear-state {
        margin: 0;
        padding: 0.8rem 1rem;
    }

    .compact {
        margin: 0;
    }

    .compact header {
        min-height: 38px;
        padding: 0.5rem 0.7rem;
    }

    .compact .warning-row {
        padding: 0.5rem 0.7rem;
    }

    .compact .warning-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
    }

    .compact .more {
        padding: 0.45rem 0.7rem;
    }

    @media (max-width: 640px) {
        .metrics {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .metrics > div:nth-child(odd) {
            border-right: 1px solid #2b3039;
        }

        .metrics > div:nth-child(even) {
            border-right: none;
        }

        .compact .warning-row {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.2rem;
        }
    }
</style>
