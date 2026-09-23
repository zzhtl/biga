<script lang="ts">
    import { Repeat } from "lucide-svelte";
    import type { CycleAnalysis, CyclePhase } from "../types";

    export let cycle: CycleAnalysis | null = null;
    let showHistory = false;

    const steps: { phase: CyclePhase; label: string }[] = [
        { phase: "box", label: "箱体震荡" },
        { phase: "markup", label: "主升浪" },
        { phase: "top", label: "高位横盘" },
        { phase: "markdown", label: "下跌通道" },
    ];

    $: badge = cycle?.tentative_base
        ? "下跌后的试探性底部"
        : cycle?.cycle_kind === "range_high"
          ? "非典型：未经暴涨"
          : null;
    // 同一阶段的几条概率常共用一条稳定性说明，去重后统一放在列表下方
    $: oddsNotes = [...new Set((cycle?.odds ?? []).map((odds) => odds.note).filter(Boolean))];

    function signed(value: number, digits = 1): string {
        return `${value > 0 ? "+" : ""}${value.toFixed(digits)}%`;
    }
</script>

{#if cycle}
    <section class="cycle-panel phase-{cycle.phase}" aria-label="周期阶段">
        <header>
            <div class="title-row">
                <Repeat size={18} aria-hidden="true" />
                <h3>周期阶段</h3>
                <span class="phase-badge">{cycle.phase_label}</span>
                {#if badge}
                    <span class="kind-badge">{badge}</span>
                {/if}
            </div>
            <span class="since">{cycle.phase_since} 起第 {cycle.days_in_phase} 个交易日</span>
        </header>

        <ol class="steps" aria-label="周期：箱体震荡 → 主升浪 → 高位横盘 → 下跌通道 → 回到箱体">
            {#each steps as step}
                <li class:active={step.phase === cycle.phase}><span>{step.label}</span></li>
            {/each}
        </ol>

        <p class="summary">{cycle.summary}</p>

        {#if cycle.key_levels.length}
            <div class="table-wrap">
                <table class="levels">
                    <thead>
                        <tr><th>关键价位</th><th class="num">价格</th><th class="num">距现价</th><th>按模型的含义</th></tr>
                    </thead>
                    <tbody>
                        {#each cycle.key_levels as level}
                            <tr>
                                <td>{level.label}</td>
                                <td class="num">{level.price.toFixed(2)}</td>
                                <td class="num" class:up={level.distance_percent > 0} class:down={level.distance_percent < 0}>
                                    {signed(level.distance_percent)}
                                </td>
                                <td class="meaning">{level.meaning}</td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {/if}

        {#if cycle.odds.length}
            <div class="odds">
                <div class="section-title">历史结局频率<small>全库样本的历史频率，不是对本股的预测</small></div>
                {#each cycle.odds as odds}
                    <div class="odds-row">
                        <span class="odds-label">{odds.label}</span>
                        <span class="bar" aria-hidden="true"><span style="width: {odds.probability * 100}%"></span></span>
                        <strong>{(odds.probability * 100).toFixed(0)}%</strong>
                        <span class="odds-n">{odds.hits}/{odds.samples} 例</span>
                    </div>
                {/each}
                {#each oddsNotes as note}
                    <p class="odds-note">{note}</p>
                {/each}
            </div>
        {/if}

        {#if cycle.facts.length}
            <div class="facts">
                {#each cycle.facts as fact}
                    <span>{fact}</span>
                {/each}
            </div>
        {/if}

        {#if cycle.history.length}
            <div class="history-block">
                <button
                    class="history-toggle"
                    type="button"
                    aria-expanded={showHistory}
                    onclick={() => (showHistory = !showHistory)}
                >
                    {showHistory ? "收起" : "展开"}本股历史回撤（最近 {cycle.history.length} 次）
                </button>
                {#if showHistory}
                    <div class="table-wrap">
                        <table class="history">
                            <thead>
                                <tr>
                                    <th>峰值</th>
                                    <th>谷底</th>
                                    <th class="num">最大回撤</th>
                                    <th class="num">主升</th>
                                    <th class="num">下跌天数</th>
                                    <th>结局</th>
                                </tr>
                            </thead>
                            <tbody>
                                {#each cycle.history as episode}
                                    <tr class:ongoing={episode.outcome === null}>
                                        <td>{episode.peak_date}<small>{episode.peak_price.toFixed(2)}</small></td>
                                        <td>{episode.trough_date}<small>{episode.trough_price.toFixed(2)}</small></td>
                                        <td class="num down">{episode.max_drawdown_percent.toFixed(1)}%</td>
                                        <td class="num">
                                            {#if episode.rally_percent !== null && episode.rally_days !== null}
                                                {signed(episode.rally_percent, 0)}<small>{episode.rally_days} 日</small>
                                            {:else}
                                                <span class="muted">非典型</span>
                                            {/if}
                                        </td>
                                        <td class="num">{episode.decline_days}</td>
                                        <td>{episode.outcome_label}</td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                {/if}
            </div>
        {/if}

        <footer>
            <span>价格口径：{cycle.price_basis}</span>
            <span>{cycle.method_note}</span>
        </footer>
    </section>
{/if}

<style>
    .cycle-panel {
        --phase-color: #31b5c8;
        margin: 1rem 0;
        border: 1px solid #374151;
        border-left: 4px solid var(--phase-color);
        border-radius: 8px;
        background: #171b22;
        overflow: hidden;
    }

    /* A 股惯例：涨红跌绿；横盘是见顶警戒，用琥珀色 */
    .phase-markup {
        --phase-color: #f15b64;
    }

    .phase-top {
        --phase-color: #e8ae4a;
    }

    .phase-markdown {
        --phase-color: #35c889;
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
        flex-wrap: wrap;
    }

    h3 {
        margin: 0;
        font-size: 1rem;
        letter-spacing: 0;
    }

    .phase-badge,
    .kind-badge {
        border-radius: 4px;
        padding: 0.15rem 0.45rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .phase-badge {
        color: #0f1216;
        background: var(--phase-color);
    }

    .kind-badge {
        color: #d1d5db;
        background: #303640;
    }

    .since {
        color: #9ca3af;
        font-size: 0.78rem;
        white-space: nowrap;
    }

    .steps {
        margin: 0;
        padding: 0.65rem 1rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem;
        list-style: none;
        border-bottom: 1px solid #2b3039;
    }

    .steps li {
        color: #4b5563;
        font-size: 0.75rem;
    }

    .steps li span {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border: 1px solid #3b424e;
        border-radius: 999px;
        color: #78838f;
    }

    .steps li:not(:last-child)::after {
        content: "→";
        margin-left: 0.35rem;
    }

    .steps li:last-child::after {
        content: "↺ 回到箱体";
        margin-left: 0.35rem;
    }

    .steps li.active span {
        border-color: var(--phase-color);
        color: #f4f6f8;
        font-weight: 700;
    }

    .summary {
        margin: 0;
        padding: 0.75rem 1rem;
        color: #e5e7eb;
        font-size: 0.85rem;
        line-height: 1.6;
        border-bottom: 1px solid #2b3039;
    }

    .table-wrap {
        overflow-x: auto;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
    }

    th,
    td {
        padding: 0.45rem 1rem;
        text-align: left;
        border-bottom: 1px solid #2b3039;
        white-space: nowrap;
    }

    th {
        color: #9ca3af;
        font-size: 0.72rem;
        font-weight: 600;
    }

    td {
        color: #d1d5db;
    }

    .num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    td.meaning {
        white-space: normal;
        color: #aeb7c5;
        min-width: 16rem;
    }

    td small {
        display: block;
        color: #78838f;
        font-size: 0.7rem;
    }

    .up {
        color: #f15b64;
    }

    .down {
        color: #35c889;
    }

    .muted {
        color: #78838f;
    }

    .odds {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #2b3039;
    }

    .section-title {
        margin-bottom: 0.45rem;
        color: #d1d5db;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .section-title small {
        margin-left: 0.5rem;
        color: #78838f;
        font-weight: 400;
    }

    .odds-row {
        display: grid;
        grid-template-columns: minmax(12rem, 2fr) minmax(4rem, 1fr) 3rem 6rem;
        align-items: center;
        gap: 0.6rem;
        padding: 0.2rem 0;
        font-size: 0.8rem;
    }

    .odds-label {
        color: #cbd5e1;
    }

    .bar {
        height: 6px;
        border-radius: 3px;
        background: #2b3039;
        overflow: hidden;
    }

    .bar span {
        display: block;
        height: 100%;
        background: var(--phase-color);
    }

    .odds-row strong {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }

    .odds-n {
        color: #78838f;
        font-size: 0.72rem;
        font-variant-numeric: tabular-nums;
    }

    .odds-note {
        margin: 0.35rem 0 0;
        color: #9ca3af;
        font-size: 0.72rem;
        line-height: 1.5;
    }

    .facts {
        padding: 0.65rem 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        border-bottom: 1px solid #2b3039;
    }

    .facts span {
        padding: 0.2rem 0.45rem;
        border: 1px solid #3b424e;
        border-radius: 4px;
        color: #aeb7c5;
        font-size: 0.72rem;
        font-variant-numeric: tabular-nums;
    }

    .history-block {
        border-bottom: 1px solid #2b3039;
    }

    .history-toggle {
        margin: 0.6rem 1rem;
        padding: 0.25rem 0.5rem;
        border: 1px solid #4b5563;
        border-radius: 4px;
        background: #252b34;
        color: #d1d5db;
        font-size: 0.72rem;
        cursor: pointer;
    }

    .history-toggle:hover {
        border-color: #6b7280;
        background: #303640;
    }

    tr.ongoing td {
        background: rgba(49, 181, 200, 0.06);
    }

    footer {
        padding: 0.65rem 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        color: #78838f;
        font-size: 0.72rem;
        line-height: 1.5;
    }

    @media (max-width: 640px) {
        header {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.35rem;
        }

        .odds-row {
            grid-template-columns: 1fr 3rem;
        }

        .odds-row .bar,
        .odds-row .odds-n {
            display: none;
        }
    }
</style>
