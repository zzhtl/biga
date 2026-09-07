<script lang="ts">
    import { onMount } from "svelte";
    import { AlertTriangle, Check, ChevronDown, ChevronUp, X } from "lucide-svelte";
    import { errorMessage, recordDisciplineTrade, resolveDisciplineEvent } from "../services";
    import { discipline, refreshDiscipline } from "../stores/discipline.svelte";
    import type { DisciplineEvent, PositionRow } from "../types";
    import {
        fmtPrice,
        fmtShares,
        ruleLabel,
        todayIso,
    } from "../utils/discipline_format";

    /** 后端 validate_reason 的门槛，前端同步展示进度，但真正的拦截在 Rust 侧 */
    const MIN_REASON_CHARS = 10;

    let expandedId = $state<string | null>(null);
    /** "execute" = 录成交并标记已执行；"violate" = 不执行并留痕 */
    let mode = $state<"execute" | "violate">("execute");
    let reason = $state("");
    let sellPrice = $state("");
    let sellQuantity = $state("");
    let sellDate = $state(todayIso());
    let sellFee = $state("");
    let submitting = $state(false);
    let localError = $state("");

    const events = $derived(
        discipline.pending.filter(
            (event) =>
                event.action_required === "exit_all" ||
                event.action_required === "reduce_half",
        ),
    );

    function positionOf(event: DisciplineEvent): PositionRow | undefined {
        return discipline.board?.positions.find((row) => row.id === event.position_id);
    }

    function suggestedQuantity(event: DisciplineEvent): number | null {
        const position = positionOf(event);
        if (!position) return null;
        if (event.action_required === "exit_all") return position.quantity;
        const hit = position.verdict?.items.find((item) => item.code === event.rule_code);
        return hit?.suggested_quantity ?? null;
    }

    function open(event: DisciplineEvent, next: "execute" | "violate") {
        if (expandedId === event.id && mode === next) {
            expandedId = null;
            return;
        }
        expandedId = event.id;
        mode = next;
        reason = "";
        localError = "";
        sellPrice = event.trigger_close ? event.trigger_close.toFixed(3) : "";
        sellQuantity = String(suggestedQuantity(event) ?? "");
        sellDate = todayIso();
        sellFee = "";
    }

    async function submitExecute(event: DisciplineEvent) {
        const price = Number(sellPrice);
        const quantity = Number(sellQuantity);
        if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(quantity) || quantity <= 0) {
            localError = "请填写有效的成交价与股数";
            return;
        }
        if (!event.position_id) {
            localError = "该事件没有关联持仓，无法录入成交";
            return;
        }
        submitting = true;
        try {
            await recordDisciplineTrade({
                positionId: event.position_id,
                side: "sell",
                price,
                quantity,
                tradeDate: sellDate,
                fee: Number(sellFee) || 0,
                ruleCode: event.rule_code,
                eventId: event.id,
            });
            expandedId = null;
            await refreshDiscipline(true);
        } catch (e) {
            localError = errorMessage(e, "录入成交失败");
        } finally {
            submitting = false;
        }
    }

    async function submitViolation(event: DisciplineEvent) {
        submitting = true;
        try {
            await resolveDisciplineEvent(event.id, "violated", reason);
            expandedId = null;
            await refreshDiscipline(true);
        } catch (e) {
            localError = errorMessage(e, "记录失败");
        } finally {
            submitting = false;
        }
    }

    onMount(() => {
        if (!discipline.board) void refreshDiscipline(true);
    });
</script>

{#if events.length > 0}
    <section class="banner" aria-label="待处理的交易纪律">
        <header class="banner-head">
            <AlertTriangle size={18} aria-hidden="true" />
            <strong>{events.length} 条纪律待处理</strong>
            <span class="hint">卖错的代价是少赚，不卖错的代价是本金。</span>
        </header>

        {#each events as event (event.id)}
            {@const position = positionOf(event)}
            <article class="row">
                <div class="row-main">
                    <span class="tag">{ruleLabel(event.rule_code)}</span>
                    <span class="symbol">{position?.name ?? event.symbol}</span>
                    <span class="muted">{event.symbol}</span>
                    <span class="muted">
                        触发日 {event.event_date} · 收盘 {fmtPrice(event.trigger_close)}
                        {#if position}
                            · 成本 {fmtPrice(position.cost_price)} · 持有 {fmtShares(position.quantity)}
                        {/if}
                    </span>
                    <div class="actions">
                        <button
                            class="act execute"
                            onclick={() => open(event, "execute")}
                            disabled={submitting}
                        >
                            <Check size={14} aria-hidden="true" />已执行（录成交）
                            {#if expandedId === event.id && mode === "execute"}
                                <ChevronUp size={14} aria-hidden="true" />
                            {:else}
                                <ChevronDown size={14} aria-hidden="true" />
                            {/if}
                        </button>
                        <button
                            class="act violate"
                            onclick={() => open(event, "violate")}
                            disabled={submitting}
                        >
                            <X size={14} aria-hidden="true" />不执行（填理由）
                        </button>
                    </div>
                </div>

                {#if expandedId === event.id}
                    <div class="panel">
                        {#if mode === "execute"}
                            <div class="fields">
                                <label>
                                    成交价
                                    <input type="number" step="0.001" bind:value={sellPrice} />
                                </label>
                                <label>
                                    股数
                                    <input type="number" step="100" bind:value={sellQuantity} />
                                </label>
                                <label>
                                    成交日
                                    <input type="date" bind:value={sellDate} />
                                </label>
                                <label>
                                    手续费
                                    <input type="number" step="0.01" bind:value={sellFee} placeholder="0" />
                                </label>
                            </div>
                            <button
                                class="submit"
                                disabled={submitting}
                                onclick={() => submitExecute(event)}
                            >
                                录入并标记为已执行
                            </button>
                        {:else}
                            <label class="reason">
                                不执行的理由（至少 {MIN_REASON_CHARS} 字，会永久留档并计入复盘）
                                <textarea rows="2" bind:value={reason}></textarea>
                            </label>
                            <div class="reason-foot">
                                <span
                                    class="counter"
                                    class:short={reason.trim().length < MIN_REASON_CHARS}
                                >
                                    {reason.trim().length} / {MIN_REASON_CHARS}
                                </span>
                                <button
                                    class="submit violate"
                                    disabled={submitting || reason.trim().length < MIN_REASON_CHARS}
                                    onclick={() => submitViolation(event)}
                                >
                                    记录违纪并保留持仓
                                </button>
                            </div>
                        {/if}
                        {#if localError}
                            <p class="err">{localError}</p>
                        {/if}
                    </div>
                {/if}
            </article>
        {/each}
    </section>
{/if}

<style>
    .banner {
        border: 1px solid #ef4444;
        border-left-width: 4px;
        border-radius: 8px;
        background: rgba(239, 68, 68, 0.1);
        padding: 12px 14px;
        margin-bottom: 16px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .banner-head {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #fecaca;
        font-size: 14px;
    }
    .hint {
        color: var(--text-secondary);
        font-size: 12px;
    }
    .row {
        border-top: 1px solid rgba(239, 68, 68, 0.25);
        padding-top: 8px;
    }
    .row-main {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 13px;
    }
    .tag {
        background: rgba(239, 68, 68, 0.18);
        color: #fecaca;
        border-radius: 999px;
        padding: 2px 9px;
        font-size: 12px;
        white-space: nowrap;
    }
    .symbol {
        color: var(--text-primary);
        font-weight: 600;
    }
    .muted {
        color: var(--text-muted);
        font-size: 12px;
        font-variant-numeric: tabular-nums;
    }
    .actions {
        margin-left: auto;
        display: flex;
        gap: 8px;
    }
    .act {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 6px;
        border: 1px solid var(--border-strong);
        background: var(--surface-2);
        color: var(--text-primary);
        padding: 5px 10px;
        font-size: 12px;
        cursor: pointer;
    }
    .act:hover:not(:disabled) {
        border-color: var(--accent);
    }
    .act:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .act.violate {
        color: var(--text-secondary);
    }
    .panel {
        margin-top: 10px;
        background: var(--surface-1);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .fields {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    label {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 12px;
        color: var(--text-secondary);
    }
    input,
    textarea {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 5px;
        color: var(--text-primary);
        padding: 5px 8px;
        font-size: 13px;
        font-family: inherit;
    }
    .reason textarea {
        width: 100%;
        resize: vertical;
    }
    .reason-foot {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .counter {
        font-size: 12px;
        color: var(--success);
        font-variant-numeric: tabular-nums;
    }
    .counter.short {
        color: var(--warning);
    }
    .submit {
        margin-left: auto;
        border-radius: 6px;
        border: 1px solid var(--accent);
        background: var(--accent-muted);
        color: var(--accent-hover);
        padding: 6px 12px;
        font-size: 13px;
        cursor: pointer;
    }
    .submit.violate {
        border-color: #ef4444;
        background: rgba(239, 68, 68, 0.14);
        color: #fecaca;
    }
    .submit:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }
    .err {
        color: var(--danger);
        font-size: 12px;
        margin: 0;
    }
</style>
