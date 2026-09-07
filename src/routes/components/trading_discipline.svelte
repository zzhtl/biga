<script lang="ts">
    import { onMount } from "svelte";
    import {
        ChevronDown,
        ChevronRight,
        RefreshCw,
        ShieldCheck,
        SlidersHorizontal,
    } from "lucide-svelte";
    import {
        checkBuyDiscipline,
        errorMessage,
        getDisciplineReview,
        openDisciplinePosition,
        saveDisciplineAccount,
    } from "../services";
    import { discipline, refreshDiscipline } from "../stores/discipline.svelte";
    import type { DisciplineReview, DisciplineRules, EntryVerdict, PositionRow } from "../types";
    import {
        actionLabel,
        categoryLabel,
        complianceText,
        constraintLabel,
        fmtMoney,
        fmtPrice,
        fmtRewardRisk,
        fmtShares,
        fmtSignedPercent,
        PLACEHOLDER,
        pnlClass,
        ruleLabel,
        severityLabel,
        stopBasisLabel,
        todayIso,
    } from "../utils/discipline_format";

    const MIN_REASON_CHARS = 10;

    type Tab = "positions" | "buy" | "review";
    let tab = $state<Tab>("positions");
    let pageError = $state("");

    // —— 账户与规则 ——
    let showRules = $state(false);
    let cashInput = $state("");
    let ruleDraft = $state<DisciplineRules | null>(null);
    let savingAccount = $state(false);

    // —— 买入检查器 ——
    let buySymbol = $state("");
    let buyPrice = $state("");
    let stopPrice = $state("");
    let targetPrice = $state("");
    let buyQuantity = $state("");
    let buyDate = $state(todayIso());
    let buyFee = $state("");
    let overrideReason = $state("");
    let verdict = $state<EntryVerdict | null>(null);
    let checking = $state(false);
    let opening = $state(false);
    let checkSequence = 0;

    // —— 复盘 ——
    let review = $state<DisciplineReview | null>(null);
    let reviewLoading = $state(false);
    let expandedRow = $state<string | null>(null);
    let expandedPosition = $state<string | null>(null);

    const account = $derived(discipline.board?.account ?? null);
    const positions = $derived(discipline.board?.positions ?? []);
    const blockedItems = $derived(
        verdict?.items.filter((item) => item.action === "blocked") ?? [],
    );
    const canOpen = $derived(
        verdict !== null &&
            (verdict.allowed || overrideReason.trim().length >= MIN_REASON_CHARS),
    );

    onMount(async () => {
        // 显式初始化而不是用 $effect 回写：$effect 里既读又写同一个状态
        // （读 cashInput 判空、又写 cashInput）容易变成自触发循环，且类型检查看不出来。
        await refreshDiscipline();
        if (discipline.board) cashInput = String(discipline.board.account.cash);
    });

    function startEditRules() {
        if (!account) return;
        ruleDraft = { ...account.rules };
        showRules = true;
    }

    async function persistAccount() {
        const cash = Number(cashInput);
        if (!Number.isFinite(cash) || cash < 0) {
            pageError = "可用现金必须是非负数字";
            return;
        }
        savingAccount = true;
        try {
            await saveDisciplineAccount(cash, ruleDraft ?? undefined);
            ruleDraft = null;
            showRules = false;
            await refreshDiscipline(true);
            pageError = "";
        } catch (e) {
            pageError = errorMessage(e, "保存失败");
        } finally {
            savingAccount = false;
        }
    }

    /** 输入变化即预演准入。竞态序号保证慢请求不覆盖快请求。 */
    async function runCheck() {
        const price = Number(buyPrice);
        if (!buySymbol.trim() || !Number.isFinite(price) || price <= 0) {
            verdict = null;
            return;
        }
        const current = ++checkSequence;
        checking = true;
        try {
            const next = await checkBuyDiscipline({
                symbol: buySymbol.trim(),
                buyPrice: price,
                stopPrice: Number(stopPrice) || null,
                targetPrice: Number(targetPrice) || null,
                quantity: Number(buyQuantity) || null,
            });
            if (current !== checkSequence) return;
            verdict = next;
            pageError = "";
        } catch (e) {
            if (current !== checkSequence) return;
            verdict = null;
            pageError = errorMessage(e, "准入核算失败");
        } finally {
            if (current === checkSequence) checking = false;
        }
    }

    async function submitOpen() {
        const price = Number(buyPrice);
        const stop = Number(stopPrice);
        const target = Number(targetPrice);
        const quantity = Number(buyQuantity) || verdict?.sizing.max_shares || 0;
        if (quantity <= 0) {
            pageError = "买入股数必须大于 0";
            return;
        }
        opening = true;
        try {
            await openDisciplinePosition({
                symbol: buySymbol.trim(),
                price,
                quantity,
                stopPrice: stop,
                targetPrice: target,
                tradeDate: buyDate,
                fee: Number(buyFee) || 0,
                overrideReason: overrideReason.trim() || undefined,
            });
            buySymbol = "";
            buyPrice = "";
            stopPrice = "";
            targetPrice = "";
            buyQuantity = "";
            buyFee = "";
            overrideReason = "";
            verdict = null;
            pageError = "";
            tab = "positions";
            await refreshDiscipline(true);
        } catch (e) {
            pageError = errorMessage(e, "建仓失败");
        } finally {
            opening = false;
        }
    }

    async function loadReview() {
        reviewLoading = true;
        try {
            review = await getDisciplineReview();
            pageError = "";
        } catch (e) {
            pageError = errorMessage(e, "复盘加载失败");
        } finally {
            reviewLoading = false;
        }
    }

    function selectTab(next: Tab) {
        tab = next;
        // 每次切回都重新回放：录完成交后旧的复盘数字就过期了，
        // 而过期的差额比没有差额更糟——它会让人以为自己守纪了。
        if (next === "review") void loadReview();
    }

    function verdictOf(row: PositionRow) {
        return row.verdict;
    }

    /** 只挑出 DisciplineRules 里的数值字段，数组与枚举字段单列处理 */
    type NumericRuleKey = {
        [K in keyof DisciplineRules]: DisciplineRules[K] extends number ? K : never;
    }[keyof DisciplineRules];

    function setRule(key: NumericRuleKey, raw: string) {
        if (!ruleDraft) return;
        ruleDraft = { ...ruleDraft, [key]: Number(raw) };
    }

    /** 规则表单字段。scale_out_tiers 与 max_entry_risk_level 单列处理。 */
    const RULE_FIELDS: { key: NumericRuleKey; label: string; step: number }[] = [
        { key: "fixed_stop_pct", label: "固定止损 %", step: 0.5 },
        { key: "atr_mult", label: "ATR 倍数", step: 0.1 },
        { key: "atr_period", label: "ATR 周期", step: 1 },
        { key: "support_buffer_pct", label: "支撑缓冲 %", step: 0.5 },
        { key: "max_stop_pct", label: "止损宽度上限 %", step: 0.5 },
        { key: "trail_arm_pct", label: "移动止盈启动 %", step: 1 },
        { key: "trail_pct", label: "回撤触发 %", step: 1 },
        { key: "scale_out_fraction", label: "分批减仓比例", step: 0.05 },
        { key: "time_stop_bars", label: "时间止损（交易日）", step: 1 },
        { key: "time_stop_min_gain_pct", label: "时间止损最低浮盈 %", step: 0.5 },
        { key: "max_risk_pct", label: "单笔风险 %", step: 0.25 },
        { key: "min_rr", label: "最低盈亏比", step: 0.25 },
        { key: "max_single_pct", label: "单票上限 %", step: 1 },
        { key: "max_total_pct", label: "总仓上限 %", step: 1 },
        { key: "max_holdings", label: "持股只数上限", step: 1 },
        { key: "cooldown_bars", label: "冷静期（交易日）", step: 1 },
        { key: "breaker_losses", label: "熔断连亏笔数", step: 1 },
        { key: "breaker_bars", label: "熔断停手（交易日）", step: 1 },
        { key: "stale_days", label: "行情陈旧阈值（日）", step: 1 },
    ];
</script>

<div class="page">
    <header class="page-header">
        <h1><ShieldCheck size={20} aria-hidden="true" />交易纪律</h1>
        <button class="secondary-button" onclick={() => refreshDiscipline()} disabled={discipline.loading}>
            <RefreshCw size={16} class={discipline.loading ? "spin" : ""} aria-hidden="true" />
            重新扫描
        </button>
    </header>

    {#if pageError || discipline.error}
        <div class="status-panel error">{pageError || discipline.error}</div>
    {/if}

    <!-- 账户 -->
    <section class="account">
        <div class="account-grid">
            <label class="cash">
                可用现金（元）
                <input type="number" step="100" bind:value={cashInput} />
            </label>
            <div class="stat">
                <span>总资产</span>
                <strong>{fmtMoney(account?.total_equity)}</strong>
                <em>现金 + 持仓市值，现算</em>
            </div>
            <div class="stat">
                <span>持仓市值</span>
                <strong>{fmtMoney(account?.total_market_value)}</strong>
                <em>
                    {account
                        ? `${account.open_position_count} / ${account.rules.max_holdings} 只`
                        : PLACEHOLDER}
                </em>
            </div>
            <div class="stat">
                <span>单笔最大亏损</span>
                <strong>
                    {account
                        ? fmtMoney((account.total_equity * account.rules.max_risk_pct) / 100)
                        : PLACEHOLDER}
                </strong>
                <em>总资产 × {account?.rules.max_risk_pct ?? PLACEHOLDER}%</em>
            </div>
            <div class="account-actions">
                <button class="secondary-button" onclick={startEditRules}>
                    <SlidersHorizontal size={15} aria-hidden="true" />规则
                </button>
                <button class="primary-button" onclick={persistAccount} disabled={savingAccount}>
                    保存
                </button>
            </div>
        </div>

        {#if showRules && ruleDraft}
            <div class="rules">
                <p class="rules-note">
                    这些数字是行业约定俗成的经验值，<strong>不是从本项目数据优化出来的</strong>。
                    本系统不提供参数寻优——在几十笔样本上寻优等于过拟合。
                </p>
                <div class="rules-grid">
                    {#each RULE_FIELDS as field (field.key)}
                        <label>
                            {field.label}
                            <input
                                type="number"
                                step={field.step}
                                value={ruleDraft[field.key]}
                                oninput={(e) => setRule(field.key, e.currentTarget.value)}
                            />
                        </label>
                    {/each}
                    <label>
                        分批止盈第一档 %
                        <input
                            type="number"
                            step="1"
                            value={ruleDraft.scale_out_tiers[0]}
                            oninput={(e) => {
                                if (ruleDraft)
                                    ruleDraft.scale_out_tiers = [
                                        Number(e.currentTarget.value),
                                        ruleDraft.scale_out_tiers[1],
                                    ];
                            }}
                        />
                    </label>
                    <label>
                        分批止盈第二档 %
                        <input
                            type="number"
                            step="1"
                            value={ruleDraft.scale_out_tiers[1]}
                            oninput={(e) => {
                                if (ruleDraft)
                                    ruleDraft.scale_out_tiers = [
                                        ruleDraft.scale_out_tiers[0],
                                        Number(e.currentTarget.value),
                                    ];
                            }}
                        />
                    </label>
                    <label>
                        买入准入风险上限
                        <select bind:value={ruleDraft.max_entry_risk_level}>
                            <option value="low">低（最严）</option>
                            <option value="medium">中</option>
                            <option value="high">高（不拦）</option>
                        </select>
                    </label>
                </div>
            </div>
        {/if}
    </section>

    <div class="tabs">
        <button class:active={tab === "positions"} onclick={() => selectTab("positions")}>
            持仓看板
        </button>
        <button class:active={tab === "buy"} onclick={() => selectTab("buy")}>买入检查</button>
        <button class:active={tab === "review"} onclick={() => selectTab("review")}>纪律复盘</button>
    </div>

    {#if tab === "positions"}
        <div class="table-frame">
            <table>
                <thead>
                    <tr>
                        <th>代码 / 名称</th>
                        <th>成本</th>
                        <th>现价</th>
                        <th>浮动盈亏</th>
                        <th>生效止损</th>
                        <th>距止损</th>
                        <th>持仓最高 / 回撤</th>
                        <th>持有</th>
                        <th>裁决</th>
                    </tr>
                </thead>
                <tbody>
                    {#each positions as row (row.id)}
                        {@const v = verdictOf(row)}
                        <tr class:clickable={!!v} onclick={() => (expandedPosition = expandedPosition === row.id ? null : row.id)}>
                            <td>
                                <div class="cell-main">
                                    {#if v}
                                        {#if expandedPosition === row.id}
                                            <ChevronDown size={14} aria-hidden="true" />
                                        {:else}
                                            <ChevronRight size={14} aria-hidden="true" />
                                        {/if}
                                    {/if}
                                    <strong>{row.name}</strong>
                                    <span class="muted">{row.symbol}</span>
                                </div>
                            </td>
                            <td class="num">{fmtPrice(row.cost_price)}</td>
                            <td class="num">{fmtPrice(row.last_close)}</td>
                            <td class="num {pnlClass(v?.metrics.unrealized_pnl ?? null)}">
                                {fmtMoney(v?.metrics.unrealized_pnl)}
                                <em>{fmtSignedPercent(v?.metrics.unrealized_pnl_percent)}</em>
                            </td>
                            <td class="num">
                                {fmtPrice(row.stop_price)}
                                <em>{stopBasisLabel(row.stop_basis)}</em>
                            </td>
                            <td class="num">{fmtSignedPercent(v?.metrics.distance_to_stop_percent)}</td>
                            <td class="num">
                                {fmtPrice(row.highest_price)}
                                <em>{fmtSignedPercent(
                                    v?.metrics.drawdown_from_high_percent === null ||
                                    v?.metrics.drawdown_from_high_percent === undefined
                                        ? null
                                        : -v.metrics.drawdown_from_high_percent,
                                )}</em>
                            </td>
                            <td class="num">{v ? `${v.metrics.holding_trading_days} 日` : PLACEHOLDER}</td>
                            <td>
                                {#if row.unavailable_reason}
                                    <span class="verdict warn">数据不足</span>
                                {:else if v}
                                    <span class="verdict {v.action}">{actionLabel(v.action)}</span>
                                {/if}
                            </td>
                        </tr>
                        {#if expandedPosition === row.id}
                            <tr class="detail-row">
                                <td colspan="9">
                                    {#if row.unavailable_reason}
                                        <p class="muted">{row.unavailable_reason}</p>
                                    {:else if v}
                                        {#if v.items.length === 0}
                                            <p class="muted">当前未触发任何纪律条款。</p>
                                        {/if}
                                        {#each v.items as item (item.code)}
                                            <article class="item sev-{item.severity}">
                                                <header>
                                                    <span class="chip">{categoryLabel(item.category)}</span>
                                                    <strong>{item.title}</strong>
                                                    <span class="pill {item.action}">
                                                        {actionLabel(item.action)} · {severityLabel(item.severity)}
                                                    </span>
                                                </header>
                                                <p>{item.detail}</p>
                                                <div class="evidence">
                                                    {#each item.evidence as line, index (index)}
                                                        <span>{line}</span>
                                                    {/each}
                                                </div>
                                            </article>
                                        {/each}
                                    {/if}
                                </td>
                            </tr>
                        {/if}
                    {:else}
                        <tr>
                            <td colspan="9" class="table-state">
                                还没有持仓。到「买入检查」录入第一笔——先填止损价和目标价，纪律才能开始工作。
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {:else if tab === "buy"}
        <section class="buy">
            <div class="buy-form">
                <label>代码<input bind:value={buySymbol} oninput={runCheck} placeholder="600519" /></label>
                <label>买入价<input type="number" step="0.001" bind:value={buyPrice} oninput={runCheck} /></label>
                <label>止损价<input type="number" step="0.001" bind:value={stopPrice} oninput={runCheck} /></label>
                <label>目标价<input type="number" step="0.001" bind:value={targetPrice} oninput={runCheck} /></label>
                <label>股数<input type="number" step="100" bind:value={buyQuantity} oninput={runCheck} placeholder="留空 = 取最大" /></label>
                <label>成交日<input type="date" bind:value={buyDate} /></label>
                <label>手续费<input type="number" step="0.01" bind:value={buyFee} placeholder="0" /></label>
            </div>

            {#if checking}
                <p class="muted">核算中…</p>
            {/if}

            {#if verdict}
                <div class="verdict-box" class:blocked={!verdict.allowed}>
                    <header>
                        <strong>
                            {verdict.allowed ? "未发现纪律冲突" : `被 ${blockedItems.length} 条纪律拦下`}
                        </strong>
                        <span class="muted">盈亏比 {fmtRewardRisk(verdict.reward_risk_ratio)}</span>
                    </header>
                    <p class="disclaimer">{verdict.disclaimer}</p>

                    <div class="sizing">
                        <div><span>最大可买</span><strong>{fmtShares(verdict.sizing.max_shares)}</strong></div>
                        <div><span>对应金额</span><strong>{fmtMoney(verdict.sizing.max_amount)}</strong></div>
                        <div><span>每股风险</span><strong>{fmtPrice(verdict.sizing.risk_per_share)}</strong></div>
                        <div><span>风险预算</span><strong>{fmtMoney(verdict.sizing.risk_budget)}</strong></div>
                        <div><span>卡在哪</span><strong>{constraintLabel(verdict.sizing.binding_constraint)}</strong></div>
                        <div><span>交易单位</span><strong>{verdict.sizing.min_lot} 股起 / {verdict.sizing.lot_step} 股步长</strong></div>
                    </div>
                    <div class="evidence">
                        {#each verdict.sizing.evidence as line, index (index)}
                            <span>{line}</span>
                        {/each}
                    </div>

                    {#each verdict.items as item (item.code)}
                        <article class="item sev-{item.severity}">
                            <header>
                                <span class="chip">{categoryLabel(item.category)}</span>
                                <strong>{item.title}</strong>
                                <span class="pill {item.action}">{actionLabel(item.action)}</span>
                            </header>
                            <p>{item.detail}</p>
                            <div class="evidence">
                                {#each item.evidence as line, index (index)}
                                    <span>{line}</span>
                                {/each}
                            </div>
                        </article>
                    {/each}

                    {#if !verdict.allowed}
                        <label class="reason">
                            仍要买入的理由（至少 {MIN_REASON_CHARS} 字，会永久留档并计入复盘）
                            <textarea rows="2" bind:value={overrideReason}></textarea>
                        </label>
                    {/if}

                    <button class="primary-button" disabled={!canOpen || opening} onclick={submitOpen}>
                        {verdict.allowed ? "建仓并记录成交" : "越权建仓并留痕"}
                    </button>
                </div>
            {/if}
        </section>
    {:else}
        <section class="review">
            {#if reviewLoading}
                <p class="muted">回放中…</p>
            {:else if !review}
                <!-- 没有兜底分支的话，加载失败时整段会静默渲染成空白 -->
                <div class="table-state">
                    复盘数据尚未载入。
                    <button class="secondary-button" onclick={loadReview}>重新回放</button>
                </div>
            {:else}
                <div class="review-summary">
                    <div><span>已平仓</span><strong>{review.closed_count} 笔</strong></div>
                    <div><span>可回放</span><strong>{review.replayed_count} 笔</strong></div>
                    <div><span>无法回放</span><strong>{review.unresolved_count} 笔</strong><em>跌停/停牌，已剔除</em></div>
                    <div><span>守纪率</span><strong>{complianceText(review.compliance_rate)}</strong><em>已执行 {review.complied_count} / 违纪 {review.violated_count}</em></div>
                    <div><span>实际盈亏</span><strong class={pnlClass(review.actual_pnl_total)}>{fmtMoney(review.actual_pnl_total)}</strong></div>
                    <div><span>严格守纪盈亏</span><strong class={pnlClass(review.disciplined_pnl_total)}>{fmtMoney(review.disciplined_pnl_total)}</strong></div>
                    <div class="diff"><span>差额</span><strong class={pnlClass(review.difference_total)}>{fmtMoney(review.difference_total)}</strong><em>这就是「人性」的价格</em></div>
                </div>
                <p class="disclaimer">{review.disclaimer}</p>
                {#if !review.fee_recorded}
                    <p class="muted">未记录任何手续费，两侧均按 0 计。</p>
                {/if}

                {#if review.by_rule.length > 0}
                    <div class="rule-rank">
                        {#each review.by_rule as stat (stat.rule_code)}
                            <div>
                                <span>{ruleLabel(stat.rule_code)}</span>
                                <span class="muted">{stat.violated_count} 次</span>
                                <strong class={pnlClass(stat.difference_total)}>{fmtMoney(stat.difference_total)}</strong>
                            </div>
                        {/each}
                    </div>
                {/if}

                <div class="table-frame">
                    <table>
                        <thead>
                            <tr>
                                <th>代码 / 名称</th>
                                <th>建仓 → 清仓</th>
                                <th>成本</th>
                                <th>实际盈亏</th>
                                <th>守纪盈亏</th>
                                <th>差额</th>
                                <th>当时填的理由</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each review.rows as row (row.position_id)}
                                <tr class="clickable" onclick={() => (expandedRow = expandedRow === row.position_id ? null : row.position_id)}>
                                    <td><strong>{row.name}</strong> <span class="muted">{row.symbol}</span></td>
                                    <td class="num">{row.open_date} → {row.close_date}</td>
                                    <td class="num">{fmtPrice(row.cost_price)}</td>
                                    <td class="num {pnlClass(row.outcome.actual_pnl)}">{fmtMoney(row.outcome.actual_pnl)}</td>
                                    <td class="num">
                                        {#if row.outcome.unresolved}
                                            <span class="muted">无法回放</span>
                                        {:else}
                                            <span class={pnlClass(row.outcome.disciplined_pnl)}>{fmtMoney(row.outcome.disciplined_pnl)}</span>
                                        {/if}
                                    </td>
                                    <td class="num {pnlClass(row.outcome.difference)}">
                                        {row.outcome.unresolved ? PLACEHOLDER : fmtMoney(row.outcome.difference)}
                                    </td>
                                    <td class="reason-cell">
                                        {#each row.violations as violation (violation.id)}
                                            <span>「{violation.reason}」</span>
                                        {:else}
                                            <span class="muted">{row.outcome.same_as_actual ? "纪律未要求更早离场" : "无违纪记录"}</span>
                                        {/each}
                                    </td>
                                </tr>
                                {#if expandedRow === row.position_id}
                                    <tr class="detail-row">
                                        <td colspan="7">
                                            {#if row.outcome.unresolved_reason}
                                                <p class="muted">{row.outcome.unresolved_reason}</p>
                                            {/if}
                                            {#each row.outcome.fills as fill, index (index)}
                                                <p class="fill">
                                                    {ruleLabel(fill.rule_code)}：{fill.trigger_date} 收盘触发 →
                                                    {fill.fill_date} 开盘 {fmtPrice(fill.fill_price)} 成交
                                                    {fmtShares(fill.quantity)}
                                                    {#if fill.deferred_bars > 0}
                                                        （因{fill.defer_reason}顺延 {fill.deferred_bars} 根）
                                                    {/if}
                                                </p>
                                            {/each}
                                            {#if row.outcome.stop_path.length > 0}
                                                <p class="muted">
                                                    止损轨迹：
                                                    {#each row.outcome.stop_path as step, index (index)}
                                                        {index > 0 ? " → " : ""}{fmtPrice(step.stop_price)}
                                                    {/each}
                                                </p>
                                            {/if}
                                        </td>
                                    </tr>
                                {/if}
                            {:else}
                                <tr><td colspan="7" class="table-state">还没有已平仓的交易可供复盘。</td></tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            {/if}
        </section>
    {/if}
</div>

<style>
    .page { display: flex; flex-direction: column; gap: 14px; }
    h1 { display: flex; align-items: center; gap: 8px; font-size: 18px; margin: 0; }
    .account { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
    .account-grid { display: flex; gap: 18px; align-items: flex-end; flex-wrap: wrap; }
    .stat { display: flex; flex-direction: column; gap: 2px; }
    .stat span { font-size: 12px; color: var(--text-secondary); }
    .stat strong { font-size: 16px; font-variant-numeric: tabular-nums; }
    .stat em, .review-summary em { display: block; font-size: 11px; color: var(--text-muted); font-style: normal; }
    .account-actions { margin-left: auto; display: flex; gap: 8px; }
    .rules { margin-top: 12px; border-top: 1px solid var(--border); padding-top: 12px; }
    .rules-note { font-size: 12px; color: var(--text-secondary); margin: 0 0 10px; }
    .rules-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px 12px; }
    label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--text-secondary); }
    input, textarea, select { background: var(--surface-2); border: 1px solid var(--border); border-radius: 5px; color: var(--text-primary); padding: 5px 8px; font-size: 13px; font-family: inherit; }
    .tabs { display: flex; gap: 6px; }
    .tabs button { background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); padding: 6px 14px; font-size: 13px; cursor: pointer; }
    .tabs button.active { background: var(--accent-muted); color: var(--accent-hover); border-color: var(--accent); }
    .num { font-variant-numeric: tabular-nums; text-align: right; }
    .num em { display: block; font-size: 11px; color: var(--text-muted); font-style: normal; }
    .muted { color: var(--text-muted); font-size: 12px; }
    .cell-main { display: flex; align-items: center; gap: 6px; }
    tr.clickable { cursor: pointer; }
    .up { color: var(--price-up); }
    .down { color: var(--price-down); }
    .verdict { border-radius: 999px; padding: 2px 9px; font-size: 12px; white-space: nowrap; }
    .verdict.must_sell { background: rgba(239, 68, 68, 0.16); color: #fecaca; }
    .verdict.must_reduce { background: rgba(245, 158, 11, 0.16); color: #fde68a; }
    .verdict.warn, .verdict.blocked { background: rgba(34, 197, 94, 0.13); color: #bbf7d0; }
    .detail-row td { background: var(--surface-1); }
    .item { border-left: 3px solid var(--border-strong); padding: 6px 10px; margin-bottom: 8px; }
    .item.sev-high { border-left-color: #ef4444; }
    .item.sev-medium { border-left-color: #f59e0b; }
    .item.sev-low { border-left-color: #22c55e; }
    .item header { display: flex; align-items: center; gap: 8px; }
    .item p { margin: 4px 0; font-size: 13px; color: var(--text-secondary); }
    .chip { background: var(--surface-3); color: var(--text-secondary); border-radius: 4px; padding: 1px 7px; font-size: 11px; }
    .pill { margin-left: auto; border-radius: 999px; padding: 1px 9px; font-size: 11px; background: var(--surface-3); color: var(--text-secondary); }
    .pill.must_sell { background: rgba(239, 68, 68, 0.16); color: #fecaca; }
    .evidence { display: flex; flex-wrap: wrap; gap: 5px; }
    .evidence span { border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; font-size: 11px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
    .buy { display: flex; flex-direction: column; gap: 12px; }
    .buy-form { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
    .verdict-box { background: var(--surface-1); border: 1px solid var(--success); border-radius: 8px; padding: 12px 14px; display: flex; flex-direction: column; gap: 10px; }
    .verdict-box.blocked { border-color: #ef4444; }
    .verdict-box header { display: flex; align-items: baseline; gap: 12px; }
    .disclaimer { font-size: 12px; color: var(--text-muted); margin: 0; line-height: 1.6; }
    .sizing { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 8px; }
    .sizing span { display: block; font-size: 11px; color: var(--text-secondary); }
    .sizing strong { font-variant-numeric: tabular-nums; }
    .reason textarea { width: 100%; resize: vertical; }
    .review { display: flex; flex-direction: column; gap: 12px; }
    .review-summary { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
    .review-summary span { display: block; font-size: 12px; color: var(--text-secondary); }
    .review-summary strong { font-size: 17px; font-variant-numeric: tabular-nums; }
    .review-summary .diff strong { font-size: 20px; }
    .rule-rank { display: flex; flex-direction: column; gap: 4px; }
    .rule-rank > div { display: flex; gap: 12px; align-items: baseline; font-size: 13px; }
    .rule-rank strong { margin-left: auto; font-variant-numeric: tabular-nums; }
    .reason-cell span { display: block; font-size: 12px; color: var(--text-secondary); }
    .fill { font-size: 12px; color: var(--text-secondary); margin: 2px 0; }
</style>
