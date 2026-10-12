<script lang="ts">
    import { onMount, tick } from 'svelte';
    import { listen } from '@tauri-apps/api/event';
    import { BrainCircuit, Play, Square, RefreshCw, Settings, History } from 'lucide-svelte';
    import AiChart from './ai_chart.svelte';
    import { listAiConnections, startAiPrediction, cancelAiPrediction, getAiPrediction, listAiPredictions } from '../services/ai';
    import { errorMessage, invokeCommand } from '../services/core';
    import { getStockInfos } from '../services/stock';
    import { acceptAiJob, canonicalAiSymbol, isActiveJob, reportText, sourceLabels, type AiConnectionConfig, type AiJob, type AiReport } from '../types/ai';
    import type { NavTarget, StockInfo } from '../types';

    let { navSymbol = null, onNavigate }: { navSymbol?: string | null; onNavigate: (target: NavTarget) => void } = $props();
    let symbol = $state('');
    let days = $state(5);
    let connectionId = $state('');
    let connections = $state<AiConnectionConfig[]>([]);
    let stocks = $state<StockInfo[]>([]);
    let jobs = $state<AiJob[]>([]);
    let job = $state<AiJob | null>(null);
    let report = $state<AiReport | null>(null);
    let error = $state('');
    let notice = $state('');
    let loading = $state(false);
    let ready = $state(false);
    let disposed = false;
    let selection = 0;
    let currentSymbol = $derived(canonicalAiSymbol(symbol));
    let active = $derived(isActiveJob(job));
    let company = $derived(stocks.find(s => canonicalAiSymbol(s.symbol) === currentSymbol)?.name ?? '');

    function applyJob(value: AiJob) {
        job = value;
        jobs = jobs.map(item => item.id === value.id ? { ...value, report_json: null } : item);
        if (value.report_json) {
            try { report = JSON.parse(value.report_json); }
            catch { error = '报告数据无法读取，请重新生成'; report = null; }
        } else { report = null; }
        if (value.error) error = value.error;
    }

    async function loadHistory(code: string) {
        const version = ++selection;
        job = null; report = null; error = ''; notice = '';
        try {
            const rows = await listAiPredictions(code || null);
            if (disposed || version !== selection) return;
            jobs = rows;
            const running = rows.find(row => isActiveJob(row) && row.symbol === code);
            if (running) {
                const value = await getAiPrediction(running.id);
                if (!disposed && version === selection) applyJob(value);
            }
        } catch (e) { if (!disposed && version === selection) error = errorMessage(e, '无法读取报告历史'); }
    }

    async function openReport(id: string) {
        const version = ++selection;
        error = ''; notice = '';
        try {
            const result = await getAiPrediction(id);
            if (disposed || version !== selection) return;
            if (canonicalAiSymbol(symbol) !== result.symbol) {
                symbol = result.symbol;
                // 让股票选择的副作用先完成，再显示所选报告。
                await tick();
            }
            const reportSelection = ++selection;
            applyJob(result);
            const rows = await listAiPredictions(result.symbol);
            if (!disposed && selection === reportSelection) jobs = rows;
        } catch (e) { error = errorMessage(e, '无法打开报告'); }
    }

    async function generate(regenerate = false) {
        if (!currentSymbol || !connectionId || !Number.isInteger(days) || days < 1 || days > 30) { error = '请选择有效股票、AI 连接和 1—30 日预测周期'; return; }
        const code = currentSymbol;
        const version = selection;
        loading = true; error = ''; notice = ''; report = null;
        try {
            localStorage.setItem('biga-ai-connection', connectionId);
            const result = await startAiPrediction(code, connectionId, days, regenerate);
            if (disposed || version !== selection) return;
            applyJob(result);
            // 事件可能早于启动命令返回，补读一次任务状态。
            const latest = await getAiPrediction(result.id);
            if (!disposed && acceptAiJob(job?.id ?? null, symbol, latest)) applyJob(latest);
            const rows = await listAiPredictions(code);
            if (!disposed && version === selection) jobs = rows;
        } catch (e) { if (!disposed && version === selection) error = errorMessage(e, '无法启动 AI 预测'); }
        finally { loading = false; }
    }

    async function cancel() {
        if (!job) return;
        try { await cancelAiPrediction(job.id); notice = '已请求取消；已发出的请求仍计入用量'; }
        catch (e) { error = errorMessage(e, '取消失败'); }
    }

    async function refreshData() {
        const code = currentSymbol;
        if (!code) return;
        loading = true; error = ''; notice = '';
        try {
            const result = await invokeCommand<{ requests: number; cache_hits: number; warnings: string[] }>('refresh_historical_data', { symbol: code });
            if (currentSymbol === code) notice = `行情准备完成：实际请求 ${result.requests} 次，缓存命中 ${result.cache_hits} 次。${result.warnings.join('；')}`;
        } catch (e) { if (currentSymbol === code) error = errorMessage(e, '行情刷新失败'); }
        finally { loading = false; }
    }

    $effect(() => { if (navSymbol) symbol = navSymbol; });
    $effect(() => { const code = currentSymbol; if (ready) void loadHistory(code); });

    onMount(() => {
        let unlisten: (() => void) | undefined;
        let polling = false;
        const timer = setInterval(async () => {
            if (!job || !isActiveJob(job) || polling) return;
            polling = true;
            const id = job.id;
            try {
                const result = await getAiPrediction(id);
                if (!disposed && acceptAiJob(job?.id ?? null, symbol, result)) applyJob(result);
            } catch (e) { if (!disposed) error = errorMessage(e, '任务状态读取失败'); }
            finally { polling = false; }
        }, 2000);
        void (async () => {
            try {
                const loaded = await listAiConnections();
                if (disposed) return;
                connections = loaded;
                const remembered = localStorage.getItem('biga-ai-connection');
                connectionId = loaded.find(c => c.id === remembered)?.id ?? loaded[0]?.id ?? '';
                ready = true;
                stocks = await getStockInfos();
                const unsubscribe = await listen<AiJob>('ai_prediction_progress', ({ payload }) => {
                    if (!disposed && acceptAiJob(job?.id ?? null, symbol, payload)) applyJob(payload);
                });
                if (disposed) unsubscribe(); else unlisten = unsubscribe;
            } catch (e) { if (!disposed) error = errorMessage(e, '初始化 AI 预测失败'); }
        })();
        return () => { disposed = true; clearInterval(timer); unlisten?.(); };
    });

    const fmt = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(2);
    const directionLabels = { up: '偏多', down: '偏空', sideways: '震荡' };
    const statusLabels: Record<string, string> = { preparing: '准备数据', running: '分析中', completed: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断' };
</script>

<div class="ai-page">
    <header class="page-header">
        <div><div class="eyebrow"><BrainCircuit size={15} /> AI RESEARCH</div><h1>AI 股票预测</h1><p>让 Claude / Codex 基于真实行情，形成有依据的走势分析。</p></div>
        <button class="settings" onclick={() => onNavigate({ view: 'settings' })}><Settings size={16} />配置 AI</button>
    </header>

    <section class="controls" aria-label="预测参数">
        <div class="inputs">
            <label>股票代码<input list="ai-stock-options" bind:value={symbol} placeholder="输入六位代码" disabled={loading} aria-label="股票代码" /></label>
            <datalist id="ai-stock-options">{#each stocks as stock}<option value={stock.symbol}>{stock.name}</option>{/each}</datalist>
            <label>AI 连接<select bind:value={connectionId} disabled={loading}><option value="" disabled>请选择连接</option>{#each connections as c}<option value={c.id}>{c.name} · {c.provider} / {c.transport === 'ssh' ? '远程' : '本地'}</option>{/each}</select></label>
            <label>预测交易日<input type="number" bind:value={days} min="1" max="30" disabled={loading} /></label>
        </div>
        <div class="actions">
            <button class="primary" disabled={loading || active || !connectionId || !currentSymbol} onclick={() => generate()}><Play size={16} />{loading ? '处理中…' : '生成 AI 预测'}</button>
            {#if report}<button disabled={loading || active} onclick={() => generate(true)}>重新生成</button>{/if}
            <button disabled={loading || active || !currentSymbol} onclick={refreshData}><RefreshCw size={15} />刷新行情</button>
            {#if active}<button class="cancel" onclick={cancel}><Square size={14} />取消任务</button>{/if}
            <span class="quota">缓存优先 · 每次预测最多 3 次行情请求</span>
        </div>
    </section>

    {#if ready && !connections.length}<div class="empty"><BrainCircuit size={32} /><h2>连接你的 AI，开始第一份分析</h2><p>在设置中添加本机或 SSH 服务器上的 Claude / Codex。查看页面不会消耗额度。</p><button class="primary" onclick={() => onNavigate({ view: 'settings' })}>添加 AI 连接</button></div>{/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if notice}<p class="notice" role="status">{notice}</p>{/if}
    {#if job}<div class="progress" role="status"><span class:pulse={active}>{job.stage}</span><span>行情请求 {job.requests} / 3 · 缓存命中 {job.cache_hits}</span></div>{/if}

    {#if report}
        <section class="report">
            <div class="report-heading"><div><h2>{company} <span>{report.snapshot.symbol}</span></h2><p>{report.connection_name} · {report.model} · {new Date(report.generated_at).toLocaleString()}</p></div><span class="direction" class:up={report.assessment.direction === 'up'} class:down={report.assessment.direction === 'down'}>{directionLabels[report.assessment.direction]}</span></div>
            <p class="summary">{reportText(report.assessment.summary)}</p>
            <div class="metrics"><div><span>数据截止</span><strong>{report.snapshot.as_of}</strong></div><div><span>基准收盘价</span><strong>¥ {fmt(report.snapshot.bars.at(-1)?.close)}</strong></div><div><span>期末基准情景</span><strong>¥ {fmt(report.assessment.forecast.at(-1)?.expected)}</strong></div><div><span>期末情景范围</span><strong>{fmt(report.assessment.forecast.at(-1)?.lower)} — {fmt(report.assessment.forecast.at(-1)?.upper)}</strong></div></div>
            {#if report.snapshot.warnings.length}<div class="warnings">{#each report.snapshot.warnings as warning}<p>{warning}</p>{/each}</div>{/if}
            <AiChart {report} />
        </section>
        <div class="report-grid">
            <section class="panel"><h2>关键分析依据</h2>{#each report.assessment.factors as factor}<div class="factor"><p>{reportText(factor.text)}</p><div class="tags">{#each factor.sources as source}<span>{sourceLabels[source] ?? source}</span>{/each}</div></div>{/each}</section>
            <section class="panel"><h2>三种可能情景</h2><div class="scenario"><h3>乐观情景</h3><p>{reportText(report.assessment.bullish)}</p></div><div class="scenario"><h3>基准情景</h3><p>{reportText(report.assessment.base)}</p></div><div class="scenario"><h3>悲观情景</h3><p>{reportText(report.assessment.bearish)}</p></div></section>
            <section class="panel"><h2>主要风险</h2><ul>{#each report.assessment.risks as risk}<li>{reportText(risk)}</li>{/each}</ul></section>
            <section class="panel"><h2>判断失效条件</h2><ul>{#each report.assessment.invalidation as condition}<li>{reportText(condition)}</li>{/each}</ul></section>
        </div>
        <details class="panel"><summary>逐日预测与数据来源</summary><div class="table-wrap"><table><thead><tr><th>交易日</th><th>情景下界</th><th>基准价格</th><th>情景上界</th></tr></thead><tbody>{#each report.assessment.forecast as point, i}<tr><td>{report.dates[i]}</td><td>{fmt(point.lower)}</td><td>{fmt(point.expected)}</td><td>{fmt(point.upper)}</td></tr>{/each}</tbody></table></div><ul>{#each report.snapshot.sources as source}<li>{sourceLabels[source.resource]}：{source.fetched_at ? new Date(source.fetched_at * 1000).toLocaleString() : '未取得查询时间'} · {source.stale ? '过期或不可用' : '缓存有效'}</li>{/each}</ul><p class="muted">AI 请求用量（CLI 实际返回）：{JSON.stringify(report.usage)}</p></details>
    {:else if ready && connections.length && !active && !error}
        <div class="empty"><BrainCircuit size={32} /><h2>从一只股票开始</h2><p>选择股票后生成报告，或打开已有报告。文字结论、行情依据与未来情景将在这里展示。</p></div>
    {/if}

    <section class="panel history"><h2><History size={17} />报告历史</h2>{#if jobs.length}<div class="history-list">{#each jobs as item}<button onclick={() => openReport(item.id)}><strong>{item.symbol}</strong><span>{new Date(item.created_at).toLocaleString()}</span><span>{statusLabels[item.status]}</span></button>{/each}</div>{:else}<p class="muted">暂无报告。报告会自动保存，重新打开不调用行情或 AI。</p>{/if}</section>
</div>

<style>
    .ai-page { max-width: 1400px; margin: 0 auto; }
    .eyebrow { display: flex; align-items: center; gap: .4rem; color: var(--accent); font-size: .7rem; letter-spacing: .13em; margin-bottom: .6rem; }
    .page-header p, .report-heading p { color: var(--text-secondary); font-size: .84rem; }
    button { display: inline-flex; justify-content: center; align-items: center; gap: .45rem; padding: .65rem .9rem; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--surface-2); color: var(--text-primary); cursor: pointer; }
    button:hover { border-color: var(--accent); }
    button:disabled { opacity: .45; cursor: default; }
    .primary { color: #09252a; background: var(--accent); border-color: var(--accent); font-weight: 650; }
    .controls, .report, .panel { border: 1px solid var(--border); border-radius: 10px; padding: 1.3rem; background: var(--surface-1); margin-bottom: 1rem; min-width: 0; }
    .inputs { display: grid; grid-template-columns: 1fr 1.6fr .7fr; gap: 1rem; }
    label { display: flex; flex-direction: column; gap: .5rem; color: var(--text-secondary); font-size: .8rem; }
    input, select { width: 100%; min-width: 0; }
    .actions { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-top: 1rem; }
    .quota { color: var(--text-muted); font-size: .76rem; margin-left: auto; }
    .progress { display: flex; justify-content: space-between; gap: 1rem; padding: .85rem 1rem; background: var(--accent-muted); border-radius: 6px; margin-bottom: 1rem; font-size: .8rem; }
    .error { color: var(--danger); background: rgba(239,107,115,.08); padding: 1rem; border-radius: 6px; }
    .notice, .warnings { color: var(--warning); font-size: .84rem; }
    .warnings { border-left: 2px solid var(--warning); padding-left: 1rem; }
    .warnings p { margin: .3rem 0; }
    .empty { text-align: center; padding: 3rem 1rem; color: var(--text-secondary); border: 1px dashed var(--border); border-radius: 10px; margin-bottom: 1rem; }
    .empty h2 { color: var(--text-primary); font-size: 1.1rem; }
    h2 { font-size: 1rem; margin: 0 0 .8rem; display: flex; gap: .5rem; align-items: center; }
    .report-heading { display: flex; justify-content: space-between; gap: 1rem; align-items: center; }
    .report-heading h2 { font-size: 1.2rem; }
    .report-heading h2 span { color: var(--text-muted); }
    .direction { border: 1px solid var(--border); border-radius: 20px; padding: .45rem 1.1rem; white-space: nowrap; }
    .direction.up { color: var(--price-up); border-color: var(--price-up); }
    .direction.down { color: var(--price-down); border-color: var(--price-down); }
    .summary { line-height: 1.9; white-space: pre-wrap; font-size: .98rem; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; border-block: 1px solid var(--border); padding: 1rem 0; margin: 1.3rem 0; }
    .metrics span, .metrics strong { display: block; }
    .metrics span { font-size: .75rem; color: var(--text-muted); margin-bottom: .4rem; }
    .metrics strong { font-size: 1rem; font-variant-numeric: tabular-nums; }
    .report-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .report-grid .panel { margin: 0 0 1rem; }
    .factor + .factor, .scenario + .scenario { border-top: 1px solid var(--border); padding-top: .7rem; margin-top: .8rem; }
    .factor p, .scenario p, li { line-height: 1.8; color: var(--text-secondary); font-size: .85rem; white-space: pre-wrap; overflow-wrap: anywhere; }
    .tags { display: flex; gap: .4rem; flex-wrap: wrap; }
    .tags span { background: var(--accent-muted); color: var(--accent); padding: .15rem .45rem; border-radius: 4px; font-size: .7rem; }
    h3 { font-size: .85rem; color: var(--accent); margin: .5rem 0; }
    .muted { color: var(--text-muted); font-size: .8rem; overflow-wrap: anywhere; }
    .table-wrap { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: .85rem; margin-top: 1rem; }
    th, td { text-align: right; padding: .7rem; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
    th:first-child, td:first-child { text-align: left; }
    summary { cursor: pointer; }
    .history-list { display: grid; gap: .45rem; }
    .history-list button { justify-content: space-between; font-size: .8rem; }
    .history-list span { color: var(--text-muted); }
    .pulse { animation: pulse 1.5s infinite; }
    @keyframes pulse { 50% { opacity: .5; } }
    @media (prefers-reduced-motion: reduce) { .pulse { animation: none; } }
    @media (max-width: 900px) { .inputs { grid-template-columns: 1fr; } .report-grid { grid-template-columns: 1fr; } .metrics { grid-template-columns: 1fr 1fr; } .quota { margin-left: 0; } .progress { flex-wrap: wrap; } }
</style>
