<script lang="ts">
    import { onMount } from 'svelte';
    import * as echarts from 'echarts';
    import type { AiReport } from '../types/ai';
    import { aiChartOptions, volumeChartOptions } from '../utils/ai_chart';
    let { report }: { report: AiReport } = $props();
    let priceElement: HTMLDivElement;
    let volumeElement: HTMLDivElement;
    let price = $state<echarts.ECharts | null>(null);
    let volume = $state<echarts.ECharts | null>(null);
    onMount(() => {
        price = echarts.init(priceElement); volume = echarts.init(volumeElement);
        const observer = new ResizeObserver(() => { price?.resize(); volume?.resize(); });
        observer.observe(priceElement); observer.observe(volumeElement);
        return () => { observer.disconnect(); price?.dispose(); volume?.dispose(); };
    });
    $effect(() => { price?.setOption(aiChartOptions(report), { notMerge: true }); volume?.setOption(volumeChartOptions(report), { notMerge: true }); });
</script>

<div class="price-chart" bind:this={priceElement} role="img" aria-label="历史收盘与 AI 未来情景走势，详细价格见下方表格"></div>
<p>阴影表示 AI 情景估计范围，未经过概率校准。预测起点为 {report.snapshot.as_of}。</p>
<h3>分析依据 · 最近 60 日成交量</h3>
<div class="volume-chart" bind:this={volumeElement} role="img" aria-label="最近 60 日实际成交量"></div>

<style>
    .price-chart { width: 100%; height: 390px; }
    .volume-chart { width: 100%; height: 190px; }
    p { color: var(--text-muted); font-size: .8rem; }
    h3 { font-size: .9rem; font-weight: 500; }
    @media (max-width: 700px) { .price-chart { height: 320px; } }
</style>
