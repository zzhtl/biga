<script lang="ts">
    import { onMount } from "svelte";
    import * as echarts from "echarts";
    import type { LastRealData, Prediction } from "../types";

    export let predictions: Prediction[] = [];
    export let lastRealData: LastRealData | null = null;

    let chartContainer: HTMLDivElement;
    let chart: echarts.ECharts | null = null;
    let resizeObserver: ResizeObserver | null = null;

    onMount(() => {
        chart = echarts.init(chartContainer);
        resizeObserver = new ResizeObserver(() => chart?.resize());
        resizeObserver.observe(chartContainer);
        updateChart();

        return () => {
            resizeObserver?.disconnect();
            chart?.dispose();
            chart = null;
        };
    });

    $: if (chart) updateChart();

    function updateChart() {
        if (!chart || !predictions.length) return;

        const labels = predictions.map((item) => item.target_date.slice(5));
        const center = predictions.map((item) => item.predicted_price);
        const lower80 = predictions.map(
            (item) => item.interval?.lower_price ?? item.predicted_price,
        );
        const upper80 = predictions.map(
            (item) => item.interval?.upper_price ?? item.predicted_price,
        );
        const lower95 = predictions.map(
            (item) => item.stress_interval?.lower_price ?? item.interval?.lower_price ?? item.predicted_price,
        );
        const upper95 = predictions.map(
            (item) => item.stress_interval?.upper_price ?? item.interval?.upper_price ?? item.predicted_price,
        );

        if (lastRealData) {
            labels.unshift(lastRealData.date.slice(5));
            center.unshift(lastRealData.price);
            lower80.unshift(lastRealData.price);
            upper80.unshift(lastRealData.price);
            lower95.unshift(lastRealData.price);
            upper95.unshift(lastRealData.price);
        }

        chart.setOption(
            {
                animationDuration: 280,
                backgroundColor: "transparent",
                color: ["#e5e7eb", "#22d3ee", "#f59e0b"],
                grid: { left: 58, right: 24, top: 48, bottom: 42 },
                legend: {
                    top: 8,
                    right: 16,
                    textStyle: { color: "#aeb7c5", fontSize: 11 },
                    data: ["历史漂移中枢", "80%校准区间", "95%压力边界"],
                },
                tooltip: {
                    trigger: "axis",
                    backgroundColor: "#11151b",
                    borderColor: "#3b424e",
                    textStyle: { color: "#e5e7eb" },
                    valueFormatter: (value: unknown) =>
                        typeof value === "number" ? value.toFixed(2) : String(value ?? "—"),
                },
                xAxis: {
                    type: "category",
                    boundaryGap: false,
                    data: labels,
                    axisLine: { lineStyle: { color: "#4b5563" } },
                    axisLabel: { color: "#9ca3af" },
                },
                yAxis: {
                    type: "value",
                    scale: true,
                    axisLabel: { color: "#9ca3af" },
                    splitLine: { lineStyle: { color: "rgba(148, 163, 184, 0.12)" } },
                },
                series: [
                    {
                        name: "80%下沿基线",
                        type: "line",
                        stack: "interval80",
                        data: lower80,
                        symbol: "none",
                        lineStyle: { opacity: 0 },
                        areaStyle: { opacity: 0 },
                        tooltip: { show: false },
                    },
                    {
                        name: "80%校准区间",
                        type: "line",
                        stack: "interval80",
                        data: upper80.map((value, index) => value - lower80[index]),
                        symbol: "none",
                        lineStyle: { opacity: 0 },
                        areaStyle: { color: "rgba(34, 211, 238, 0.24)" },
                    },
                    {
                        name: "95%压力边界",
                        type: "line",
                        data: lower95,
                        symbol: "none",
                        lineStyle: { color: "#f59e0b", type: "dashed", width: 1.3 },
                    },
                    {
                        name: "95%压力边界",
                        type: "line",
                        data: upper95,
                        symbol: "none",
                        lineStyle: { color: "#f59e0b", type: "dashed", width: 1.3 },
                    },
                    {
                        name: "历史漂移中枢",
                        type: "line",
                        data: center,
                        symbolSize: 6,
                        lineStyle: { color: "#e5e7eb", width: 2 },
                        itemStyle: { color: "#e5e7eb" },
                    },
                ],
            },
            { notMerge: true },
        );
    }
</script>

<div class="range-chart" bind:this={chartContainer} aria-label="预测校准区间图"></div>

<style>
    .range-chart {
        width: 100%;
        height: 340px;
        min-height: 300px;
    }

    @media (max-width: 640px) {
        .range-chart {
            height: 300px;
        }
    }
</style>
