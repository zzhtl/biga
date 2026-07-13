<script lang="ts">
    import { onMount } from "svelte";
    import { BarChart, CandlestickChart, LineChart } from "echarts/charts";
    import {
        DataZoomComponent,
        GridComponent,
        TooltipComponent,
    } from "echarts/components";
    import { init, use, type ECharts, type EChartsCoreOption } from "echarts/core";
    import { CanvasRenderer } from "echarts/renderers";
    import type { HistoricalData } from "../types";
    import { calculateKdj, calculateMacd } from "../utils/historical_indicators";

    use([
        BarChart,
        CandlestickChart,
        LineChart,
        DataZoomComponent,
        GridComponent,
        TooltipComponent,
        CanvasRenderer,
    ]);

    let { data }: { data: HistoricalData[] } = $props();
    let container: HTMLDivElement;
    let chart: ECharts | null = null;

    function option(): EChartsCoreOption {
        const sorted = [...data].sort((left, right) => left.date.localeCompare(right.date));
        const dates = sorted.map((item) => item.date.slice(0, 10));
        const closes = sorted.map((item) => item.close);
        const macd = calculateMacd(closes);
        const kdj = calculateKdj(
            sorted.map((item) => item.high),
            sorted.map((item) => item.low),
            closes,
        );

        return {
            animation: false,
            backgroundColor: "transparent",
            tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
            grid: [
                { left: 58, right: 26, top: 24, height: "43%" },
                { left: 58, right: 26, top: "52%", height: "11%" },
                { left: 58, right: 26, top: "68%", height: "10%" },
                { left: 58, right: 26, top: "83%", height: "10%" },
            ],
            xAxis: [
                { type: "category", data: dates, boundaryGap: true, axisLabel: { show: false } },
                { type: "category", data: dates, gridIndex: 1, axisLabel: { show: false } },
                { type: "category", data: dates, gridIndex: 2, axisLabel: { show: false } },
                { type: "category", data: dates, gridIndex: 3, axisLabel: { color: "#78838f", fontSize: 10 } },
            ],
            yAxis: [
                { scale: true, axisLabel: { color: "#78838f" }, splitLine: { lineStyle: { color: "#292e35" } } },
                { scale: true, gridIndex: 1, axisLabel: { show: false }, splitLine: { show: false } },
                { scale: true, gridIndex: 2, axisLabel: { color: "#78838f", fontSize: 9 }, splitLine: { show: false } },
                { min: 0, max: 100, gridIndex: 3, axisLabel: { color: "#78838f", fontSize: 9 }, splitLine: { show: false } },
            ],
            dataZoom: [
                { type: "inside", xAxisIndex: [0, 1, 2, 3], start: 0, end: 100 },
                { type: "slider", xAxisIndex: [0, 1, 2, 3], bottom: 0, height: 18, borderColor: "#343a42", fillerColor: "rgba(49,181,200,0.16)", textStyle: { color: "#78838f" } },
            ],
            series: [
                {
                    name: "K线",
                    type: "candlestick",
                    data: sorted.map((item) => [item.open, item.close, item.low, item.high]),
                    itemStyle: { color: "#f15b64", borderColor: "#f15b64", color0: "#35c889", borderColor0: "#35c889" },
                },
                {
                    name: "成交量",
                    type: "bar",
                    xAxisIndex: 1,
                    yAxisIndex: 1,
                    data: sorted.map((item) => ({ value: item.volume, itemStyle: { color: item.change >= 0 ? "#f15b64" : "#35c889" } })),
                },
                { name: "MACD", type: "line", xAxisIndex: 2, yAxisIndex: 2, data: macd.macd, symbol: "none", lineStyle: { color: "#31b5c8", width: 1 } },
                { name: "Signal", type: "line", xAxisIndex: 2, yAxisIndex: 2, data: macd.signal, symbol: "none", lineStyle: { color: "#e8ae4a", width: 1 } },
                { name: "Histogram", type: "bar", xAxisIndex: 2, yAxisIndex: 2, data: macd.histogram.map((value) => ({ value, itemStyle: { color: value >= 0 ? "#f15b64" : "#35c889" } })) },
                { name: "K", type: "line", xAxisIndex: 3, yAxisIndex: 3, data: kdj.k, symbol: "none", lineStyle: { color: "#e8ae4a", width: 1 } },
                { name: "D", type: "line", xAxisIndex: 3, yAxisIndex: 3, data: kdj.d, symbol: "none", lineStyle: { color: "#31b5c8", width: 1 } },
                { name: "J", type: "line", xAxisIndex: 3, yAxisIndex: 3, data: kdj.j, symbol: "none", lineStyle: { color: "#ef6b73", width: 1 } },
            ],
        };
    }

    onMount(() => {
        chart = init(container);
        chart.setOption(option(), true);
        const observer = new ResizeObserver(() => chart?.resize());
        observer.observe(container);
        return () => {
            observer.disconnect();
            chart?.dispose();
            chart = null;
        };
    });

    $effect(() => {
        if (chart && data.length > 0) chart.setOption(option(), true);
    });
</script>

<div bind:this={container} class="chart" role="img" aria-label="历史K线、成交量、MACD与KDJ图表"></div>

<style>
    .chart { width: 100%; height: 520px; }
    @media (max-height: 800px) { .chart { height: 440px; } }
</style>
