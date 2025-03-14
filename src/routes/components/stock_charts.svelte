<script lang="ts">
    import { onMount } from "svelte";
    import * as echarts from "echarts";

    export let data: Array<{ date: string; value: number }>;
    export let title: string;

    let chart: echarts.ECharts;
    let chartContainer: HTMLDivElement;

    onMount(() => {
        chart = echarts.init(chartContainer);
        updateChart();
        return () => chart.dispose();
    });

    $: if (chart) updateChart();

    function updateChart() {
        chart.setOption({
            title: { text: title, left: "center" },
            xAxis: {
                type: "category",
                data: data.map((d) => d.date),
            },
            yAxis: { type: "value" },
            series: [
                {
                    data: data.map((d) => d.value),
                    type: "line",
                    smooth: true,
                    areaStyle: {},
                },
            ],
            tooltip: { trigger: "axis" },
        });
        chart.resize();
    }
</script>

<div bind:this={chartContainer} class="chart-container" />

<style>
    .chart-container {
        width: 100%;
        height: 400px;
    }
</style>
