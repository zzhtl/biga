<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";

    // 定义类型
    interface RealtimeData {
        symbol: string;
        name: string;
        date: Date;
        close: number;
        volume: number;
        amount: number;
        amplitude: number;
        turnover_rate: number;
        change: number;
        change_percent: number;
    }

    let stocks = $state<RealtimeData[]>([]);
    let loading = $state(true);
    let error = $state<string | null>(null);
    // 添加日期格式化函数
    const formatDate = (date: Date) => {
        return new Intl.DateTimeFormat("zh-CN").format(date);
    };

    // 添加成交量格式化函数
    const formatVolume = (volume: number) => {
        if (volume >= 1_000_000) return `${(volume / 1_000_000).toFixed(1)}M`;
        if (volume >= 1_000) return `${(volume / 1_000).toFixed(1)}K`;
        return volume.toString();
    };

    // 组件挂载时获取数据
    onMount(async () => {
        try {
            const response = await invoke<RealtimeData[]>("get_realtime_data");
            stocks = response.map((item) => ({
                ...item,
                date: new Date(item.date), // 确保日期转换为Date对象
            }));
            error = null;
        } catch (err) {
            console.error("获取数据失败:", err);
            error = "无法获取实时数据，请稍后重试";
        } finally {
            loading = false;
        }
    });
</script>

<div class="container">
    <h1>实时股票行情</h1>

    <div class="data-grid">
        <div class="header-row">
            <div>股票代码</div>
            <div>名称</div>
            <div>日期</div>
            <div>最新价</div>
            <div>成交量</div>
            <div>成交额</div>
            <div>振幅</div>
            <div>换手率</div>
            <div>涨跌额</div>
            <div>涨跌幅</div>
        </div>

        {#each stocks as stock}
            <div class="data-row">
                <div class="symbol">{stock.symbol}</div>
                <div class="name">{stock.name}</div>
                <div class="date">{formatDate(stock.date)}</div>
                <div class="close">{stock.close}</div>
                <div class="volume">{stock.volume}手</div>
                <div class="amount">{stock.amount}</div>
                <div class="amplitude">{stock.amplitude}&</div>
                <div class="turnover_rate">{stock.turnover_rate}%</div>
                <div
                    class:negative={stock.change > 0}
                    class:positive={stock.change < 0}
                >
                    {stock.change > 0 ? "+" : ""}{stock.change}
                </div>
                <div
                    class:negative={stock.change_percent > 0}
                    class:positive={stock.change_percent < 0}
                >
                    {stock.change_percent > 0 ? "+" : ""}{stock.change_percent}%
                </div>
            </div>
        {/each}
    </div>
</div>

<style>
    /* 添加加载和错误状态样式 */
    .loading,
    .error {
        padding: 2rem;
        text-align: center;
        color: #666;
    }

    .error {
        color: #ef4444;
    }
    .header-row,
    .data-row {
        grid-template-columns: repeat(7, 1fr);
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }

    h1 {
        font-size: 1.8rem;
        margin-bottom: 2rem;
        color: var(--text-primary);
    }

    .data-grid {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        overflow: hidden;
    }

    .header-row,
    .data-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr;
        gap: 1rem;
        padding: 1rem;
        align-items: center;
    }

    .header-row {
        background: rgba(255, 255, 255, 0.1);
        font-weight: 600;
    }

    .data-row {
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        transition: background 0.2s ease;
    }

    .data-row:hover {
        background: rgba(255, 255, 255, 0.02);
    }

    .positive {
        color: #10b981;
    }

    .negative {
        color: #ef4444;
    }

    .symbol {
        font-weight: 500;
    }

    /* 移动端适配 */
    @media (max-width: 768px) {
        .header-row,
        .data-row {
            grid-template-columns: repeat(3, 1fr);
        }

        /* 隐藏日期和名称列 */
        .header-row div:nth-child(2),
        .header-row div:nth-child(3),
        .data-row div:nth-child(2),
        .data-row div:nth-child(3) {
            display: none;
        }
    }
</style>
