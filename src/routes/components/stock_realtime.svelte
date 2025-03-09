<script lang="ts">
    let stocks = $state([
        { symbol: "AAPL", price: 192.34, change: +1.23, volume: "12.3M" },
        { symbol: "TSLA", price: 260.54, change: -0.78, volume: "8.9M" },
        { symbol: "NVDA", price: 467.7, change: +3.15, volume: "15.2M" },
    ]);
</script>

<div class="container">
    <h1>实时股票行情</h1>

    <div class="data-grid">
        <div class="header-row">
            <div>股票代码</div>
            <div>最新价</div>
            <div>涨跌幅</div>
            <div>成交量</div>
        </div>

        {#each stocks as stock}
            <div class="data-row">
                <div class="symbol">{stock.symbol}</div>
                <div class="price">${stock.price.toFixed(2)}</div>
                <div
                    class:positive={stock.change > 0}
                    class:negative={stock.change < 0}
                >
                    {stock.change > 0 ? "+" : ""}{stock.change.toFixed(2)}%
                </div>
                <div class="volume">{stock.volume}</div>
            </div>
        {/each}
    </div>
</div>

<style>
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
        grid-template-columns: 1fr 1fr 1fr 1fr;
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

    @media (max-width: 768px) {
        .header-row,
        .data-row {
            grid-template-columns: repeat(2, 1fr);
        }

        .header-row div:nth-child(3),
        .data-row div:nth-child(3) {
            display: none;
        }
    }
</style>
