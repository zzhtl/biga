// 测试技术指标计算
// 运行: cargo run --example test_indicators

fn calculate_ema(data: &[f64], period: usize) -> f64 {
    if data.is_empty() || period == 0 || data.len() < period {
        return 0.0;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0..period].iter().sum::<f64>() / period as f64;
    
    for i in period..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }
    
    ema
}

fn calculate_ema_series(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 || data.len() < period {
        return Vec::new();
    }
    
    let mut ema_values = Vec::new();
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0..period].iter().sum::<f64>() / period as f64;
    ema_values.push(ema);
    
    for i in period..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
        ema_values.push(ema);
    }
    
    ema_values
}

fn calculate_macd_full(prices: &[f64]) -> (f64, f64, f64) {
    if prices.len() < 26 {
        return (0.0, 0.0, 0.0);
    }
    
    let ema12_series = calculate_ema_series(prices, 12);
    let ema26_series = calculate_ema_series(prices, 26);
    
    if ema12_series.is_empty() || ema26_series.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let mut dif_values = Vec::new();
    let offset = ema26_series.len().saturating_sub(ema12_series.len());
    
    for i in 0..ema26_series.len() {
        let ema12_idx = if i >= offset { i - offset } else { continue };
        if ema12_idx < ema12_series.len() {
            let dif = ema12_series[ema12_idx] - ema26_series[i];
            dif_values.push(dif);
        }
    }
    
    if dif_values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let dif = *dif_values.last().unwrap();
    let dea = if dif_values.len() >= 9 {
        calculate_ema(&dif_values, 9)
    } else {
        dif_values.iter().sum::<f64>() / dif_values.len() as f64
    };
    
    let macd_histogram = 2.0 * (dif - dea);
    
    (dif, dea, macd_histogram)
}

fn calculate_kdj(highs: &[f64], lows: &[f64], closes: &[f64], n: usize) -> (f64, f64, f64) {
    if highs.len() < n || lows.len() < n || closes.len() < n {
        return (50.0, 50.0, 50.0);
    }
    
    let len = highs.len();
    let mut k = 50.0;
    let mut d = 50.0;
    
    for i in n..=len {
        let start = i.saturating_sub(n);
        let end = i;
        
        let highest = highs[start..end].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = lows[start..end].iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest == lowest {
            continue;
        }
        
        let close = closes[end - 1];
        let rsv = (close - lowest) / (highest - lowest) * 100.0;
        
        k = (2.0 / 3.0) * k + (1.0 / 3.0) * rsv;
        d = (2.0 / 3.0) * d + (1.0 / 3.0) * k;
    }
    
    let j = 3.0 * k - 2.0 * d;
    
    (k, d, j)
}

fn calculate_rsi(prices: &[f64]) -> f64 {
    if prices.len() < 15 {
        return 50.0;
    }
    
    let period = 14;
    let mut first_gain = 0.0;
    let mut first_loss = 0.0;
    
    for i in 1..=period {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            first_gain += change;
        } else {
            first_loss += -change;
        }
    }
    
    let mut avg_gain = first_gain / period as f64;
    let mut avg_loss = first_loss / period as f64;
    
    for i in (period + 1)..prices.len() {
        let change = prices[i] - prices[i-1];
        
        if change > 0.0 {
            avg_gain = (avg_gain * (period - 1) as f64 + change) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64) / period as f64;
        } else {
            avg_gain = (avg_gain * (period - 1) as f64) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + (-change)) / period as f64;
        }
    }
    
    if avg_loss == 0.0 {
        100.0
    } else {
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

fn main() {
    println!("====== 技术指标计算验证 ======\n");
    
    // 测试数据：20天的收盘价
    let test_prices = vec![
        10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.8, 11.3, 11.6, 12.0,
        12.3, 12.1, 12.5, 12.8, 12.6, 13.0, 13.2, 13.5, 13.3, 13.8,
        14.0, 14.2, 14.5, 14.3, 14.8, 15.0, 15.2, 15.5, 15.3, 15.8,
    ];
    
    let test_highs = vec![
        10.2, 10.7, 11.2, 11.0, 11.5, 11.8, 12.0, 11.5, 11.8, 12.2,
        12.5, 12.3, 12.7, 13.0, 12.8, 13.2, 13.4, 13.7, 13.5, 14.0,
        14.2, 14.4, 14.7, 14.5, 15.0, 15.2, 15.4, 15.7, 15.5, 16.0,
    ];
    
    let test_lows = vec![
        9.8, 10.3, 10.8, 10.5, 11.0, 11.3, 11.5, 11.0, 11.4, 11.8,
        12.0, 11.8, 12.3, 12.5, 12.3, 12.8, 13.0, 13.3, 13.0, 13.5,
        13.8, 14.0, 14.3, 14.0, 14.5, 14.8, 15.0, 15.3, 15.0, 15.5,
    ];
    
    // 测试EMA计算
    println!("1. EMA计算测试");
    println!("   数据长度: {}", test_prices.len());
    let ema12 = calculate_ema(&test_prices, 12);
    let ema26 = calculate_ema(&test_prices, 26);
    println!("   EMA12: {:.4}", ema12);
    println!("   EMA26: {:.4}", ema26);
    println!("   DIF应该约为: {:.4}\n", ema12 - ema26);
    
    // 测试MACD计算
    println!("2. MACD计算测试");
    let (dif, dea, hist) = calculate_macd_full(&test_prices);
    println!("   DIF:  {:.4}", dif);
    println!("   DEA:  {:.4}", dea);
    println!("   HIST: {:.4}", hist);
    println!("   验证: HIST应该 = 2 * (DIF - DEA) = {:.4}\n", 2.0 * (dif - dea));
    
    // 测试KDJ计算
    println!("3. KDJ计算测试");
    let (k, d, j) = calculate_kdj(&test_highs, &test_lows, &test_prices, 9);
    println!("   K值: {:.2}", k);
    println!("   D值: {:.2}", d);
    println!("   J值: {:.2}", j);
    println!("   验证: J应该 = 3K - 2D = {:.2}", 3.0 * k - 2.0 * d);
    println!("   K值范围: 0-100 {}", if k >= 0.0 && k <= 100.0 { "✓" } else { "✗" });
    println!("   D值范围: 0-100 {}", if d >= 0.0 && d <= 100.0 { "✓" } else { "✗" });
    println!();
    
    // 测试RSI计算
    println!("4. RSI计算测试");
    let rsi = calculate_rsi(&test_prices);
    println!("   RSI: {:.2}", rsi);
    println!("   RSI范围: 0-100 {}", if rsi >= 0.0 && rsi <= 100.0 { "✓" } else { "✗" });
    println!("   趋势: {}", if rsi > 50.0 { "看涨" } else if rsi < 50.0 { "看跌" } else { "中性" });
    println!();
    
    // 测试边界情况
    println!("5. 边界情况测试");
    
    // 数据不足的情况
    let short_prices = vec![10.0, 10.5, 11.0];
    let (dif_short, dea_short, hist_short) = calculate_macd_full(&short_prices);
    println!("   短数据MACD: DIF={:.4}, DEA={:.4}, HIST={:.4}", dif_short, dea_short, hist_short);
    
    // 价格不变的情况
    let flat_prices = vec![10.0; 30];
    let flat_highs = vec![10.1; 30];
    let flat_lows = vec![9.9; 30];
    let (k_flat, d_flat, j_flat) = calculate_kdj(&flat_highs, &flat_lows, &flat_prices, 9);
    println!("   价格不变KDJ: K={:.2}, D={:.2}, J={:.2}", k_flat, d_flat, j_flat);
    
    let rsi_flat = calculate_rsi(&flat_prices);
    println!("   价格不变RSI: {:.2}", rsi_flat);
    println!();
    
    println!("====== 测试完成 ======");
} 