// 完整的MACD测试 - 使用足够的数据点

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
    
    let mut dif_series = Vec::new();
    let offset = 14;
    
    for i in 0..ema26_series.len() {
        let ema12_idx = offset + i;
        if ema12_idx < ema12_series.len() {
            let dif = ema12_series[ema12_idx] - ema26_series[i];
            dif_series.push(dif);
        }
    }
    
    if dif_series.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    
    let dif = *dif_series.last().unwrap();
    
    let dea = if dif_series.len() >= 9 {
        calculate_ema(&dif_series, 9)
    } else {
        dif
    };
    
    let macd_histogram = 2.0 * (dif - dea);
    
    (dif, dea, macd_histogram)
}

fn main() {
    println!("====== MACD完整测试（足够数据点） ======\n");
    
    // 生成50个递增的价格数据
    let mut prices = Vec::new();
    for i in 0..50 {
        let base = 10.0 + (i as f64) * 0.3;
        let noise = ((i as f64) * 0.7).sin() * 0.5;
        prices.push(base + noise);
    }
    
    println!("数据长度: {}", prices.len());
    println!("价格范围: {:.2} ~ {:.2}\n", prices[0], prices[prices.len()-1]);
    
    // 测试MACD计算
    let (dif, dea, hist) = calculate_macd_full(&prices);
    
    println!("====== MACD计算结果 ======");
    println!("DIF:  {:.6}", dif);
    println!("DEA:  {:.6}", dea);
    println!("HIST: {:.6}", hist);
    println!("\n验证: HIST = 2 × (DIF - DEA)");
    println!("      {:.6} = 2 × ({:.6} - {:.6})", hist, dif, dea);
    println!("      {:.6} = {:.6} {}", hist, 2.0 * (dif - dea), if (hist - 2.0 * (dif - dea)).abs() < 0.0001 { "✓" } else { "✗" });
    
    // 验证DIF序列长度
    let ema12_series = calculate_ema_series(&prices, 12);
    let ema26_series = calculate_ema_series(&prices, 26);
    
    let mut dif_series = Vec::new();
    let offset = 14;
    for i in 0..ema26_series.len() {
        let ema12_idx = offset + i;
        if ema12_idx < ema12_series.len() {
            dif_series.push(ema12_series[ema12_idx] - ema26_series[i]);
        }
    }
    
    println!("\n====== DIF序列信息 ======");
    println!("DIF序列长度: {}", dif_series.len());
    println!("是否足够计算DEA(9): {}", if dif_series.len() >= 9 { "是 ✓" } else { "否 ✗" });
    
    if dif_series.len() >= 5 {
        println!("\n最后5个DIF值:");
        let start = dif_series.len().saturating_sub(5);
        for i in start..dif_series.len() {
            println!("  DIF[{}] = {:.6}", i, dif_series[i]);
        }
    }
    
    // 手动验证DEA计算
    if dif_series.len() >= 9 {
        let manual_dea = calculate_ema(&dif_series, 9);
        println!("\n手动计算DEA(9): {:.6}", manual_dea);
        println!("函数返回DEA:    {:.6}", dea);
        println!("差异: {:.8}", (manual_dea - dea).abs());
        println!("验证: {}", if (manual_dea - dea).abs() < 0.0001 { "通过 ✓" } else { "失败 ✗" });
    }
    
    // 测试金叉死叉检测
    println!("\n====== 金叉死叉检测 ======");
    if dif_series.len() >= 2 {
        let prev_dif = dif_series[dif_series.len() - 2];
        
        // 计算前一个DEA
        let prev_dif_series = &dif_series[0..dif_series.len()-1];
        let prev_dea = if prev_dif_series.len() >= 9 {
            calculate_ema(prev_dif_series, 9)
        } else {
            prev_dif
        };
        
        println!("前一时刻: DIF={:.6}, DEA={:.6}", prev_dif, prev_dea);
        println!("当前时刻: DIF={:.6}, DEA={:.6}", dif, dea);
        
        let is_golden_cross = prev_dif <= prev_dea && dif > dea;
        let is_death_cross = prev_dif >= prev_dea && dif < dea;
        
        if is_golden_cross {
            println!("\n✓ 检测到金叉信号！（DIF从下方穿越DEA）");
        } else if is_death_cross {
            println!("\n✓ 检测到死叉信号！（DIF从上方穿越DEA）");
        } else {
            let relationship = if dif > dea { "DIF在DEA上方" } else { "DIF在DEA下方" };
            println!("\n○ 无交叉信号（{}）", relationship);
        }
    }
    
    println!("\n====== 测试完成 ======");
} 