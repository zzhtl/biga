// MACD计算详细调试测试

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

fn main() {
    println!("====== MACD计算详细调试 ======\n");
    
    let prices = vec![
        10.0, 10.5, 11.0, 10.8, 11.2, 11.5, 11.8, 11.3, 11.6, 12.0,
        12.3, 12.1, 12.5, 12.8, 12.6, 13.0, 13.2, 13.5, 13.3, 13.8,
        14.0, 14.2, 14.5, 14.3, 14.8, 15.0, 15.2, 15.5, 15.3, 15.8,
    ];
    
    println!("数据长度: {}\n", prices.len());
    
    // 步骤1: 计算EMA12
    println!("步骤1: 计算EMA12");
    let ema12 = calculate_ema(&prices, 12);
    println!("  最终EMA12 = {:.6}\n", ema12);
    
    // 步骤2: 计算EMA26  
    println!("步骤2: 计算EMA26");
    let ema26 = calculate_ema(&prices, 26);
    println!("  最终EMA26 = {:.6}\n", ema26);
    
    // 步骤3: 计算当前DIF
    println!("步骤3: 计算当前DIF");
    let current_dif = ema12 - ema26;
    println!("  当前DIF = EMA12 - EMA26 = {:.6}\n", current_dif);
    
    // 步骤4: 计算完整的DIF序列（用于计算DEA）
    println!("步骤4: 计算完整DIF序列");
    let ema12_series = calculate_ema_series(&prices, 12);
    let ema26_series = calculate_ema_series(&prices, 26);
    
    println!("  EMA12序列长度: {} (从第12个数据点开始)", ema12_series.len());
    println!("  EMA26序列长度: {} (从第26个数据点开始)", ema26_series.len());
    
    // 计算DIF序列 - 需要对齐
    let mut dif_series = Vec::new();
    let offset = 14; // ema12_series[14]对应原始数据的第26个点
    
    for i in 0..ema26_series.len() {
        let ema12_idx = offset + i;
        if ema12_idx < ema12_series.len() {
            let dif = ema12_series[ema12_idx] - ema26_series[i];
            dif_series.push(dif);
        }
    }
    
    println!("  DIF序列长度: {}", dif_series.len());
    if !dif_series.is_empty() {
        println!("  最后5个DIF值:");
        let start = dif_series.len().saturating_sub(5);
        for (idx, &val) in dif_series[start..].iter().enumerate() {
            println!("    DIF[{}] = {:.6}", start + idx, val);
        }
        println!("  最后一个DIF = {:.6}", dif_series.last().unwrap());
    }
    println!();
    
    // 步骤5: 计算DEA (DIF的9日EMA)
    println!("步骤5: 计算DEA (DIF的9日EMA)");
    let dea = if dif_series.len() >= 9 {
        calculate_ema(&dif_series, 9)
    } else {
        println!("  警告: DIF序列不足9个，使用简单平均");
        dif_series.iter().sum::<f64>() / dif_series.len() as f64
    };
    println!("  DEA = {:.6}\n", dea);
    
    // 步骤6: 计算MACD柱
    println!("步骤6: 计算MACD柱状图");
    let hist = 2.0 * (current_dif - dea);
    println!("  HIST = 2 × (DIF - DEA) = 2 × ({:.6} - {:.6}) = {:.6}\n", current_dif, dea, hist);
    
    // 验证
    println!("====== 最终结果 ======");
    println!("DIF:  {:.6}", current_dif);
    println!("DEA:  {:.6}", dea);
    println!("HIST: {:.6}", hist);
    println!("\n验证: HIST = 2 × (DIF - DEA) = {:.6} ✓", 2.0 * (current_dif - dea));
    
    // 使用另一种方法验证DIF（逐点计算）
    println!("\n====== 验证：逐点计算DIF ======");
    let mut manual_dif_series = Vec::new();
    for i in 26..=prices.len() {
        let sub_prices = &prices[0..i];
        let e12 = calculate_ema(sub_prices, 12);
        let e26 = calculate_ema(sub_prices, 26);
        manual_dif_series.push(e12 - e26);
    }
    
    println!("逐点计算DIF序列长度: {}", manual_dif_series.len());
    if !manual_dif_series.is_empty() {
        println!("最后一个DIF = {:.6}", manual_dif_series.last().unwrap());
        
        let manual_dea = if manual_dif_series.len() >= 9 {
            calculate_ema(&manual_dif_series, 9)
        } else {
            manual_dif_series.iter().sum::<f64>() / manual_dif_series.len() as f64
        };
        println!("对应的DEA = {:.6}", manual_dea);
    }
} 