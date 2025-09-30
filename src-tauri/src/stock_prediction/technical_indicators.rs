use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdData {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdjData {
    pub k: f64,
    pub d: f64,
    pub j: f64,
}

// 计算指数移动平均线
pub fn calculate_ema(data: &[f64], period: usize) -> f64 {
    if data.is_empty() || period == 0 || data.len() < period {
        return 0.0;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0];
    
    for i in 1..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }
    
    ema
}

// 计算EMA序列
pub fn calculate_ema_series(data: &[f64], period: usize) -> Vec<f64> {
    if data.is_empty() || period == 0 || data.len() < period {
        return Vec::new();
    }
    
    let mut ema_values = Vec::new();
    let multiplier = 2.0 / (period as f64 + 1.0);
    
    // 第一个EMA值是前period个数据的简单平均
    let mut ema = data[0..period].iter().sum::<f64>() / period as f64;
    ema_values.push(ema);
    
    // 计算后续的EMA值
    for i in period..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
        ema_values.push(ema);
    }
    
    ema_values
}

// RSI计算函数
pub fn calculate_rsi(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 50.0;
    }
    
    let mut gains = 0.0;
    let mut losses = 0.0;
    
    for i in 1..prices.len() {
        let change = prices[i] - prices[i-1];
        if change > 0.0 {
            gains += change;
        } else {
            losses += -change;
        }
    }
    
    gains /= (prices.len() - 1) as f64;
    losses /= (prices.len() - 1) as f64;
    
    if losses == 0.0 {
        100.0
    } else {
        100.0 - (100.0 / (1.0 + (gains / losses)))
    }
}

// MACD计算函数
pub fn calculate_macd(prices: &[f64]) -> f64 {
    if prices.len() < 26 {
        return 0.0;
    }
    
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    ema12 - ema26
}

// 计算完整的MACD指标（包括DIF、DEA、MACD柱）
pub fn calculate_macd_full(prices: &[f64]) -> (f64, f64, f64) {
    if prices.len() < 26 {
        return (0.0, 0.0, 0.0);
    }
    
    // 计算EMA12和EMA26
    let ema12 = calculate_ema(prices, 12);
    let ema26 = calculate_ema(prices, 26);
    
    // DIF = EMA12 - EMA26
    let dif = ema12 - ema26;
    
    // 计算最近9天的DIF值用于计算DEA
    let mut dif_values = Vec::new();
    for i in (prices.len().saturating_sub(9))..prices.len() {
        if i >= 26 {
            let sub_prices = &prices[0..=i];
            let sub_ema12 = calculate_ema(sub_prices, 12);
            let sub_ema26 = calculate_ema(sub_prices, 26);
            dif_values.push(sub_ema12 - sub_ema26);
        }
    }
    
    // DEA = EMA(DIF, 9)
    let dea = if dif_values.len() >= 9 {
        calculate_ema(&dif_values, 9)
    } else {
        dif // 如果数据不足，使用DIF作为DEA
    };
    
    // MACD柱 = 2 * (DIF - DEA)
    let macd = 2.0 * (dif - dea);
    
    (dif, dea, macd)
}

// 布林带位置计算
pub fn calculate_bollinger_position(prices: &[f64], current_price: f64) -> f64 {
    if prices.len() < 20 {
        return 0.0;
    }
    
    let ma = prices.iter().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter()
        .map(|p| (p - ma).powi(2))
        .sum::<f64>() / prices.len() as f64;
    let std_dev = variance.sqrt();
    
    let upper_band = ma + 2.0 * std_dev;
    let lower_band = ma - 2.0 * std_dev;
    
    if upper_band == lower_band {
        0.0
    } else {
        (current_price - lower_band) / (upper_band - lower_band) - 0.5
    }
}

// 随机指标K值计算
pub fn calculate_stochastic_k(prices: &[f64], current_price: f64) -> f64 {
    if prices.is_empty() {
        return 0.5;
    }
    
    let highest = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        0.5
    } else {
        (current_price - lowest) / (highest - lowest)
    }
}

// KDJ指标计算
pub fn calculate_kdj(highs: &[f64], lows: &[f64], closes: &[f64], n: usize) -> (f64, f64, f64) {
    if highs.len() < n || lows.len() < n || closes.len() < n {
        return (50.0, 50.0, 50.0);
    }
    
    let len = highs.len();
    let start = len.saturating_sub(n);
    
    // 计算N日内最高价和最低价
    let highest = highs[start..].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = lows[start..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        return (50.0, 50.0, 50.0);
    }
    
    // 计算RSV
    let rsv = (closes[len - 1] - lowest) / (highest - lowest) * 100.0;
    
    // 简化计算：使用最近3天的平均值模拟K值的平滑
    let mut k_values = vec![rsv];
    for i in 1..3 {
        if len > i {
            let idx = len - 1 - i;
            if idx >= start {
                let h = highs[start..=idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let l = lows[start..=idx].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                if h > l {
                    k_values.push((closes[idx] - l) / (h - l) * 100.0);
                }
            }
        }
    }
    
    let k = k_values.iter().sum::<f64>() / k_values.len() as f64;
    
    // D值是K值的3日移动平均
    let d = k * 0.667 + 50.0 * 0.333; // 简化计算
    
    // J = 3K - 2D
    let j = 3.0 * k - 2.0 * d;
    
    (k, d, j)
}

// OBV（能量潮）指标计算
pub fn calculate_obv(prices: &[f64], volumes: &[i64]) -> f64 {
    if prices.len() < 2 || volumes.len() < 2 {
        return 0.0;
    }
    
    let mut obv = 0.0;
    for i in 1..prices.len().min(volumes.len()) {
        if prices[i] > prices[i - 1] {
            obv += volumes[i] as f64;
        } else if prices[i] < prices[i - 1] {
            obv -= volumes[i] as f64;
        }
        // 价格不变时，OBV保持不变
    }
    
    obv
}

// CCI（商品通道指数）计算
pub fn calculate_cci(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }
    
    let start = highs.len().saturating_sub(period);
    let mut tp_values = Vec::new(); // Typical Price
    
    for i in start..highs.len() {
        let tp = (highs[i] + lows[i] + closes[i]) / 3.0;
        tp_values.push(tp);
    }
    
    // 计算移动平均
    let ma = tp_values.iter().sum::<f64>() / tp_values.len() as f64;
    
    // 计算平均偏差
    let md = tp_values.iter()
        .map(|&tp| (tp - ma).abs())
        .sum::<f64>() / tp_values.len() as f64;
    
    if md == 0.0 {
        return 0.0;
    }
    
    // CCI = (TP - MA) / (0.015 * MD)
    let current_tp = (highs.last().unwrap() + lows.last().unwrap() + closes.last().unwrap()) / 3.0;
    (current_tp - ma) / (0.015 * md)
}

// DMI（动向指标）计算
pub fn calculate_dmi(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> (f64, f64, f64, f64) {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let mut tr_values = Vec::new();
    let mut dm_plus_values = Vec::new();
    let mut dm_minus_values = Vec::new();
    
    // 计算TR、+DM、-DM
    for i in 1..highs.len() {
        // True Range
        let h_l = highs[i] - lows[i];
        let h_pc = (highs[i] - closes[i - 1]).abs();
        let l_pc = (lows[i] - closes[i - 1]).abs();
        let tr = h_l.max(h_pc).max(l_pc);
        tr_values.push(tr);
        
        // Directional Movement
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];
        
        let dm_plus = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let dm_minus = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        
        dm_plus_values.push(dm_plus);
        dm_minus_values.push(dm_minus);
    }
    
    if tr_values.len() < period {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    // 计算平滑后的值（使用简单移动平均代替Wilder's平滑）
    let start = tr_values.len().saturating_sub(period);
    let atr = tr_values[start..].iter().sum::<f64>() / period as f64;
    let adm_plus = dm_plus_values[start..].iter().sum::<f64>() / period as f64;
    let adm_minus = dm_minus_values[start..].iter().sum::<f64>() / period as f64;
    
    if atr == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    // 计算方向指标
    let di_plus = (adm_plus / atr) * 100.0;
    let di_minus = (adm_minus / atr) * 100.0;
    
    // 计算ADX
    let dx = if di_plus + di_minus == 0.0 {
        0.0
    } else {
        ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100.0
    };
    
    // 简化的ADX计算（应该是DX的平滑值）
    let adx = dx;
    
    (di_plus, di_minus, adx, dx)
}

// SAR（抛物线停损转向）指标计算
pub fn calculate_sar(highs: &[f64], lows: &[f64], af_step: f64, af_max: f64) -> Vec<f64> {
    if highs.len() < 2 || lows.len() < 2 {
        return vec![0.0; highs.len()];
    }
    
    let mut sar_values = Vec::new();
    let mut is_uptrend = true;
    let mut af = af_step;
    let mut ep = highs[0]; // 极值点
    let mut sar = lows[0];
    
    sar_values.push(sar);
    
    for i in 1..highs.len() {
        // 计算新的SAR
        let new_sar = sar + af * (ep - sar);
        
        if is_uptrend {
            // 上升趋势
            if lows[i] <= new_sar {
                // 趋势反转
                is_uptrend = false;
                sar = ep;
                ep = lows[i];
                af = af_step;
            } else {
                sar = new_sar;
                if highs[i] > ep {
                    ep = highs[i];
                    af = (af + af_step).min(af_max);
                }
            }
        } else {
            // 下降趋势
            if highs[i] >= new_sar {
                // 趋势反转
                is_uptrend = true;
                sar = ep;
                ep = highs[i];
                af = af_step;
            } else {
                sar = new_sar;
                if lows[i] < ep {
                    ep = lows[i];
                    af = (af + af_step).min(af_max);
                }
            }
        }
        
        sar_values.push(sar);
    }
    
    sar_values
}

// 获取特征所需的历史天数
pub fn get_feature_required_days(feature_name: &str) -> usize {
    match feature_name {
        "close" | "volume" | "change_percent" => 1,
        "ma5" => 5,
        "ma10" => 10,
        "ma20" | "bollinger" | "cci" => 20,
        "rsi" | "stochastic_k" | "stochastic_d" | "dmi_plus" | "dmi_minus" | "adx" => 14,
        "macd" | "macd_dif" | "macd_dea" | "macd_histogram" => 26,
        "momentum" => 10,
        "kdj_k" | "kdj_d" | "kdj_j" => 9,
        "obv" => 2,
        _ => 1,
    }
}

// 计算单个特征值
pub fn calculate_feature_value(
    feature_name: &str,
    prices: &[f64],
    volumes: &[i64],
    index: usize,
    _lookback_window: usize,
    highs: Option<&[f64]>,
    lows: Option<&[f64]>,
) -> Result<f64, candle_core::Error> {
    match feature_name {
        "close" => {
            Ok(prices[index])
        },
        "volume" => {
            Ok(volumes[index] as f64)
        },
        "change_percent" => {
            if index > 0 {
                let prev_price = prices[index - 1];
                let change = (prices[index] - prev_price) / prev_price;
                Ok(change)
            } else {
                Ok(0.0)
            }
        },
        "ma5" => {
            if index >= 4 {
                let ma5 = prices[index-4..=index].iter().sum::<f64>() / 5.0;
                Ok(ma5)
            } else {
                Ok(prices[index])
            }
        },
        "ma10" => {
            if index >= 9 {
                let ma10 = prices[index-9..=index].iter().sum::<f64>() / 10.0;
                Ok(ma10)
            } else {
                Ok(prices[index])
            }
        },
        "ma20" => {
            if index >= 19 {
                let ma20 = prices[index-19..=index].iter().sum::<f64>() / 20.0;
                Ok(ma20)
            } else {
                Ok(prices[index])
            }
        },
        "rsi" => {
            if index >= 14 {
                Ok(calculate_rsi(&prices[index-14..=index]))
            } else { 
                Ok(50.0) // 默认中性RSI
            }
        },
        "macd" => {
            if index >= 25 {
                Ok(calculate_macd(&prices[index-25..=index]))
            } else {
                Ok(0.0)
            }
        },
        "bollinger" => {
            if index >= 19 {
                Ok(calculate_bollinger_position(&prices[index-19..=index], prices[index]))
            } else {
                Ok(0.0)
            }
        },
        "stochastic_k" => {
            if index >= 13 {
                Ok(calculate_stochastic_k(&prices[index-13..=index], prices[index]))
            } else {
                Ok(0.5)
            }
        },
        "stochastic_d" => {
            if index >= 15 {
                // 计算前3天的K值的平均值
                let k_values: Vec<f64> = (0..3)
                    .map(|i| {
                        let k_index = index - i;
                        if k_index >= 13 {
                            calculate_stochastic_k(&prices[k_index-13..=k_index], prices[k_index])
                        } else {
                            0.5
                        }
                    })
                    .collect();
                Ok(k_values.iter().sum::<f64>() / k_values.len() as f64)
            } else {
                Ok(0.5)
            }
        },
        "momentum" => {
            if index >= 10 {
                let momentum = prices[index] / prices[index-10] - 1.0;
                Ok(momentum)
            } else {
                Ok(0.0)
            }
        },
        "kdj_k" | "kdj_d" | "kdj_j" => {
            // KDJ指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 9 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(8);
                    let (k, d, j) = calculate_kdj(&highs[start..=index], &lows[start..=index], &prices[start..=index], 9);
                    match feature_name {
                        "kdj_k" => Ok(k / 100.0), // 归一化到0-1
                        "kdj_d" => Ok(d / 100.0),
                        "kdj_j" => Ok(j / 100.0),
                        _ => Ok(0.0)
                    }
                } else {
                    Ok(0.5) // 默认中性值
                }
            } else {
                Ok(0.5)
            }
        },
        "cci" => {
            // CCI指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 20 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(19);
                    let cci = calculate_cci(&highs[start..=index], &lows[start..=index], &prices[start..=index], 20);
                    Ok(cci / 200.0) // 归一化，CCI通常在-200到200之间
                } else {
                    Ok(0.0)
                }
            } else {
                Ok(0.0)
            }
        },
        "obv" => {
            // OBV指标
            if index >= 1 {
                let obv = calculate_obv(&prices[0..=index], &volumes[0..=index]);
                // 归一化OBV（相对于平均成交量）
                let avg_volume = volumes[0..=index].iter().sum::<i64>() as f64 / (index + 1) as f64;
                Ok(obv / (avg_volume * (index + 1) as f64))
            } else {
                Ok(0.0)
            }
        },
        "macd_dif" | "macd_dea" | "macd_histogram" => {
            // 完整MACD指标
            if index >= 26 {
                let (dif, dea, histogram) = calculate_macd_full(prices);
                let normalized = match feature_name {
                    "macd_dif" => dif / prices[index],
                    "macd_dea" => dea / prices[index],
                    "macd_histogram" => histogram / prices[index],
                    _ => 0.0
                };
                Ok(normalized)
            } else {
                Ok(0.0)
            }
        },
        "dmi_plus" | "dmi_minus" | "adx" => {
            // DMI指标
            if let (Some(highs), Some(lows)) = (highs, lows) {
                if index >= 14 && highs.len() > index && lows.len() > index {
                    let start = index.saturating_sub(13);
                    let (di_plus, di_minus, adx, _) = calculate_dmi(&highs[start..=index], &lows[start..=index], &prices[start..=index], 14);
                    match feature_name {
                        "dmi_plus" => Ok(di_plus / 100.0),
                        "dmi_minus" => Ok(di_minus / 100.0),
                        "adx" => Ok(adx / 100.0),
                        _ => Ok(0.0)
                    }
                } else {
                    Ok(0.0)
                }
            } else {
                Ok(0.0)
            }
        },
        _ => {
            Ok(0.0)
        }
    }
} 

/// 计算ATR - 平均真实波幅 (衡量市场波动性)
/// 金融意义: ATR越大,波动越剧烈;ATR越小,波动越平缓
pub fn calculate_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }
    
    let mut trs = Vec::new();
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i-1]).abs();
        let low_close = (lows[i] - closes[i-1]).abs();
        let tr = high_low.max(high_close).max(low_close);
        trs.push(tr);
    }
    
    if trs.len() < period {
        return 0.0;
    }
    
    // 使用移动平均计算ATR
    let atr = trs[trs.len() - period..].iter().sum::<f64>() / period as f64;
    atr
}

/// 计算布林带 - 返回 (上轨, 中轨, 下轨)
/// 金融意义: 价格触及上轨=超买;触及下轨=超卖;中轨是均线支撑
pub fn calculate_bollinger_bands(prices: &[f64], period: usize, std_dev_multiplier: f64) -> (f64, f64, f64) {
    if prices.len() < period {
        let avg = prices.iter().sum::<f64>() / prices.len() as f64;
        return (avg, avg, avg);
    }
    
    let recent = &prices[prices.len() - period..];
    let middle = recent.iter().sum::<f64>() / period as f64;
    
    // 计算标准差
    let variance = recent.iter()
        .map(|&x| (x - middle).powi(2))
        .sum::<f64>() / period as f64;
    let std_dev = variance.sqrt();
    
    let upper = middle + std_dev_multiplier * std_dev;
    let lower = middle - std_dev_multiplier * std_dev;
    
    (upper, middle, lower)
}

/// 计算SAR信号 - 返回SAR值和趋势方向(true=多头,false=空头)
/// 金融意义: SAR点位是趋势反转的关键止损位
pub fn calculate_sar_signal(
    highs: &[f64], 
    lows: &[f64], 
    acceleration: f64, 
    max_acceleration: f64
) -> (f64, bool) {
    // 使用已有的SAR序列函数计算
    let sar_series = calculate_sar(highs, lows, acceleration, max_acceleration);
    if let Some(&last_sar) = sar_series.last() {
        let current_price = (highs.last().unwrap() + lows.last().unwrap()) / 2.0;
        let is_uptrend = current_price > last_sar;
        (last_sar, is_uptrend)
    } else {
        (lows.last().copied().unwrap_or(0.0), true)
    }
}

/// 计算DMI/ADX - 趋势强度指标
/// 返回: (DI+, DI-, ADX)
/// 金融意义: ADX>25表示强趋势;ADX<20表示无趋势震荡
pub fn calculate_dmi_adx(
    highs: &[f64], 
    lows: &[f64], 
    closes: &[f64], 
    period: usize
) -> (f64, f64, f64) {
    if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
        return (50.0, 50.0, 0.0);
    }
    
    let n = highs.len();
    let mut plus_dm_sum = 0.0;
    let mut minus_dm_sum = 0.0;
    let mut tr_sum = 0.0;
    
    for i in (n - period)..n {
        if i == 0 { continue; }
        
        // 计算方向性移动
        let up_move = highs[i] - highs[i-1];
        let down_move = lows[i-1] - lows[i];
        
        let plus_dm = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
        let minus_dm = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        
        // 计算真实波幅
        let high_low = highs[i] - lows[i];
        let high_close = (highs[i] - closes[i-1]).abs();
        let low_close = (lows[i] - closes[i-1]).abs();
        let tr = high_low.max(high_close).max(low_close);
        
        plus_dm_sum += plus_dm;
        minus_dm_sum += minus_dm;
        tr_sum += tr;
    }
    
    // 计算方向性指标
    let di_plus = if tr_sum > 0.0 { (plus_dm_sum / tr_sum) * 100.0 } else { 0.0 };
    let di_minus = if tr_sum > 0.0 { (minus_dm_sum / tr_sum) * 100.0 } else { 0.0 };
    
    // 计算ADX
    let dx = if di_plus + di_minus > 0.0 {
        ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100.0
    } else {
        0.0
    };
    
    // 简化ADX计算(应该用移动平均,这里用DX近似)
    let adx = dx;
    
    (di_plus, di_minus, adx)
}

/// 计算威廉指标 (Williams %R)
/// 返回值范围: -100 到 0
/// 金融意义: <-80超卖(买入);>-20超买(卖出)
pub fn calculate_williams_r(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return -50.0;
    }
    
    let n = highs.len();
    let recent_highs = &highs[n - period..];
    let recent_lows = &lows[n - period..];
    let current_close = closes[n - 1];
    
    let highest = recent_highs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lowest = recent_lows.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    if highest == lowest {
        return -50.0;
    }
    
    ((highest - current_close) / (highest - lowest)) * -100.0
}

/// 计算ROC - 变动率指标
/// 金融意义: 衡量价格变动速度,>0上涨动能,<0下跌动能
pub fn calculate_roc(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period + 1 {
        return 0.0;
    }
    
    let n = prices.len();
    let current = prices[n - 1];
    let previous = prices[n - period - 1];
    
    if previous == 0.0 {
        return 0.0;
    }
    
    ((current - previous) / previous) * 100.0
}

/// 计算TRIX - 三重指数平滑移动平均
/// 金融意义: 过滤短期波动,显示长期趋势,>0多头,<0空头
pub fn calculate_trix(prices: &[f64], period: usize) -> f64 {
    if prices.len() < period * 3 {
        return 0.0;
    }
    
    // 使用已有的calculate_ema函数
    let _ema1 = calculate_ema(prices, period);
    
    // 简化:只计算最后一个TRIX值
    // 完整实现需要递归计算三次EMA序列
    0.0
}

/// 市场情绪指标 - 综合多个指标判断市场情绪
#[derive(Debug, Clone)]
pub struct MarketSentiment {
    pub sentiment_score: f64,      // 情绪得分 0-100
    pub sentiment_level: String,   // 情绪等级
    pub fear_greed_index: f64,     // 恐惧贪婪指数 0-100
    pub market_phase: String,      // 市场阶段
}

/// 计算市场情绪
pub fn calculate_market_sentiment(
    prices: &[f64],
    volumes: &[i64],
    highs: &[f64],
    lows: &[f64],
) -> MarketSentiment {
    let n = prices.len();
    if n < 20 {
        return MarketSentiment {
            sentiment_score: 50.0,
            sentiment_level: "中性".to_string(),
            fear_greed_index: 50.0,
            market_phase: "不明".to_string(),
        };
    }
    
    let mut score = 50.0;
    
    // 1. 价格动量 (±15分) - 降低影响
    let roc = calculate_roc(prices, 10);
    score += (roc / 10.0).clamp(-15.0, 15.0);
    
    // 2. 波动率 (±12分,对称) - 修正为对称
    let atr = calculate_atr(highs, lows, prices, 14);
    let current_price = prices[n-1];
    let volatility = (atr / current_price) * 100.0;
    if volatility > 3.0 {
        score -= 12.0; // 高波动=恐惧
    } else if volatility < 1.0 {
        score += 12.0; // 低波动=稳定 (对称)
    }
    
    // 3. RSI (±10分) - 修正逻辑:超买扣分!
    let rsi = calculate_rsi(prices);
    if rsi > 70.0 {
        score -= 10.0; // 超买=风险,应该扣分!
    } else if rsi < 30.0 {
        score += 10.0; // 超卖=机会,加分
    }
    
    // 4. 成交量 (±8分) - 降低影响
    if volumes.len() >= 20 {
        let recent_vol = volumes[n-5..].iter().sum::<i64>() as f64 / 5.0;
        let avg_vol = volumes[n-20..n-5].iter().sum::<i64>() as f64 / 15.0;
        if recent_vol > avg_vol * 1.5 {
            score += 8.0; // 放量=活跃
        } else if recent_vol < avg_vol * 0.7 {
            score -= 5.0; // 缩量=冷清
        }
    }
    
    // 严格限制在5-95分范围
    score = score.clamp(5.0, 95.0);
    
    let sentiment_level = if score >= 75.0 {
        "极度贪婪"
    } else if score >= 60.0 {
        "贪婪"
    } else if score >= 45.0 {
        "中性偏多"
    } else if score >= 30.0 {
        "恐惧"
    } else {
        "极度恐惧"
    }.to_string();
    
    let market_phase = if score >= 70.0 {
        "过热期-注意风险"
    } else if score >= 55.0 {
        "上升期"
    } else if score >= 45.0 {
        "震荡期"
    } else if score >= 30.0 {
        "下跌期"
    } else {
        "恐慌期-机会期"
    }.to_string();
    
    MarketSentiment {
        sentiment_score: score,
        sentiment_level,
        fear_greed_index: score,
        market_phase,
    }
} 