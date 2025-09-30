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