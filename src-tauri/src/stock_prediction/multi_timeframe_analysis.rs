use crate::db::models::HistoricalData;
use crate::stock_prediction::technical_indicators::{MacdData, KdjData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::Datelike;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockData {
    pub symbol: String,
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl From<&HistoricalData> for StockData {
    fn from(data: &HistoricalData) -> Self {
        StockData {
            symbol: data.symbol.clone(),
            date: data.date.format("%Y-%m-%d").to_string(),
            open: data.open,
            high: data.high,
            low: data.low,
            close: data.close,
            volume: data.volume as f64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeSignal {
    pub symbol: String,
    pub date: String,
    pub daily_macd_signal: MacdSignal,
    pub weekly_macd_signal: MacdSignal,
    pub monthly_macd_signal: MacdSignal,
    pub daily_kdj_signal: KdjSignal,
    pub weekly_kdj_signal: KdjSignal,
    pub monthly_kdj_signal: KdjSignal,
    pub combined_signal_strength: f64,
    pub prediction_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacdSignal {
    pub macd: f64,
    pub signal: f64,
    pub histogram: f64,
    pub is_golden_cross: bool,
    pub is_death_cross: bool,
    pub signal_strength: f64,
    pub trend_direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdjSignal {
    pub k: f64,
    pub d: f64,
    pub j: f64,
    pub is_golden_cross: bool,
    pub is_death_cross: bool,
    pub signal_strength: f64,
    pub overbought_oversold: String,
}

#[derive(Debug, Clone)]
pub struct TimeframeData {
    pub daily: Vec<StockData>,
    pub weekly: Vec<StockData>,
    pub monthly: Vec<StockData>,
}

/// 将日线数据转换为周线数据
pub fn convert_to_weekly(daily_data: &[StockData]) -> Vec<StockData> {
    let mut weekly_data = Vec::new();
    let mut current_week: Vec<&StockData> = Vec::new();
    
    for data in daily_data {
        // 解析日期
        if let Ok(date) = chrono::NaiveDate::parse_from_str(&data.date, "%Y-%m-%d") {
            let week_start = date - chrono::Duration::days(date.weekday().num_days_from_monday() as i64);
            
            if current_week.is_empty() {
                current_week.push(data);
            } else {
                let current_week_start = chrono::NaiveDate::parse_from_str(&current_week[0].date, "%Y-%m-%d")
                    .unwrap() - chrono::Duration::days(chrono::NaiveDate::parse_from_str(&current_week[0].date, "%Y-%m-%d").unwrap().weekday().num_days_from_monday() as i64);
                
                if week_start == current_week_start {
                    current_week.push(data);
                } else {
                    // 生成周线数据
                    if !current_week.is_empty() {
                        weekly_data.push(create_weekly_candle(&current_week));
                    }
                    current_week.clear();
                    current_week.push(data);
                }
            }
        }
    }
    
    // 处理最后一周
    if !current_week.is_empty() {
        weekly_data.push(create_weekly_candle(&current_week));
    }
    
    weekly_data
}

/// 将日线数据转换为月线数据
pub fn convert_to_monthly(daily_data: &[StockData]) -> Vec<StockData> {
    let mut monthly_data = Vec::new();
    let mut current_month: Vec<&StockData> = Vec::new();
    
    for data in daily_data {
        if let Ok(date) = chrono::NaiveDate::parse_from_str(&data.date, "%Y-%m-%d") {
            let month_key = format!("{}-{:02}", date.year(), date.month());
            
            if current_month.is_empty() {
                current_month.push(data);
            } else {
                let current_month_date = chrono::NaiveDate::parse_from_str(&current_month[0].date, "%Y-%m-%d").unwrap();
                let current_month_key = format!("{}-{:02}", current_month_date.year(), current_month_date.month());
                
                if month_key == current_month_key {
                    current_month.push(data);
                } else {
                    // 生成月线数据
                    if !current_month.is_empty() {
                        monthly_data.push(create_monthly_candle(&current_month));
                    }
                    current_month.clear();
                    current_month.push(data);
                }
            }
        }
    }
    
    // 处理最后一个月
    if !current_month.is_empty() {
        monthly_data.push(create_monthly_candle(&current_month));
    }
    
    monthly_data
}

/// 创建周线K线数据
fn create_weekly_candle(week_data: &[&StockData]) -> StockData {
    let open = week_data.first().unwrap().open;
    let close = week_data.last().unwrap().close;
    let high = week_data.iter().map(|d| d.high).fold(0.0, f64::max);
    let low = week_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min);
    let volume = week_data.iter().map(|d| d.volume).sum::<f64>();
    
    StockData {
        symbol: week_data[0].symbol.clone(),
        date: week_data.last().unwrap().date.clone(), // 使用周末日期
        open,
        high,
        low,
        close,
        volume,
    }
}

/// 创建月线K线数据
fn create_monthly_candle(month_data: &[&StockData]) -> StockData {
    let open = month_data.first().unwrap().open;
    let close = month_data.last().unwrap().close;
    let high = month_data.iter().map(|d| d.high).fold(0.0, f64::max);
    let low = month_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min);
    let volume = month_data.iter().map(|d| d.volume).sum::<f64>();
    
    StockData {
        symbol: month_data[0].symbol.clone(),
        date: month_data.last().unwrap().date.clone(), // 使用月末日期
        open,
        high,
        low,
        close,
        volume,
    }
}

/// 计算MACD信号
pub fn calculate_macd_signal(data: &[StockData], fast_period: usize, slow_period: usize, signal_period: usize) -> Vec<MacdSignal> {
    let closes: Vec<f64> = data.iter().map(|d| d.close).collect();
    let macd_data = calculate_macd_full(&closes, fast_period, slow_period, signal_period);
    let mut signals = Vec::new();
    
    for i in 1..macd_data.len() {
        let current = &macd_data[i];
        let previous = &macd_data[i - 1];
        
        let is_golden_cross = previous.macd <= previous.signal && current.macd > current.signal;
        let is_death_cross = previous.macd >= previous.signal && current.macd < current.signal;
        
        // 计算信号强度
        let signal_strength = if is_golden_cross || is_death_cross {
            let macd_momentum = (current.macd - previous.macd).abs();
            let signal_momentum = (current.signal - previous.signal).abs();
            let histogram_strength = current.histogram.abs();
            
            (macd_momentum + signal_momentum + histogram_strength) / 3.0
        } else {
            0.0
        };
        
        // 判断趋势方向
        let trend_direction = if current.macd > current.signal {
            if current.histogram > previous.histogram {
                "强势上涨".to_string()
            } else {
                "弱势上涨".to_string()
            }
        } else if current.histogram < previous.histogram {
            "强势下跌".to_string()
        } else {
            "弱势下跌".to_string()
        };
        
        signals.push(MacdSignal {
            macd: current.macd,
            signal: current.signal,
            histogram: current.histogram,
            is_golden_cross,
            is_death_cross,
            signal_strength,
            trend_direction,
        });
    }
    
    signals
}

/// 计算完整的MACD指标序列
fn calculate_macd_full(closes: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> Vec<MacdData> {
    if closes.len() < slow_period {
        return Vec::new();
    }
    
    let mut macd_data = Vec::new();
    let mut dif_values = Vec::new();
    
    // 计算EMA12和EMA26序列
    for i in slow_period..=closes.len() {
        let sub_closes = &closes[0..i];
        let ema12 = calculate_ema_for_period(sub_closes, fast_period);
        let ema26 = calculate_ema_for_period(sub_closes, slow_period);
        let dif = ema12 - ema26;
        dif_values.push(dif);
    }
    
    // 计算DEA（DIF的9日EMA）
    for i in signal_period..=dif_values.len() {
        let sub_dif = &dif_values[0..i];
        let dea = calculate_ema_for_period(sub_dif, signal_period);
        let dif = dif_values[i - 1];
        let histogram = 2.0 * (dif - dea);
        
        macd_data.push(MacdData {
            macd: dif,
            signal: dea,
            histogram,
        });
    }
    
    macd_data
}

/// 计算指定周期的EMA
fn calculate_ema_for_period(data: &[f64], period: usize) -> f64 {
    if data.len() < period {
        return data.iter().sum::<f64>() / data.len() as f64;
    }
    
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema = data[0..period].iter().sum::<f64>() / period as f64;
    
    for i in period..data.len() {
        ema = (data[i] - ema) * multiplier + ema;
    }
    
    ema
}

/// 计算KDJ信号
pub fn calculate_kdj_signal(data: &[StockData], k_period: usize, d_period: usize, j_period: usize) -> Vec<KdjSignal> {
    let kdj_data = calculate_kdj_full(data, k_period, d_period, j_period);
    let mut signals = Vec::new();
    
    for i in 1..kdj_data.len() {
        let current = &kdj_data[i];
        let previous = &kdj_data[i - 1];
        
        let is_golden_cross = previous.k <= previous.d && current.k > current.d && current.k < 80.0;
        let is_death_cross = previous.k >= previous.d && current.k < current.d && current.k > 20.0;
        
        // 计算信号强度
        let signal_strength = if is_golden_cross || is_death_cross {
            let k_momentum = (current.k - previous.k).abs();
            let d_momentum = (current.d - previous.d).abs();
            let j_divergence = (current.j - current.k).abs();
            
            (k_momentum + d_momentum + j_divergence) / 30.0 // 归一化到0-1
        } else {
            0.0
        };
        
        // 判断超买超卖状态
        let overbought_oversold = if current.k > 80.0 && current.d > 80.0 {
            "超买".to_string()
        } else if current.k < 20.0 && current.d < 20.0 {
            "超卖".to_string()
        } else if current.k > 50.0 && current.d > 50.0 {
            "多头区域".to_string()
        } else {
            "空头区域".to_string()
        };
        
        signals.push(KdjSignal {
            k: current.k,
            d: current.d,
            j: current.j,
            is_golden_cross,
            is_death_cross,
            signal_strength,
            overbought_oversold,
        });
    }
    
    signals
}

/// 计算完整的KDJ指标序列
fn calculate_kdj_full(data: &[StockData], k_period: usize, d_period: usize, _j_period: usize) -> Vec<KdjData> {
    if data.len() < k_period {
        return Vec::new();
    }
    
    let mut kdj_data = Vec::new();
    let mut k_values = Vec::new();
    
    // 计算K值序列
    for i in k_period..=data.len() {
        let sub_data = &data[i - k_period..i];
        let highest = sub_data.iter().map(|d| d.high).fold(0.0, f64::max);
        let lowest = sub_data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min);
        let current_close = data[i - 1].close;
        
        let rsv = if highest > lowest {
            (current_close - lowest) / (highest - lowest) * 100.0
        } else {
            50.0
        };
        
        // K值是RSV的平滑值
        let k = if k_values.is_empty() {
            rsv
        } else {
            (rsv + 2.0 * k_values.last().unwrap()) / 3.0
        };
        
        k_values.push(k);
    }
    
    // 计算D值和J值
    let mut d_values = Vec::new();
    for i in 0..k_values.len() {
        let d = if i < d_period - 1 {
            k_values[i]
        } else {
            let start_idx = i + 1 - d_period;
            k_values[start_idx..=i].iter().sum::<f64>() / d_period as f64
        };
        
        let k = k_values[i];
        let j = 3.0 * k - 2.0 * d;
        
        d_values.push(d);
        kdj_data.push(KdjData { k, d, j });
    }
    
    kdj_data
}

/// 生成多时间周期信号
pub fn generate_multi_timeframe_signals(symbol: &str, historical_data: &[HistoricalData]) -> Vec<MultiTimeframeSignal> {
    // 转换为内部数据格式
    let daily_data: Vec<StockData> = historical_data.iter().map(|d| d.into()).collect();
    let weekly_data = convert_to_weekly(&daily_data);
    let monthly_data = convert_to_monthly(&daily_data);
    
    // 计算各时间周期的信号
    let daily_macd_signals = calculate_macd_signal(&daily_data, 12, 26, 9);
    let weekly_macd_signals = calculate_macd_signal(&weekly_data, 12, 26, 9);
    let monthly_macd_signals = calculate_macd_signal(&monthly_data, 12, 26, 9);
    
    let daily_kdj_signals = calculate_kdj_signal(&daily_data, 9, 3, 3);
    let weekly_kdj_signals = calculate_kdj_signal(&weekly_data, 9, 3, 3);
    let monthly_kdj_signals = calculate_kdj_signal(&monthly_data, 9, 3, 3);
    
    let mut multi_signals = Vec::new();
    
    // 对齐信号数据（以日线为基准）
    for (i, daily_data_point) in daily_data.iter().enumerate() {
        if i >= daily_macd_signals.len() || i >= daily_kdj_signals.len() {
            continue;
        }
        
        // 找到对应的周线和月线信号
        let weekly_index = find_corresponding_index(&daily_data_point.date, &weekly_data);
        let monthly_index = find_corresponding_index(&daily_data_point.date, &monthly_data);
        
        let weekly_macd = weekly_index
            .and_then(|idx| weekly_macd_signals.get(idx))
            .cloned()
            .unwrap_or_else(create_default_macd_signal);
            
        let monthly_macd = monthly_index
            .and_then(|idx| monthly_macd_signals.get(idx))
            .cloned()
            .unwrap_or_else(create_default_macd_signal);
            
        let weekly_kdj = weekly_index
            .and_then(|idx| weekly_kdj_signals.get(idx))
            .cloned()
            .unwrap_or_else(create_default_kdj_signal);
            
        let monthly_kdj = monthly_index
            .and_then(|idx| monthly_kdj_signals.get(idx))
            .cloned()
            .unwrap_or_else(create_default_kdj_signal);
        
        // 计算综合信号强度
        let combined_signal_strength = calculate_combined_signal_strength(
            &daily_macd_signals[i],
            &weekly_macd,
            &monthly_macd,
            &daily_kdj_signals[i],
            &weekly_kdj,
            &monthly_kdj,
        );
        
        // 计算预测置信度
        let prediction_confidence = calculate_prediction_confidence(
            &daily_macd_signals[i],
            &weekly_macd,
            &monthly_macd,
            &daily_kdj_signals[i],
            &weekly_kdj,
            &monthly_kdj,
        );
        
        multi_signals.push(MultiTimeframeSignal {
            symbol: symbol.to_string(),
            date: daily_data_point.date.clone(),
            daily_macd_signal: daily_macd_signals[i].clone(),
            weekly_macd_signal: weekly_macd,
            monthly_macd_signal: monthly_macd,
            daily_kdj_signal: daily_kdj_signals[i].clone(),
            weekly_kdj_signal: weekly_kdj,
            monthly_kdj_signal: monthly_kdj,
            combined_signal_strength,
            prediction_confidence,
        });
    }
    
    multi_signals
}

/// 找到对应的索引
fn find_corresponding_index(date: &str, data: &[StockData]) -> Option<usize> {
    if let Ok(target_date) = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d") {
        for (i, stock_data) in data.iter().enumerate() {
            if let Ok(stock_date) = chrono::NaiveDate::parse_from_str(&stock_data.date, "%Y-%m-%d") {
                if stock_date <= target_date {
                    return Some(i);
                }
            }
        }
    }
    None
}

/// 创建默认MACD信号
fn create_default_macd_signal() -> MacdSignal {
    MacdSignal {
        macd: 0.0,
        signal: 0.0,
        histogram: 0.0,
        is_golden_cross: false,
        is_death_cross: false,
        signal_strength: 0.0,
        trend_direction: "无信号".to_string(),
    }
}

/// 创建默认KDJ信号
fn create_default_kdj_signal() -> KdjSignal {
    KdjSignal {
        k: 50.0,
        d: 50.0,
        j: 50.0,
        is_golden_cross: false,
        is_death_cross: false,
        signal_strength: 0.0,
        overbought_oversold: "中性".to_string(),
    }
}

/// 计算综合信号强度
fn calculate_combined_signal_strength(
    daily_macd: &MacdSignal,
    weekly_macd: &MacdSignal,
    monthly_macd: &MacdSignal,
    daily_kdj: &KdjSignal,
    weekly_kdj: &KdjSignal,
    monthly_kdj: &KdjSignal,
) -> f64 {
    let mut total_strength = 0.0;
    let mut signal_count = 0;
    
    // MACD信号权重：月线 > 周线 > 日线
    if daily_macd.is_golden_cross || daily_macd.is_death_cross {
        total_strength += daily_macd.signal_strength * 1.0;
        signal_count += 1;
    }
    
    if weekly_macd.is_golden_cross || weekly_macd.is_death_cross {
        total_strength += weekly_macd.signal_strength * 2.0;
        signal_count += 2;
    }
    
    if monthly_macd.is_golden_cross || monthly_macd.is_death_cross {
        total_strength += monthly_macd.signal_strength * 3.0;
        signal_count += 3;
    }
    
    // KDJ信号权重：月线 > 周线 > 日线
    if daily_kdj.is_golden_cross || daily_kdj.is_death_cross {
        total_strength += daily_kdj.signal_strength * 1.0;
        signal_count += 1;
    }
    
    if weekly_kdj.is_golden_cross || weekly_kdj.is_death_cross {
        total_strength += weekly_kdj.signal_strength * 2.0;
        signal_count += 2;
    }
    
    if monthly_kdj.is_golden_cross || monthly_kdj.is_death_cross {
        total_strength += monthly_kdj.signal_strength * 3.0;
        signal_count += 3;
    }
    
    if signal_count > 0 {
        total_strength / signal_count as f64
    } else {
        0.0
    }
}

/// 计算预测置信度
fn calculate_prediction_confidence(
    daily_macd: &MacdSignal,
    weekly_macd: &MacdSignal,
    monthly_macd: &MacdSignal,
    daily_kdj: &KdjSignal,
    weekly_kdj: &KdjSignal,
    monthly_kdj: &KdjSignal,
) -> f64 {
    let mut confidence = 0.5; // 基础置信度
    
    // 多时间周期信号一致性加分
    let macd_bullish_signals = [daily_macd.is_golden_cross, weekly_macd.is_golden_cross, monthly_macd.is_golden_cross]
        .iter().filter(|&&x| x).count();
    let macd_bearish_signals = [daily_macd.is_death_cross, weekly_macd.is_death_cross, monthly_macd.is_death_cross]
        .iter().filter(|&&x| x).count();
    
    let kdj_bullish_signals = [daily_kdj.is_golden_cross, weekly_kdj.is_golden_cross, monthly_kdj.is_golden_cross]
        .iter().filter(|&&x| x).count();
    let kdj_bearish_signals = [daily_kdj.is_death_cross, weekly_kdj.is_death_cross, monthly_kdj.is_death_cross]
        .iter().filter(|&&x| x).count();
    
    // 信号一致性越高，置信度越高
    if macd_bullish_signals >= 2 && kdj_bullish_signals >= 2 {
        confidence += 0.3; // 多重买入信号
    } else if macd_bearish_signals >= 2 && kdj_bearish_signals >= 2 {
        confidence += 0.3; // 多重卖出信号
    } else if macd_bullish_signals >= 1 && kdj_bullish_signals >= 1 {
        confidence += 0.15; // 部分买入信号
    } else if macd_bearish_signals >= 1 && kdj_bearish_signals >= 1 {
        confidence += 0.15; // 部分卖出信号
    }
    
    // 月线信号权重最高
    if monthly_macd.is_golden_cross || monthly_macd.is_death_cross {
        confidence += 0.2;
    }
    if monthly_kdj.is_golden_cross || monthly_kdj.is_death_cross {
        confidence += 0.2;
    }
    
    // 信号强度加权
    let avg_signal_strength = (daily_macd.signal_strength + weekly_macd.signal_strength + monthly_macd.signal_strength
        + daily_kdj.signal_strength + weekly_kdj.signal_strength + monthly_kdj.signal_strength) / 6.0;
    
    confidence += avg_signal_strength * 0.1;
    
    confidence.min(1.0).max(0.0)
}

/// 获取最新的多时间周期信号
pub fn get_latest_multi_timeframe_signal(symbol: &str, historical_data: &[HistoricalData]) -> Option<MultiTimeframeSignal> {
    let signals = generate_multi_timeframe_signals(symbol, historical_data);
    signals.last().cloned()
}

/// 分析多时间周期信号的预测价值
pub fn analyze_signal_prediction_value(signals: &[MultiTimeframeSignal]) -> HashMap<String, f64> {
    let mut analysis = HashMap::new();
    
    let mut monthly_golden_accuracy = 0.0;
    let mut monthly_death_accuracy = 0.0;
    let mut weekly_golden_accuracy = 0.0;
    let mut weekly_death_accuracy = 0.0;
    let mut combined_signal_accuracy = 0.0;
    
    let mut monthly_golden_count = 0;
    let mut monthly_death_count = 0;
    let mut weekly_golden_count = 0;
    let mut weekly_death_count = 0;
    let mut combined_signal_count = 0;
    
    for signal in signals {
        // 月线MACD金叉准确率分析
        if signal.monthly_macd_signal.is_golden_cross {
            monthly_golden_accuracy += signal.prediction_confidence;
            monthly_golden_count += 1;
        }
        
        // 月线MACD死叉准确率分析
        if signal.monthly_macd_signal.is_death_cross {
            monthly_death_accuracy += signal.prediction_confidence;
            monthly_death_count += 1;
        }
        
        // 周线KDJ金叉准确率分析
        if signal.weekly_kdj_signal.is_golden_cross {
            weekly_golden_accuracy += signal.prediction_confidence;
            weekly_golden_count += 1;
        }
        
        // 周线KDJ死叉准确率分析
        if signal.weekly_kdj_signal.is_death_cross {
            weekly_death_accuracy += signal.prediction_confidence;
            weekly_death_count += 1;
        }
        
        // 综合信号强度分析
        if signal.combined_signal_strength > 0.5 {
            combined_signal_accuracy += signal.prediction_confidence;
            combined_signal_count += 1;
        }
    }
    
    if monthly_golden_count > 0 {
        analysis.insert("monthly_macd_golden_accuracy".to_string(), monthly_golden_accuracy / monthly_golden_count as f64);
    }
    if monthly_death_count > 0 {
        analysis.insert("monthly_macd_death_accuracy".to_string(), monthly_death_accuracy / monthly_death_count as f64);
    }
    if weekly_golden_count > 0 {
        analysis.insert("weekly_kdj_golden_accuracy".to_string(), weekly_golden_accuracy / weekly_golden_count as f64);
    }
    if weekly_death_count > 0 {
        analysis.insert("weekly_kdj_death_accuracy".to_string(), weekly_death_accuracy / weekly_death_count as f64);
    }
    if combined_signal_count > 0 {
        analysis.insert("combined_signal_accuracy".to_string(), combined_signal_accuracy / combined_signal_count as f64);
    }
    
    analysis.insert("total_signals".to_string(), signals.len() as f64);
    analysis.insert("monthly_golden_signals".to_string(), monthly_golden_count as f64);
    analysis.insert("monthly_death_signals".to_string(), monthly_death_count as f64);
    analysis.insert("weekly_golden_signals".to_string(), weekly_golden_count as f64);
    analysis.insert("weekly_death_signals".to_string(), weekly_death_count as f64);
    analysis.insert("strong_combined_signals".to_string(), combined_signal_count as f64);
    
    analysis
} 