use chrono::{Weekday, Datelike};
use crate::stock_prediction::types::{HistoricalVolatilityFeatures, Direction, TechnicalSignals, Prediction};

// A股交易规则工具函数 - 改进版本，包含节假日判断
pub fn is_trading_day(date: chrono::NaiveDate) -> bool {
    // 首先检查是否为工作日
    match date.weekday() {
        Weekday::Sat | Weekday::Sun => return false,
        _ => {}
    }
    
    // 检查中国法定节假日（简化版本，实际应该有更完整的节假日数据）
    let year = date.year();
    let month = date.month();
    let day = date.day();
    
    // 固定节假日
    match (month, day) {
        // 元旦
        (1, 1) => return false,
        // 清明节（大约4月4-6日，简化处理）
        (4, 4..=6) => return false,
        // 劳动节
        (5, 1..=3) => return false,
        // 国庆节
        (10, 1..=7) => return false,
        _ => {}
    }
    
    // 春节假期（农历新年，根据年份大致判断，实际应该用农历计算）
    match year {
        2024 => {
            // 2024年春节: 2月10-17日
            if month == 2 && day >= 10 && day <= 17 {
                return false;
            }
        },
        2025 => {
            // 2025年春节: 1月29日-2月4日
            if (month == 1 && day >= 29) || (month == 2 && day <= 4) {
                return false;
            }
        },
        2023 => {
            // 2023年春节: 1月21-27日
            if month == 1 && day >= 21 && day <= 27 {
                return false;
            }
        },
        _ => {
            // 对于其他年份，简化处理：假设春节大致在1月底2月初
            if (month == 1 && day >= 20) || (month == 2 && day <= 10) {
                // 这里应该有更精确的农历计算，暂时简化
                // 可以考虑引入专门的农历库或节假日API
            }
        }
    }
    
    true
}

pub fn get_next_trading_day(date: chrono::NaiveDate) -> chrono::NaiveDate {
    let mut next_date = date + chrono::Duration::days(1);
    while !is_trading_day(next_date) {
        next_date += chrono::Duration::days(1);
        // 防止无限循环，最多向前查找30天
        if (next_date - date).num_days() > 30 {
            println!("⚠️ 警告：查找下一个交易日超过30天，可能存在节假日配置问题");
            break;
        }
    }
    next_date
}

// A股涨跌停限制（考虑ST股票和科创板）
pub fn clamp_daily_change(change_percent: f64) -> f64 {
    // A股普通股票涨跌停限制：±10%
    // ST股票：±5%（这里简化为普通股票处理）
    // 科创板：±20%（这里简化为普通股票处理）
    change_percent.clamp(-10.0, 10.0)
}

// 数据平滑处理函数
pub fn smooth_price_data(prices: &[f64]) -> Vec<f64> {
    let mut smoothed = prices.to_vec();
    
    // 使用中位数滤波器移除价格异常值
    for i in 2..smoothed.len()-2 {
        let window: Vec<f64> = smoothed[i-2..=i+2].to_vec();
        let mut sorted_window = window.clone();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_window[2];
        
        // 如果当前值与中位数相差超过20%，用中位数替换
        if (smoothed[i] - median).abs() / median > 0.2 {
            smoothed[i] = median;
        }
    }
    
    smoothed
}

pub fn smooth_volume_data(volumes: &[i64]) -> Vec<i64> {
    let mut smoothed = volumes.to_vec();
    
    // 移除成交量异常值
    for i in 2..smoothed.len()-2 {
        let window: Vec<i64> = smoothed[i-2..=i+2].to_vec();
        let avg = window.iter().sum::<i64>() as f64 / window.len() as f64;
        
        // 如果当前值与平均值相差超过5倍，用平均值替换
        if (smoothed[i] as f64 - avg).abs() / avg > 5.0 {
            smoothed[i] = avg as i64;
        }
    }
    
    smoothed
}

// 计算历史波动率
pub fn calculate_historical_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 20 {
        return 0.02; // 默认2%波动率
    }
    
    // 计算过去20天的价格变化率
    let mut daily_returns = Vec::new();
    for i in 1..std::cmp::min(21, prices.len()) {
        let return_rate = (prices[prices.len() - i] - prices[prices.len() - i - 1]) / prices[prices.len() - i - 1];
        daily_returns.push(return_rate);
    }
    
    // 计算标准差
    let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
    let variance = daily_returns.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / daily_returns.len() as f64;
    
    variance.sqrt().min(0.1) // 限制最大波动率为10%
}

// 计算近期趋势
pub fn calculate_recent_trend(prices: &[f64]) -> f64 {
    if prices.len() < 10 {
        return 0.0;
    }
    
    let recent_len = std::cmp::min(10, prices.len());
    let recent_prices = &prices[prices.len() - recent_len..];
    
    // 使用简单线性回归计算趋势
    let n = recent_len as f64;
    let sum_x = (0..recent_len).sum::<usize>() as f64;
    let sum_y = recent_prices.iter().sum::<f64>();
    let sum_xy = recent_prices.iter().enumerate()
        .map(|(i, &price)| i as f64 * price)
        .sum::<f64>();
    let sum_x2 = (0..recent_len).map(|i| (i * i) as f64).sum::<f64>();
    
    // 趋势斜率
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let avg_price = sum_y / n;
    
    // 标准化趋势 (相对于平均价格)
    (slope / avg_price).clamp(-0.05, 0.05) // 限制在±5%范围内
}

// 计算支撑阻力位
pub fn calculate_support_resistance(prices: &[f64], current_price: f64) -> f64 {
    if prices.len() < 30 {
        return 0.01; // 默认1%影响
    }
    
    let recent_len = std::cmp::min(30, prices.len());
    let recent_prices = &prices[prices.len() - recent_len..];
    
    // 寻找局部高点和低点
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    
    for i in 1..recent_prices.len() - 1 {
        if recent_prices[i] > recent_prices[i-1] && recent_prices[i] > recent_prices[i+1] {
            highs.push(recent_prices[i]);
        }
        if recent_prices[i] < recent_prices[i-1] && recent_prices[i] < recent_prices[i+1] {
            lows.push(recent_prices[i]);
        }
    }
    
    // 找到最近的支撑位和阻力位
    let resistance = highs.iter().fold(0.0, |acc, &x| if x > current_price && (acc == 0.0 || x < acc) { x } else { acc });
    let support = lows.iter().fold(0.0, |acc, &x| if x < current_price && x > acc { x } else { acc });
    
    // 计算支撑阻力影响
    let sr_strength = if resistance > 0.0 && support > 0.0 {
        let resistance_dist = (resistance - current_price) / current_price;
        let support_dist = (current_price - support) / current_price;
        (resistance_dist - support_dist) * 0.5 // 平衡支撑阻力影响
    } else if resistance > 0.0 {
        (resistance - current_price) / current_price * 0.3
    } else if support > 0.0 {
        (current_price - support) / current_price * 0.3
    } else {
        0.0
    };
    
    sr_strength.clamp(-0.03, 0.03) // 限制在±3%范围内
}

// 改进的准确率计算，更加重视方向预测
pub fn calculate_direction_focused_accuracy(predictions: &[f64], actuals: &[f64]) -> (f64, f64) {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return (0.0, 0.0);
    }
    
    let mut direction_correct = 0;
    let mut total_predictions = 0;
    let mut price_error_sum = 0.0;
    
    for i in 1..predictions.len().min(actuals.len()) {
        // 计算预测和实际的变化方向
        let pred_change = predictions[i] - predictions[i-1];
        let actual_change = actuals[i] - actuals[i-1];
        
        // 方向分类（使用更严格的阈值）
        let pred_direction = Direction::from_change_percent(pred_change / predictions[i-1] * 100.0);
        let actual_direction = Direction::from_change_percent(actual_change / actuals[i-1] * 100.0);
        
        // 方向准确性检查
        if pred_direction == actual_direction {
            direction_correct += 1;
        }
        
        // 价格准确性（相对误差）
        let relative_error = ((predictions[i] - actuals[i]) / actuals[i]).abs();
        price_error_sum += relative_error;
        
        total_predictions += 1;
    }
    
    if total_predictions == 0 {
        return (0.0, 0.0);
    }
    
    let direction_accuracy = direction_correct as f64 / total_predictions as f64;
    let price_accuracy = (1.0 - (price_error_sum / total_predictions as f64)).max(0.0);
    
    // 方向准确率权重提高到70%，价格准确率30%
    let combined_accuracy = direction_accuracy * 0.7 + price_accuracy * 0.3;
    
    (direction_accuracy, combined_accuracy.min(0.85)) // 限制最高准确率保持现实性
}

// 分析历史波动特征的函数
pub fn analyze_historical_volatility_pattern(prices: &[f64], days: usize) -> HistoricalVolatilityFeatures {
    let window = days.min(prices.len() - 1);
    if window < 5 {
        return HistoricalVolatilityFeatures {
            avg_daily_change: 0.01,
            avg_up_change: 0.01,
            avg_down_change: 0.01,
            max_consecutive_up: 2,
            max_consecutive_down: 2,
            up_down_ratio: 1.0,
            volatility_pattern: "数据不足".to_string(),
        };
    }
    
    // 计算日涨跌幅
    let mut daily_changes = Vec::with_capacity(window);
    for i in 1..=window {
        let idx = prices.len() - i;
        let change = (prices[idx] - prices[idx - 1]) / prices[idx - 1];
        daily_changes.push(change);
    }
    
    // 计算平均涨跌幅
    let avg_daily_change = daily_changes.iter().map(|c| c.abs()).sum::<f64>() / daily_changes.len() as f64;
    
    // 区分上涨和下跌
    let up_changes: Vec<f64> = daily_changes.iter().filter(|&&c| c > 0.0).cloned().collect();
    let down_changes: Vec<f64> = daily_changes.iter().filter(|&&c| c < 0.0).cloned().collect();
    
    let avg_up_change = if !up_changes.is_empty() {
        up_changes.iter().sum::<f64>() / up_changes.len() as f64
    } else {
        0.01
    };
    
    let avg_down_change = if !down_changes.is_empty() {
        down_changes.iter().sum::<f64>().abs() / down_changes.len() as f64
    } else {
        0.01
    };
    
    // 计算连续上涨/下跌天数
    let mut max_up = 0;
    let mut max_down = 0;
    let mut current_up = 0;
    let mut current_down = 0;
    
    for &change in daily_changes.iter() {
        if change > 0.0 {
            current_up += 1;
            current_down = 0;
            max_up = max_up.max(current_up);
        } else if change < 0.0 {
            current_down += 1;
            current_up = 0;
            max_down = max_down.max(current_down);
        } else {
            // 持平时重置计数
            current_up = 0;
            current_down = 0;
        }
    }
    
    // 上涨/下跌天数比例
    let up_days = daily_changes.iter().filter(|&&c| c > 0.0).count();
    let down_days = daily_changes.iter().filter(|&&c| c < 0.0).count();
    let up_down_ratio = if down_days > 0 {
        up_days as f64 / down_days as f64
    } else {
        up_days as f64
    };
    
    // 确定波动模式
    let volatility_pattern = if avg_daily_change > 0.02 {
        if max_up > 3 && max_down > 3 {
            "大幅波动型".to_string()
        } else if max_up > max_down {
            "大幅上涨型".to_string()
        } else {
            "大幅下跌型".to_string()
        }
    } else if avg_daily_change > 0.01 {
        if up_down_ratio > 1.5 {
            "温和上涨型".to_string()
        } else if up_down_ratio < 0.67 {
            "温和下跌型".to_string()
        } else {
            "震荡型".to_string()
        }
    } else {
        "低波动型".to_string()
    };
    
    HistoricalVolatilityFeatures {
        avg_daily_change,
        avg_up_change,
        avg_down_change,
        max_consecutive_up: max_up,
        max_consecutive_down: max_down,
        up_down_ratio,
        volatility_pattern,
    }
}

// 打印最后一条真实数据和第一条预测数据的对比
pub fn print_last_real_vs_prediction(
    dates: &[String], 
    prices: &[f64], 
    technical_signals: &TechnicalSignals,
    prediction: &Prediction,
    last_change_percent: f64
) {
    if dates.is_empty() || prices.is_empty() {
        println!("⚠️ 没有足够的历史数据进行对比");
        return;
    }

    let last_date = dates.last().unwrap();
    let last_price = prices.last().unwrap();

    println!("\n📊 真实数据与预测对比:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📅 最后一条真实数据 ({}):", last_date);
    println!("   价格: {:.2}, 涨跌幅: {:.2}%", last_price, last_change_percent);
    println!("   MACD: DIF={:.4}, DEA={:.4}, HIST={:.4}, 金叉={}, 死叉={}", 
             technical_signals.macd_dif, technical_signals.macd_dea, 
             technical_signals.macd_histogram, 
             technical_signals.macd_golden_cross, technical_signals.macd_death_cross);
    println!("   KDJ: K={:.2}, D={:.2}, J={:.2}, 金叉={}, 死叉={}, 超买={}, 超卖={}", 
             technical_signals.kdj_k, technical_signals.kdj_d, technical_signals.kdj_j,
             technical_signals.kdj_golden_cross, technical_signals.kdj_death_cross,
             technical_signals.kdj_overbought, technical_signals.kdj_oversold);
    println!("   RSI: {:.2}, CCI: {:.2}", technical_signals.rsi, technical_signals.cci);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📈 第一天预测 ({}):", prediction.target_date);
    println!("   价格: {:.2}, 涨跌幅: {:.2}%, 置信度: {:.2}%", 
             prediction.predicted_price, prediction.predicted_change_percent, 
             prediction.confidence * 100.0);
    
    if let Some(tech) = &prediction.technical_indicators {
        println!("   MACD: DIF={:.4}, DEA={:.4}, HIST={:.4}, 金叉={}, 死叉={}", 
                tech.macd_dif, tech.macd_dea, tech.macd_histogram, 
                tech.macd_golden_cross, tech.macd_death_cross);
        println!("   KDJ: K={:.2}, D={:.2}, J={:.2}, 金叉={}, 死叉={}, 超买={}, 超卖={}", 
                tech.kdj_k, tech.kdj_d, tech.kdj_j,
                tech.kdj_golden_cross, tech.kdj_death_cross,
                tech.kdj_overbought, tech.kdj_oversold);
        println!("   RSI: {:.2}, CCI: {:.2}", tech.rsi, tech.cci);
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

// 趋势状态枚举
#[derive(Debug, Clone, PartialEq)]
pub enum TrendState {
    StrongBullish,  // 强烈看涨（日线月线都金叉且在零轴上方）
    Bullish,        // 看涨（主要指标显示上涨趋势）
    Neutral,        // 中性（指标混合信号）
    Bearish,        // 看跌（主要指标显示下跌趋势）
    StrongBearish,  // 强烈看跌（日线月线都死叉且在零轴下方）
}

// 趋势分析结构体
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub daily_trend: TrendState,
    pub monthly_trend: TrendState,
    pub overall_trend: TrendState,
    pub trend_strength: f64,  // 趋势强度 -1.0 到 1.0
    pub trend_confidence: f64, // 趋势置信度 0.0 到 1.0
    pub bias_multiplier: f64,  // 预测偏向倍数 0.5 到 2.0
    pub trend_description: String,
}

// 通过日线和月线MACD、KDJ分析股票趋势状态
pub fn analyze_stock_trend(
    prices: &[f64], 
    highs: &[f64], 
    lows: &[f64], 
    _volumes: &[i64]
) -> TrendAnalysis {
    use crate::stock_prediction::technical_indicators::{calculate_macd_full, calculate_kdj};
    
    let len = prices.len();
    
    // 需要足够的数据进行趋势分析
    if len < 120 { // 至少需要约半年数据
        return TrendAnalysis {
            daily_trend: TrendState::Neutral,
            monthly_trend: TrendState::Neutral,
            overall_trend: TrendState::Neutral,
            trend_strength: 0.0,
            trend_confidence: 0.3,
            bias_multiplier: 1.0,
            trend_description: "数据不足，无法准确判断趋势".to_string(),
        };
    }
    
    // === 日线级别分析（最近30天数据） ===
    let daily_period = 30.min(len);
    let daily_data = &prices[len-daily_period..];
    let daily_highs = &highs[len-daily_period..];
    let daily_lows = &lows[len-daily_period..];
    
    // 计算日线MACD序列（用于判断金叉死叉）
    let mut daily_macd_signals = Vec::new();
    for i in 26..daily_data.len() {
        let (dif, dea, histogram) = calculate_macd_full(&daily_data[0..=i]);
        daily_macd_signals.push((dif, dea, histogram));
    }
    
    // 计算日线KDJ序列
    let mut daily_kdj_signals = Vec::new();
    for i in 9..daily_data.len() {
        let start = i.saturating_sub(8);
        let (k, d, j) = calculate_kdj(&daily_highs[start..=i], &daily_lows[start..=i], &daily_data[start..=i], 9);
        daily_kdj_signals.push((k, d, j));
    }
    
    // === 月线级别分析（模拟月线：每20个交易日一个月） ===
    let monthly_period = 120.min(len); // 最多6个月数据
    let monthly_step = 20; // 每20个交易日作为一个月
    let mut monthly_prices = Vec::new();
    let mut monthly_highs = Vec::new();
    let mut monthly_lows = Vec::new();
    
    // 构造月线数据（取每个月的收盘价、最高价、最低价）
    for i in (monthly_step..=monthly_period).step_by(monthly_step) {
        let start_idx = len - monthly_period + i - monthly_step;
        let end_idx = len - monthly_period + i - 1;
        
        if end_idx < len {
            monthly_prices.push(prices[end_idx]); // 收盘价
            monthly_highs.push(highs[start_idx..=end_idx].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))); // 最高价
            monthly_lows.push(lows[start_idx..=end_idx].iter().fold(f64::INFINITY, |a, &b| a.min(b))); // 最低价
        }
    }
    
    // 计算月线MACD和KDJ（如果有足够数据）
    let (monthly_macd_dif, monthly_macd_dea, monthly_macd_histogram) = if monthly_prices.len() >= 6 {
        calculate_macd_full(&monthly_prices)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    let (monthly_kdj_k, monthly_kdj_d, monthly_kdj_j) = if monthly_prices.len() >= 3 && monthly_highs.len() >= 3 && monthly_lows.len() >= 3 {
        calculate_kdj(&monthly_highs, &monthly_lows, &monthly_prices, monthly_prices.len().min(3))
    } else {
        (50.0, 50.0, 50.0)
    };
    
    // === 日线趋势判断 ===
    let daily_trend = if let (Some(latest_macd), Some(prev_macd)) = (daily_macd_signals.last(), daily_macd_signals.get(daily_macd_signals.len().saturating_sub(2))) {
        let macd_golden_cross = prev_macd.0 <= prev_macd.1 && latest_macd.0 > latest_macd.1;
        let macd_death_cross = prev_macd.0 >= prev_macd.1 && latest_macd.0 < latest_macd.1;
        let macd_above_zero = latest_macd.2 > 0.0; // MACD柱在零轴上方
        let macd_rising = latest_macd.2 > prev_macd.2; // MACD柱上升
        
        let kdj_golden_cross = if let (Some(latest_kdj), Some(prev_kdj)) = (daily_kdj_signals.last(), daily_kdj_signals.get(daily_kdj_signals.len().saturating_sub(2))) {
            prev_kdj.0 <= prev_kdj.1 && latest_kdj.0 > latest_kdj.1
        } else { false };
        
        let kdj_death_cross = if let (Some(latest_kdj), Some(prev_kdj)) = (daily_kdj_signals.last(), daily_kdj_signals.get(daily_kdj_signals.len().saturating_sub(2))) {
            prev_kdj.0 >= prev_kdj.1 && latest_kdj.0 < latest_kdj.1
        } else { false };
        
        // 日线趋势综合判断
        if macd_golden_cross && kdj_golden_cross && macd_above_zero {
            TrendState::StrongBullish
        } else if (macd_golden_cross || kdj_golden_cross) && (macd_above_zero || macd_rising) {
            TrendState::Bullish
        } else if macd_death_cross && kdj_death_cross && !macd_above_zero {
            TrendState::StrongBearish
        } else if (macd_death_cross || kdj_death_cross) && (!macd_above_zero || !macd_rising) {
            TrendState::Bearish
        } else {
            TrendState::Neutral
        }
    } else {
        TrendState::Neutral
    };
    
    // === 月线趋势判断 ===
    let monthly_trend = if monthly_prices.len() >= 3 {
        let monthly_macd_bullish = monthly_macd_dif > monthly_macd_dea && monthly_macd_histogram > 0.0;
        let monthly_macd_bearish = monthly_macd_dif < monthly_macd_dea && monthly_macd_histogram < 0.0;
        let monthly_kdj_bullish = monthly_kdj_k > monthly_kdj_d && monthly_kdj_j > 50.0;
        let monthly_kdj_bearish = monthly_kdj_k < monthly_kdj_d && monthly_kdj_j < 50.0;
        
        if monthly_macd_bullish && monthly_kdj_bullish {
            TrendState::StrongBullish
        } else if monthly_macd_bullish || monthly_kdj_bullish {
            TrendState::Bullish
        } else if monthly_macd_bearish && monthly_kdj_bearish {
            TrendState::StrongBearish
        } else if monthly_macd_bearish || monthly_kdj_bearish {
            TrendState::Bearish
        } else {
            TrendState::Neutral
        }
    } else {
        TrendState::Neutral
    };
    
    // === 综合趋势判断 ===
    // 重新设计：日线权重更大，月线作为背景确认
    let overall_trend = match (&daily_trend, &monthly_trend) {
        // 日线强势信号 + 月线配合 = 强势
        (TrendState::StrongBullish, TrendState::StrongBullish) | 
        (TrendState::StrongBullish, TrendState::Bullish) => TrendState::StrongBullish,
        
        (TrendState::StrongBearish, TrendState::StrongBearish) | 
        (TrendState::StrongBearish, TrendState::Bearish) => TrendState::StrongBearish,
        
        // 日线强势但月线不配合 = 降为一般强度（日线为主）
        (TrendState::StrongBullish, _) => TrendState::Bullish,
        (TrendState::StrongBearish, _) => TrendState::Bearish,
        
        // 日线一般强度 + 月线强势背景 = 保持或增强
        (TrendState::Bullish, TrendState::StrongBullish) => TrendState::Bullish,
        (TrendState::Bearish, TrendState::StrongBearish) => TrendState::Bearish,
        
        // 日线一般强度，以日线为主
        (TrendState::Bullish, TrendState::Bullish) | 
        (TrendState::Bullish, TrendState::Neutral) |
        (TrendState::Bullish, _) => TrendState::Bullish,
        
        (TrendState::Bearish, TrendState::Bearish) | 
        (TrendState::Bearish, TrendState::Neutral) |
        (TrendState::Bearish, _) => TrendState::Bearish,
        
        // 日线中性时，参考月线但权重降低
        (TrendState::Neutral, TrendState::StrongBullish) => TrendState::Bullish,
        (TrendState::Neutral, TrendState::Bullish) => TrendState::Bullish,
        (TrendState::Neutral, TrendState::StrongBearish) => TrendState::Bearish,
        (TrendState::Neutral, TrendState::Bearish) => TrendState::Bearish,
        (TrendState::Neutral, TrendState::Neutral) => TrendState::Neutral,
    };
    
    // === 计算趋势强度和置信度 ===
    // 重新设计：更重视日线指标
    let trend_strength = match &overall_trend {
                 TrendState::StrongBullish => {
             // 日线强度占70%，月线背景占30%
             let daily_strength = 0.8_f64;
             let monthly_boost = match &monthly_trend {
                 TrendState::StrongBullish | TrendState::Bullish => 0.1_f64,
                 TrendState::Neutral => 0.0_f64,
                 _ => -0.05_f64,
             };
             (daily_strength + monthly_boost).min(1.0_f64)
         },
                 TrendState::Bullish => {
             let daily_strength = 0.5_f64;
             let monthly_boost = match &monthly_trend {
                 TrendState::StrongBullish | TrendState::Bullish => 0.1_f64,
                 TrendState::Neutral => 0.0_f64,
                 _ => -0.1_f64,
             };
             (daily_strength + monthly_boost).max(0.2_f64)
         },
        TrendState::Neutral => 0.0,
                 TrendState::Bearish => {
             let daily_strength = -0.5_f64;
             let monthly_boost = match &monthly_trend {
                 TrendState::StrongBearish | TrendState::Bearish => -0.1_f64,
                 TrendState::Neutral => 0.0_f64,
                 _ => 0.1_f64,
             };
             (daily_strength + monthly_boost).min(-0.2_f64)
         },
         TrendState::StrongBearish => {
             let daily_strength = -0.8_f64;
             let monthly_boost = match &monthly_trend {
                 TrendState::StrongBearish | TrendState::Bearish => -0.1_f64,
                 TrendState::Neutral => 0.0_f64,
                 _ => 0.05_f64,
             };
             (daily_strength + monthly_boost).max(-1.0_f64)
         },
    };
    
    // 置信度计算：主要基于日线信号的一致性
    let trend_confidence = match (&daily_trend, &monthly_trend) {
        // 日线强势信号 + 月线背景支持 = 最高置信度
        (TrendState::StrongBullish, TrendState::StrongBullish) | 
        (TrendState::StrongBearish, TrendState::StrongBearish) => 0.95,
        
        // 日线强势 + 月线一般支持 = 高置信度
        (TrendState::StrongBullish, TrendState::Bullish) | 
        (TrendState::StrongBearish, TrendState::Bearish) => 0.88,
        
        // 日线强势但月线不支持 = 中高置信度（仍以日线为主）
        (TrendState::StrongBullish, _) | (TrendState::StrongBearish, _) => 0.75,
        
        // 日线一般强度 + 月线支持 = 中等置信度
        (TrendState::Bullish, TrendState::StrongBullish) | 
        (TrendState::Bullish, TrendState::Bullish) |
        (TrendState::Bearish, TrendState::StrongBearish) | 
        (TrendState::Bearish, TrendState::Bearish) => 0.70,
        
        // 日线一般强度 + 月线中性或不支持 = 较低置信度
        (TrendState::Bullish, _) | (TrendState::Bearish, _) => 0.60,
        
        // 日线中性 + 月线信号 = 低置信度
        (TrendState::Neutral, TrendState::StrongBullish) | 
        (TrendState::Neutral, TrendState::StrongBearish) => 0.45,
        (TrendState::Neutral, TrendState::Bullish) | 
        (TrendState::Neutral, TrendState::Bearish) => 0.40,
        
        // 都中性 = 最低置信度
        (TrendState::Neutral, TrendState::Neutral) => 0.30,
    };
    
    // === 计算预测偏向倍数（重视日线金叉死叉）===
    let bias_multiplier = match &daily_trend {
        TrendState::StrongBullish => {
            // 日线强烈看涨，月线背景调整
            let base = 1.9;
            match &monthly_trend {
                TrendState::StrongBullish | TrendState::Bullish => base,
                TrendState::Neutral => base * 0.9,
                _ => base * 0.8, // 月线看跌时适当降低
            }
        },
        TrendState::Bullish => {
            let base = 1.4;
            match &monthly_trend {
                TrendState::StrongBullish | TrendState::Bullish => base * 1.1,
                TrendState::Neutral => base,
                _ => base * 0.85,
            }
        },
        TrendState::Neutral => {
            // 日线中性时，月线背景有一定影响
            match &monthly_trend {
                TrendState::StrongBullish => 1.2,
                TrendState::Bullish => 1.1,
                TrendState::Neutral => 1.0,
                TrendState::Bearish => 0.9,
                TrendState::StrongBearish => 0.8,
            }
        },
        TrendState::Bearish => {
            let base = 0.6;
            match &monthly_trend {
                TrendState::StrongBearish | TrendState::Bearish => base * 0.9,
                TrendState::Neutral => base,
                _ => base * 1.15,
            }
        },
        TrendState::StrongBearish => {
            let base = 0.2;
            match &monthly_trend {
                TrendState::StrongBearish | TrendState::Bearish => base,
                TrendState::Neutral => base * 1.1,
                _ => base * 1.2, // 月线看涨时适当提高
            }
        },
    };
    
    // === 趋势描述（突出日线重要性）===
    let trend_description = match (&daily_trend, &monthly_trend) {
        (TrendState::StrongBullish, TrendState::StrongBullish) => 
            format!("强烈上涨趋势 - 日线技术指标强势金叉，月线背景完全支持，建议积极关注 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::StrongBullish, TrendState::Bullish) => 
            format!("强烈上涨趋势 - 日线强势金叉，月线背景支持，看涨信号明确 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::StrongBullish, _) => 
            format!("短期强势上涨 - 日线技术指标强势金叉，需关注短期机会 (置信度:{:.0}%)", trend_confidence * 100.0),
        
        (TrendState::Bullish, TrendState::StrongBullish) => 
            format!("上涨趋势 - 日线偏多，月线强势背景支持，谨慎乐观 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::Bullish, _) => 
            format!("短期上涨趋势 - 日线技术指标偏多，建议谨慎关注 (置信度:{:.0}%)", trend_confidence * 100.0),
        
        (TrendState::StrongBearish, TrendState::StrongBearish) => 
            format!("强烈下跌趋势 - 日线技术指标强势死叉，月线背景确认，建议规避风险 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::StrongBearish, TrendState::Bearish) => 
            format!("强烈下跌趋势 - 日线强势死叉，月线背景支持，看跌信号明确 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::StrongBearish, _) => 
            format!("短期强势下跌 - 日线技术指标强势死叉，需注意短期风险 (置信度:{:.0}%)", trend_confidence * 100.0),
        
        (TrendState::Bearish, TrendState::StrongBearish) => 
            format!("下跌趋势 - 日线偏空，月线强势背景确认，建议谨慎 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::Bearish, _) => 
            format!("短期下跌趋势 - 日线技术指标偏空，建议保持谨慎 (置信度:{:.0}%)", trend_confidence * 100.0),
        
        (TrendState::Neutral, TrendState::StrongBullish) => 
            format!("震荡偏多 - 日线信号混合，但月线强势背景支持 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::Neutral, TrendState::StrongBearish) => 
            format!("震荡偏空 - 日线信号混合，月线背景偏空 (置信度:{:.0}%)", trend_confidence * 100.0),
        (TrendState::Neutral, _) => 
            format!("震荡趋势 - 日线技术指标信号混合，方向不明，建议观望 (置信度:{:.0}%)", trend_confidence * 100.0),
    };
    
    println!("📈 股票趋势分析 (日线主导):");
    println!("   📊 日线趋势: {:?} (主要信号)", daily_trend);
    println!("   📅 月线趋势: {:?} (背景确认)", monthly_trend);
    println!("   🎯 综合趋势: {:?} (强度: {:.2}, 置信度: {:.0}%)", overall_trend, trend_strength, trend_confidence * 100.0);
    println!("   ⚖️  预测偏向: {:.2} (日线权重更大)", bias_multiplier);
    println!("   📝 {}", trend_description);
    
    TrendAnalysis {
        daily_trend,
        monthly_trend,
        overall_trend,
        trend_strength,
        trend_confidence,
        bias_multiplier,
        trend_description,
    }
}

// 保留旧的函数用于向后兼容
pub fn predict_direction_simple(
    prices: &[f64],
    highs: &[f64], 
    lows: &[f64],
    volumes: &[i64],
    current_price: f64
) -> VolumePricePredictionStrategy {
    predict_with_volume_price(prices, highs, lows, volumes, current_price)
}

pub fn calculate_conservative_change(
    strategy: &VolumePricePredictionStrategy,
    historical_volatility: f64,
    day: i32
) -> (f64, f64) {
    calculate_volume_price_change(strategy, historical_volatility, day)
}

// 量价关系预测策略 - 回归股票预测本质
#[derive(Debug, Clone)]
pub struct VolumePricePredictionStrategy {
    pub predicted_direction: String,  // "上涨", "下跌", "横盘"
    pub direction_confidence: f64,    // 方向置信度 0.0-1.0
    pub suggested_change_range: (f64, f64), // 建议的涨跌幅范围
    pub volume_price_signal: String, // 量价关系信号
    pub price_trend: String,         // 价格趋势
    pub volume_trend: String,        // 成交量趋势
    pub key_factors: Vec<String>,    // 关键影响因素
}

// 基于量价关系的核心预测函数
pub fn predict_with_volume_price(
    prices: &[f64],
    highs: &[f64], 
    lows: &[f64],
    volumes: &[i64],
    current_price: f64
) -> VolumePricePredictionStrategy {
    let len = prices.len();
    if len < 10 {
        return VolumePricePredictionStrategy {
            predicted_direction: "横盘".to_string(),
            direction_confidence: 0.3,
            suggested_change_range: (-1.0, 1.0),
            volume_price_signal: "数据不足".to_string(),
            price_trend: "未知".to_string(),
            volume_trend: "未知".to_string(),
            key_factors: vec!["数据不足".to_string()],
        };
    }
    
    println!("📊 量价关系分析:");
    
    // === 1. 价格趋势分析 ===
    let recent_5_avg = prices[len.saturating_sub(5)..].iter().sum::<f64>() / 5.0;
    let recent_10_avg = prices[len.saturating_sub(10)..].iter().sum::<f64>() / 10.0;
    let price_momentum_5d = (current_price - prices[len-5]) / prices[len-5] * 100.0;
    let price_momentum_3d = (current_price - prices[len-3]) / prices[len-3] * 100.0;
    
    let price_trend = if current_price > recent_5_avg && recent_5_avg > recent_10_avg {
        if price_momentum_5d > 3.0 {
            "强势上涨"
        } else {
            "温和上涨"
        }
    } else if current_price < recent_5_avg && recent_5_avg < recent_10_avg {
        if price_momentum_5d < -3.0 {
            "强势下跌"
        } else {
            "温和下跌"
        }
    } else {
        "横盘震荡"
    };
    
    println!("   💰 价格趋势: {}", price_trend);
    println!("   📈 5日动量: {:.2}%", price_momentum_5d);
    println!("   📈 3日动量: {:.2}%", price_momentum_3d);
    
    // === 2. 成交量趋势分析 ===
    let recent_5_vol_avg = volumes[len.saturating_sub(5)..].iter().sum::<i64>() as f64 / 5.0;
    let recent_10_vol_avg = volumes[len.saturating_sub(10)..].iter().sum::<i64>() as f64 / 10.0;
    let latest_volume = *volumes.last().unwrap() as f64;
    let prev_volume = volumes[len-2] as f64;
    
    let volume_change = (latest_volume - prev_volume) / prev_volume * 100.0;
    let volume_vs_5d = (latest_volume - recent_5_vol_avg) / recent_5_vol_avg * 100.0;
    
    let volume_trend = if latest_volume > recent_5_vol_avg * 1.5 {
        "显著放量"
    } else if latest_volume > recent_5_vol_avg * 1.2 {
        "温和放量"
    } else if latest_volume < recent_5_vol_avg * 0.7 {
        "明显缩量"
    } else if latest_volume < recent_5_vol_avg * 0.8 {
        "温和缩量"
    } else {
        "量能平稳"
    };
    
    println!("   📊 成交量趋势: {}", volume_trend);
    println!("   📊 量能变化: {:.0}% (vs前日)", volume_change);
    println!("   📊 相对5日: {:.0}%", volume_vs_5d);
    
    // === 3. 核心量价关系判断 ===
    let mut bullish_score = 0;
    let mut bearish_score = 0;
    let mut key_factors = Vec::new();
    
    // 量价关系核心逻辑
    match (price_trend, volume_trend) {
        // 最强看涨信号
        ("强势上涨", "显著放量") => {
            bullish_score += 5;
            key_factors.push("放量强势上涨".to_string());
        },
        ("强势上涨", "温和放量") => {
            bullish_score += 4;
            key_factors.push("放量上涨".to_string());
        },
        ("温和上涨", "显著放量") => {
            bullish_score += 4;
            key_factors.push("放量推升".to_string());
        },
        
        // 次强看涨信号
        ("温和上涨", "温和放量") => {
            bullish_score += 3;
            key_factors.push("温和放量上涨".to_string());
        },
        ("强势上涨", "量能平稳") => {
            bullish_score += 2;
            key_factors.push("价涨量平".to_string());
        },
        
        // 警示信号（涨势中的缩量）
        ("强势上涨", "温和缩量") | ("温和上涨", "温和缩量") => {
            bullish_score += 1;
            key_factors.push("上涨缩量警示".to_string());
        },
        ("强势上涨", "明显缩量") | ("温和上涨", "明显缩量") => {
            bearish_score += 1;
            key_factors.push("上涨无量警示".to_string());
        },
        
        // 最强看跌信号
        ("强势下跌", "显著放量") => {
            bearish_score += 5;
            key_factors.push("放量大跌".to_string());
        },
        ("强势下跌", "温和放量") => {
            bearish_score += 4;
            key_factors.push("放量下跌".to_string());
        },
        ("温和下跌", "显著放量") => {
            bearish_score += 4;
            key_factors.push("放量打压".to_string());
        },
        
        // 次强看跌信号
        ("温和下跌", "温和放量") => {
            bearish_score += 3;
            key_factors.push("温和放量下跌".to_string());
        },
        
        // 可能止跌信号
        ("强势下跌", "明显缩量") | ("温和下跌", "明显缩量") => {
            bullish_score += 2;
            key_factors.push("下跌缩量止跌".to_string());
        },
        ("强势下跌", "温和缩量") | ("温和下跌", "温和缩量") => {
            bullish_score += 1;
            key_factors.push("跌势减缓".to_string());
        },
        
        // 横盘整理
        ("横盘震荡", _) => {
            key_factors.push("横盘整理".to_string());
        },
        
        _ => {
            key_factors.push("量价关系复杂".to_string());
        }
    }
    
    // === 4. 技术位置确认 ===
    let highest_10d = highs[len.saturating_sub(10)..].iter().fold(0.0_f64, |a, &b| a.max(b));
    let lowest_10d = lows[len.saturating_sub(10)..].iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let position_ratio = (current_price - lowest_10d) / (highest_10d - lowest_10d);
    
    match position_ratio {
        p if p > 0.8 => {
            bearish_score += 1;
            key_factors.push("接近10日高位".to_string());
        },
        p if p < 0.2 => {
            bullish_score += 1;
            key_factors.push("接近10日低位".to_string());
        },
        _ => {
            key_factors.push("位置适中".to_string());
        }
    }
    
    // === 5. 连续性确认 ===
    if price_momentum_3d > 1.0 && price_momentum_5d > 2.0 {
        bullish_score += 1;
        key_factors.push("连续上涨".to_string());
    } else if price_momentum_3d < -1.0 && price_momentum_5d < -2.0 {
        bearish_score += 1;
        key_factors.push("连续下跌".to_string());
    }
    
    // === 6. 综合判断 ===
    let (predicted_direction, direction_confidence, suggested_range) = if bullish_score >= bearish_score + 3 {
        // 明确看涨
        let confidence = (0.7 + (bullish_score - bearish_score) as f64 * 0.05).min(0.95);
        ("上涨".to_string(), confidence, (0.8, 6.0))
    } else if bearish_score >= bullish_score + 3 {
        // 明确看跌
        let confidence = (0.7 + (bearish_score - bullish_score) as f64 * 0.05).min(0.95);
        ("下跌".to_string(), confidence, (-6.0, -0.8))
    } else if bullish_score > bearish_score {
        // 轻微看涨
        let confidence = 0.55 + (bullish_score - bearish_score) as f64 * 0.03;
        ("上涨".to_string(), confidence, (0.3, 3.5))
    } else if bearish_score > bullish_score {
        // 轻微看跌
        let confidence = 0.55 + (bearish_score - bullish_score) as f64 * 0.03;
        ("下跌".to_string(), confidence, (-3.5, -0.3))
    } else {
        // 横盘
        ("横盘".to_string(), 0.5, (-2.0, 2.0))
    };
    
    let volume_price_signal = match (&predicted_direction[..], bullish_score.max(bearish_score)) {
        ("上涨", score) if score >= 4 => "强烈买入".to_string(),
        ("上涨", score) if score >= 2 => "买入".to_string(),
        ("下跌", score) if score >= 4 => "强烈卖出".to_string(),
        ("下跌", score) if score >= 2 => "卖出".to_string(),
        _ => "持有".to_string(),
    };
    
    println!("   🎯 看涨信号: {} 分", bullish_score);
    println!("   🎯 看跌信号: {} 分", bearish_score);
    println!("   🎯 预测方向: {} (置信度: {:.0}%)", predicted_direction, direction_confidence * 100.0);
    println!("   🎯 交易信号: {}", volume_price_signal);
    println!("   📋 关键因素: {}", key_factors.join(", "));
    
    VolumePricePredictionStrategy {
        predicted_direction,
        direction_confidence,
        suggested_change_range: suggested_range,
        volume_price_signal,
        price_trend: price_trend.to_string(),
        volume_trend: volume_trend.to_string(),
        key_factors,
    }
}

// 基于量价关系的保守涨跌幅计算
pub fn calculate_volume_price_change(
    strategy: &VolumePricePredictionStrategy,
    historical_volatility: f64,
    day: i32
) -> (f64, f64) {
    let base_range = strategy.suggested_change_range;
    let confidence = strategy.direction_confidence;
    
    // 时间衰减 - 量价信号衰减较慢
    let time_decay = 0.95_f64.powi(day - 1);
    
    // 根据量价信号强度调整
    let signal_multiplier = match strategy.volume_price_signal.as_str() {
        "强烈买入" | "强烈卖出" => 1.3, // 强信号放大
        "买入" | "卖出" => 1.0,        // 正常信号
        _ => 0.6,                       // 弱信号缩小
    };
    
    let adjusted_min = base_range.0 * signal_multiplier * time_decay;
    let adjusted_max = base_range.1 * signal_multiplier * time_decay;
    
    // 添加随机波动（较小）
    let volatility_adj = historical_volatility.clamp(0.01, 0.03);
    let noise = (rand::random::<f64>() - 0.5) * volatility_adj * 50.0;
    
    // 根据方向选择
    let predicted_change = match strategy.predicted_direction.as_str() {
        "上涨" => {
            let target = adjusted_min + (adjusted_max - adjusted_min) * rand::random::<f64>();
            (target + noise).max(0.2) // 确保上涨至少0.2%
        },
        "下跌" => {
            let target = adjusted_min + (adjusted_max - adjusted_min) * rand::random::<f64>();
            (target + noise).min(-0.2) // 确保下跌至少-0.2%
        },
        _ => {
            // 横盘
            noise.clamp(-1.0, 1.0)
        }
    };
    
    // 应用A股涨跌停限制
    let final_change = clamp_daily_change(predicted_change);
    let confidence_adj = confidence * time_decay;
    
    (final_change, confidence_adj)
} 