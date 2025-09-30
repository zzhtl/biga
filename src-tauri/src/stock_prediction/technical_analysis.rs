use crate::stock_prediction::types::{TechnicalSignals, TradingSignal};
use crate::stock_prediction::technical_indicators::{
    calculate_macd_full, calculate_kdj, calculate_rsi, calculate_cci, calculate_obv
};

// 综合技术指标分析函数，加强MACD和KDJ策略
pub fn analyze_technical_signals(
    prices: &[f64], 
    highs: &[f64], 
    lows: &[f64], 
    volumes: &[i64]
) -> TechnicalSignals {
    let len = prices.len();
    
    // MACD计算（包括历史值用于交叉判断）
    let mut macd_dif_history = Vec::new();
    let mut macd_dea_history = Vec::new();
    let mut macd_histogram_history = Vec::new();
    
    // 计算最近30天的MACD序列，用于判断金叉死叉
    let macd_days = 30.min(len);
    for i in 0..macd_days {
        let idx = len - macd_days + i;
        if idx >= 26 {  // MACD需要至少26天数据
            let (dif, dea, histogram) = calculate_macd_full(&prices[0..=idx]);
            macd_dif_history.push(dif);
            macd_dea_history.push(dea);
            macd_histogram_history.push(histogram);
        }
    }
    
    // KDJ计算（包括历史值用于交叉判断）
    let mut kdj_k_history = Vec::new();
    let mut kdj_d_history = Vec::new();
    let mut kdj_j_history = Vec::new();
    
    // 使用引用而不是移动所有权
    let highs_ref = &highs;
    let lows_ref = &lows;
    
    // 计算最近20天的KDJ序列，用于判断金叉死叉
    let kdj_days = 20.min(len);
    for i in 0..kdj_days {
        let idx = len - kdj_days + i;
        if idx >= 9 && highs_ref.len() > idx && lows_ref.len() > idx {  // KDJ需要至少9天数据
            let start = idx.saturating_sub(8);
            let (k, d, j) = calculate_kdj(&highs_ref[start..=idx], &lows_ref[start..=idx], &prices[start..=idx], 9);
            kdj_k_history.push(k);
            kdj_d_history.push(d);
            kdj_j_history.push(j);
        }
    }
    
    // 获取最新的技术指标值
    let (macd_dif, macd_dea, macd_histogram) = if len >= 26 {
        calculate_macd_full(prices)
    } else {
        (0.0, 0.0, 0.0)
    };
    
    let (kdj_k, kdj_d, kdj_j) = if len >= 14 && highs_ref.len() >= 14 && lows_ref.len() >= 14 {
        calculate_kdj(highs_ref, lows_ref, prices, 9)
    } else {
        (50.0, 50.0, 50.0)
    };
    
    let rsi = if len >= 14 {
        calculate_rsi(&prices[len.saturating_sub(14)..])
    } else {
        50.0
    };
    
    let cci = if len >= 20 && highs_ref.len() >= 20 && lows_ref.len() >= 20 {
        calculate_cci(highs_ref, lows_ref, prices, 20)
    } else {
        0.0
    };
    
    let obv = calculate_obv(prices, volumes);
    let obv_ma = if volumes.len() >= 10 {
        let recent_obv_values: Vec<f64> = (0..10).map(|i| {
            let end = volumes.len() - i;
            calculate_obv(&prices[0..end], &volumes[0..end])
        }).collect();
        recent_obv_values.iter().sum::<f64>() / recent_obv_values.len() as f64
    } else {
        obv
    };
    
    // 增强：改进MACD交叉信号识别
    // 判断MACD交叉信号 - 标准金叉死叉判断
    let macd_golden_cross = if macd_dif_history.len() >= 2 && macd_dea_history.len() >= 2 {
        // 金叉标准定义：前一时刻DIF在DEA下方，当前时刻DIF在DEA上方
        let prev_dif = macd_dif_history[macd_dif_history.len()-2];
        let prev_dea = macd_dea_history[macd_dea_history.len()-2];
        let curr_dif = macd_dif_history[macd_dif_history.len()-1];
        let curr_dea = macd_dea_history[macd_dea_history.len()-1];
        
        // 穿越条件：前一刻DIF<=DEA，当前DIF>DEA
        prev_dif <= prev_dea && curr_dif > curr_dea
    } else {
        false
    };
    
    let macd_death_cross = if macd_dif_history.len() >= 2 && macd_dea_history.len() >= 2 {
        // 死叉标准定义：前一时刻DIF在DEA上方，当前时刻DIF在DEA下方
        let prev_dif = macd_dif_history[macd_dif_history.len()-2];
        let prev_dea = macd_dea_history[macd_dea_history.len()-2];
        let curr_dif = macd_dif_history[macd_dif_history.len()-1];
        let curr_dea = macd_dea_history[macd_dea_history.len()-1];
        
        // 穿越条件：前一刻DIF>=DEA，当前DIF<DEA
        prev_dif >= prev_dea && curr_dif < curr_dea
    } else {
        false
    };
    
    // 增强：改进MACD零轴穿越识别
    // 判断MACD零轴穿越 - 标准穿越判断
    let macd_zero_cross_up = if macd_histogram_history.len() >= 2 {
        // 零轴上穿：前一时刻柱状图为负，当前时刻为正（由负变正）
        let prev_hist = macd_histogram_history[macd_histogram_history.len()-2];
        let curr_hist = macd_histogram_history[macd_histogram_history.len()-1];
        
        prev_hist <= 0.0 && curr_hist > 0.0
    } else {
        false
    };
    
    let macd_zero_cross_down = if macd_histogram_history.len() >= 2 {
        // 零轴下穿：前一时刻柱状图为正，当前时刻为负（由正变负）
        let prev_hist = macd_histogram_history[macd_histogram_history.len()-2];
        let curr_hist = macd_histogram_history[macd_histogram_history.len()-1];
        
        prev_hist >= 0.0 && curr_hist < 0.0
    } else {
        false
    };
    
    // 增强：改进KDJ交叉信号识别
    // 判断KDJ交叉信号 - 标准金叉死叉判断
    let kdj_golden_cross = if kdj_k_history.len() >= 2 && kdj_d_history.len() >= 2 {
        // 金叉标准定义：前一时刻K在D下方，当前时刻K在D上方
        let prev_k = kdj_k_history[kdj_k_history.len()-2];
        let prev_d = kdj_d_history[kdj_d_history.len()-2];
        let curr_k = kdj_k_history[kdj_k_history.len()-1];
        let curr_d = kdj_d_history[kdj_d_history.len()-1];
        
        // 穿越条件：前一刻K<=D，当前K>D
        prev_k <= prev_d && curr_k > curr_d
    } else {
        false
    };
    
    let kdj_death_cross = if kdj_k_history.len() >= 2 && kdj_d_history.len() >= 2 {
        // 死叉标准定义：前一时刻K在D上方，当前时刻K在D下方
        let prev_k = kdj_k_history[kdj_k_history.len()-2];
        let prev_d = kdj_d_history[kdj_d_history.len()-2];
        let curr_k = kdj_k_history[kdj_k_history.len()-1];
        let curr_d = kdj_d_history[kdj_d_history.len()-1];
        
        // 穿越条件：前一刻K>=D，当前K<D
        prev_k >= prev_d && curr_k < curr_d
    } else {
        false
    };
    
    // 增强：改进KDJ超买超卖判断
    // KDJ超买超卖判断 - 增加连续性判断和更严格的条件
    let kdj_overbought = if kdj_j_history.len() >= 3 {
        // 更严格的条件：J值超过90且K、D都超过80
        let basic_condition = kdj_j > 90.0 && kdj_k > 80.0 && kdj_d > 80.0;
        
        // 连续性条件：确认是持续高位，而不是临时冲高
        let continuity_condition = kdj_j_history.len() >= 3 &&
                                 kdj_j_history[kdj_j_history.len()-2] > 80.0 &&
                                 kdj_j_history[kdj_j_history.len()-3] > 75.0;
        
        basic_condition && continuity_condition
    } else {
        kdj_j > 90.0 && kdj_k > 80.0 && kdj_d > 80.0  // 更严格的单点判断
    };
    
    let kdj_oversold = if kdj_j_history.len() >= 3 {
        // 更严格的条件：J值低于10且K、D都低于20
        let basic_condition = kdj_j < 10.0 || (kdj_k < 20.0 && kdj_d < 20.0);
        
        // 连续性条件：确认是持续低位，而不是临时下探
        let continuity_condition = kdj_j_history.len() >= 3 &&
                                 kdj_j_history[kdj_j_history.len()-2] < 20.0 &&
                                 kdj_j_history[kdj_j_history.len()-3] < 25.0;
        
        basic_condition && continuity_condition
    } else {
        kdj_j < 10.0 || (kdj_k < 20.0 && kdj_d < 20.0)  // 更严格的单点判断
    };
    
    // 生成买卖信号
    let mut buy_signals = 0;
    let mut sell_signals = 0;
    let mut signal_strength: f64 = 0.0;
    
    // MACD信号 - 加强权重
    if macd_golden_cross {
        // MACD金叉是强烈买入信号
        buy_signals += 2;
        signal_strength += 0.25;
    } else if macd_death_cross {
        // MACD死叉是强烈卖出信号
        sell_signals += 2;
        signal_strength -= 0.25;
    } else {
        // 常规MACD判断
        if macd_dif > macd_dea && macd_histogram > 0.0 {
            buy_signals += 1;
            signal_strength += 0.15;
        } else if macd_dif < macd_dea && macd_histogram < 0.0 {
            sell_signals += 1;
            signal_strength -= 0.15;
        }
    }
    
    // MACD零轴穿越 - 确认趋势信号
    if macd_zero_cross_up {
        buy_signals += 1;
        signal_strength += 0.2;
    } else if macd_zero_cross_down {
        sell_signals += 1;
        signal_strength -= 0.2;
    }
    
    // KDJ信号 - 加强权重
    if kdj_golden_cross && kdj_j < 50.0 {
        // KDJ金叉且在低位是强烈买入信号
        buy_signals += 2;
        signal_strength += 0.25;
    } else if kdj_death_cross && kdj_j > 50.0 {
        // KDJ死叉且在高位是强烈卖出信号
        sell_signals += 2;
        signal_strength -= 0.25;
    }
    
    // KDJ超买超卖
    if kdj_oversold {
        buy_signals += 1;
        signal_strength += 0.2;
    } else if kdj_overbought {
        sell_signals += 1;
        signal_strength -= 0.2;
    }
    
    // RSI信号
    if rsi < 30.0 {
        buy_signals += 1;
        signal_strength += 0.15;
    } else if rsi > 70.0 {
        sell_signals += 1;
        signal_strength -= 0.15;
    }
    
    // CCI信号
    if cci < -100.0 {
        buy_signals += 1;
        signal_strength += 0.1;
    } else if cci > 100.0 {
        sell_signals += 1;
        signal_strength -= 0.1;
    }
    
    // OBV信号
    if obv > obv_ma {
        buy_signals += 1;
        signal_strength += 0.1;
    } else if obv < obv_ma {
        sell_signals += 1;
        signal_strength -= 0.1;
    }
    
    // 增强：MACD和KDJ协同确认 - 这是最强力的信号
    if macd_golden_cross && kdj_golden_cross {
        // 双金叉，强烈买入
        buy_signals += 3;
        signal_strength += 0.3;
        
        // 如果同时出现在低位区域，进一步加强信号
        if macd_dif < 0.0 && kdj_j < 40.0 {
            buy_signals += 1;
            signal_strength += 0.1;
        }
    } else if macd_death_cross && kdj_death_cross {
        // 双死叉，强烈卖出
        sell_signals += 3;
        signal_strength -= 0.3;
        
        // 如果同时出现在高位区域，进一步加强信号
        if macd_dif > 0.0 && kdj_j > 60.0 {
            sell_signals += 1;
            signal_strength -= 0.1;
        }
    }
    
    // 增强：MACD零轴穿越与KDJ交叉结合
    if macd_zero_cross_up && kdj_golden_cross {
        buy_signals += 2;
        signal_strength += 0.25;
    } else if macd_zero_cross_down && kdj_death_cross {
        sell_signals += 2;
        signal_strength -= 0.25;
    }
    
    // 价格突破信号
    if len >= 20 {
        let ma20 = prices[len-20..].iter().sum::<f64>() / 20.0;
        if prices[len-1] > ma20 * 1.02 {
            buy_signals += 1;
            signal_strength += 0.1;
        } else if prices[len-1] < ma20 * 0.98 {
            sell_signals += 1;
            signal_strength -= 0.1;
        }
    }
    
    // 成交量信号
    if volumes.len() >= 5 {
        let vol_ma5 = volumes[volumes.len()-5..].iter().sum::<i64>() as f64 / 5.0;
        let current_vol = *volumes.last().unwrap() as f64;
        if current_vol > vol_ma5 * 1.5 && prices[len-1] > prices[len-2] {
            buy_signals += 1;
            signal_strength += 0.1;
        }
    }
    
    // 增强：计算综合信号 - 更加精细化的信号分级
    let signal = if buy_signals > sell_signals + 4 {
        // 极强买入信号：买入信号远超卖出信号
        TradingSignal::StrongBuy
    } else if buy_signals > sell_signals + 2 {
        // 强买入信号：买入信号明显超过卖出信号
        TradingSignal::Buy
    } else if sell_signals > buy_signals + 4 {
        // 极强卖出信号：卖出信号远超买入信号
        TradingSignal::StrongSell
    } else if sell_signals > buy_signals + 2 {
        // 强卖出信号：卖出信号明显超过买入信号
        TradingSignal::Sell
    } else {
        // 持有信号：买卖信号相近，市场不明朗
        TradingSignal::Hold
    };
    
    TechnicalSignals {
        macd_dif,
        macd_dea,
        macd_histogram,
        kdj_k,
        kdj_d,
        kdj_j,
        rsi,
        cci,
        obv,
        signal,
        signal_strength: signal_strength.clamp(-1.0, 1.0),
        buy_signals,
        sell_signals,
        // 交叉信号
        macd_golden_cross,
        macd_death_cross,
        kdj_golden_cross, 
        kdj_death_cross,
        kdj_overbought,
        kdj_oversold,
        macd_zero_cross_up,
        macd_zero_cross_down,
    }
} 