//! 技术指标模块
//! 
//! 提供各种技术指标的计算功能

pub mod macd;
pub mod kdj;
pub mod rsi;
pub mod bollinger;
pub mod obv;
pub mod cci;
pub mod dmi;
pub mod atr;

// 选择性重导出，避免名称冲突
pub use macd::{calculate_macd, calculate_macd_full, calculate_macd_data, MacdData};
pub use macd::{is_golden_cross, is_death_cross, is_zero_cross_up, is_zero_cross_down};
pub use kdj::{calculate_kdj, calculate_kdj_data, calculate_stochastic_k, KdjData};
pub use kdj::{is_kdj_golden_cross, is_kdj_death_cross};
pub use rsi::{calculate_rsi, calculate_rsi_with_period, rsi_signal_strength};
pub use bollinger::{calculate_bollinger_bands, calculate_bollinger_position, BollingerBands};
pub use obv::calculate_obv;
pub use cci::calculate_cci;
pub use dmi::{calculate_dmi, calculate_dmi_data, DmiData};
pub use atr::calculate_atr;

use serde::{Deserialize, Serialize};

// =============================================================================
// 综合技术指标结果
// =============================================================================

/// 技术指标值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnicalIndicatorValues {
    pub rsi: f64,
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub macd_histogram: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub kdj_j: f64,
    pub cci: f64,
    pub obv_trend: f64,
    pub macd_golden_cross: bool,
    pub macd_death_cross: bool,
    pub kdj_golden_cross: bool,
    pub kdj_death_cross: bool,
    pub kdj_overbought: bool,
    pub kdj_oversold: bool,
}

impl Default for TechnicalIndicatorValues {
    fn default() -> Self {
        Self {
            rsi: 50.0,
            macd_dif: 0.0,
            macd_dea: 0.0,
            macd_histogram: 0.0,
            kdj_k: 50.0,
            kdj_d: 50.0,
            kdj_j: 50.0,
            cci: 0.0,
            obv_trend: 0.0,
            macd_golden_cross: false,
            macd_death_cross: false,
            kdj_golden_cross: false,
            kdj_death_cross: false,
            kdj_overbought: false,
            kdj_oversold: false,
        }
    }
}

/// 技术信号结构
#[derive(Debug, Clone)]
pub struct TechnicalSignals {
    pub macd_dif: f64,
    pub macd_dea: f64,
    pub macd_histogram: f64,
    pub kdj_k: f64,
    pub kdj_d: f64,
    pub kdj_j: f64,
    pub rsi: f64,
    pub cci: f64,
    pub obv: f64,
    pub signal: TradingSignal,
    pub signal_strength: f64,
    pub buy_signals: i32,
    pub sell_signals: i32,
    pub macd_golden_cross: bool,
    pub macd_death_cross: bool,
    pub kdj_golden_cross: bool,
    pub kdj_death_cross: bool,
    pub kdj_overbought: bool,
    pub kdj_oversold: bool,
    pub macd_zero_cross_up: bool,
    pub macd_zero_cross_down: bool,
}

/// 交易信号
#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

impl TradingSignal {
    pub fn to_string(&self) -> String {
        match self {
            Self::StrongBuy => "强烈买入".to_string(),
            Self::Buy => "买入".to_string(),
            Self::Hold => "持有".to_string(),
            Self::Sell => "卖出".to_string(),
            Self::StrongSell => "强烈卖出".to_string(),
        }
    }
}

// =============================================================================
// 综合计算函数
// =============================================================================

/// 计算所有技术指标
pub fn calculate_all_indicators(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    volumes: &[i64],
) -> TechnicalIndicatorValues {
    let mut result = TechnicalIndicatorValues::default();
    
    // RSI
    if prices.len() >= 15 {
        result.rsi = rsi::calculate_rsi(prices);
    }
    
    // MACD
    if prices.len() >= 26 {
        let (dif, dea, hist) = macd::calculate_macd_full(prices);
        result.macd_dif = dif;
        result.macd_dea = dea;
        result.macd_histogram = hist;
        
        // 金叉死叉判断
        let prev_macd = if prices.len() > 27 {
            let prev_prices = &prices[..prices.len() - 1];
            macd::calculate_macd_full(prev_prices)
        } else {
            (dif, dea, hist)
        };
        
        result.macd_golden_cross = prev_macd.0 <= prev_macd.1 && dif > dea;
        result.macd_death_cross = prev_macd.0 >= prev_macd.1 && dif < dea;
    }
    
    // KDJ
    if highs.len() >= 9 && lows.len() >= 9 && prices.len() >= 9 {
        let (k, d, j) = kdj::calculate_kdj(highs, lows, prices, 9);
        result.kdj_k = k;
        result.kdj_d = d;
        result.kdj_j = j;
        
        result.kdj_overbought = j > 80.0;
        result.kdj_oversold = j < 20.0;
        
        // KDJ 金叉死叉
        if highs.len() > 10 {
            let prev_len = highs.len() - 1;
            let (prev_k, prev_d, _) = kdj::calculate_kdj(
                &highs[..prev_len],
                &lows[..prev_len],
                &prices[..prev_len],
                9,
            );
            result.kdj_golden_cross = prev_k <= prev_d && k > d;
            result.kdj_death_cross = prev_k >= prev_d && k < d;
        }
    }
    
    // CCI
    if highs.len() >= 20 {
        result.cci = cci::calculate_cci(highs, lows, prices, 20);
    }
    
    // OBV 趋势
    if prices.len() >= 2 && volumes.len() >= 2 {
        let obv = obv::calculate_obv(prices, volumes);
        let avg_vol = volumes.iter().sum::<i64>() as f64 / volumes.len() as f64;
        result.obv_trend = obv / (avg_vol * volumes.len() as f64);
    }
    
    result
}

/// 计算单个特征值
pub fn calculate_feature_value(
    feature_name: &str,
    prices: &[f64],
    volumes: &[i64],
    index: usize,
    highs: Option<&[f64]>,
    lows: Option<&[f64]>,
) -> f64 {
    match feature_name {
        "close" => prices.get(index).copied().unwrap_or(0.0),
        "volume" => volumes.get(index).map(|&v| v as f64).unwrap_or(0.0),
        "change_percent" => {
            if index > 0 {
                let prev = prices[index - 1];
                if prev > 0.0 {
                    (prices[index] - prev) / prev
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        "ma5" => {
            if index >= 4 {
                prices[index - 4..=index].iter().sum::<f64>() / 5.0
            } else {
                prices[index]
            }
        }
        "ma10" => {
            if index >= 9 {
                prices[index - 9..=index].iter().sum::<f64>() / 10.0
            } else {
                prices[index]
            }
        }
        "ma20" => {
            if index >= 19 {
                prices[index - 19..=index].iter().sum::<f64>() / 20.0
            } else {
                prices[index]
            }
        }
        "rsi" => {
            if index >= 14 {
                rsi::calculate_rsi(&prices[index - 14..=index])
            } else {
                50.0
            }
        }
        "macd" => {
            if index >= 25 {
                macd::calculate_macd(&prices[index - 25..=index])
            } else {
                0.0
            }
        }
        "macd_dif" | "macd_dea" | "macd_histogram" => {
            if index >= 26 {
                let (dif, dea, hist) = macd::calculate_macd_full(&prices[..=index]);
                let current = prices[index];
                match feature_name {
                    "macd_dif" => dif / current,
                    "macd_dea" => dea / current,
                    "macd_histogram" => hist / current,
                    _ => 0.0,
                }
            } else {
                0.0
            }
        }
        "kdj_k" | "kdj_d" | "kdj_j" => {
            if let (Some(h), Some(l)) = (highs, lows) {
                if index >= 9 && h.len() > index && l.len() > index {
                    let start = index.saturating_sub(8);
                    let (k, d, j) = kdj::calculate_kdj(
                        &h[start..=index],
                        &l[start..=index],
                        &prices[start..=index],
                        9,
                    );
                    match feature_name {
                        "kdj_k" => k / 100.0,
                        "kdj_d" => d / 100.0,
                        "kdj_j" => j / 100.0,
                        _ => 0.0,
                    }
                } else {
                    0.5
                }
            } else {
                0.5
            }
        }
        "bollinger" => {
            if index >= 19 {
                bollinger::calculate_bollinger_position(&prices[index - 19..=index], prices[index])
            } else {
                0.0
            }
        }
        "cci" => {
            if let (Some(h), Some(l)) = (highs, lows) {
                if index >= 20 && h.len() > index && l.len() > index {
                    let start = index.saturating_sub(19);
                    cci::calculate_cci(&h[start..=index], &l[start..=index], &prices[start..=index], 20) / 200.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
        "obv" => {
            if index >= 1 {
                let obv_val = obv::calculate_obv(&prices[0..=index], &volumes[0..=index]);
                let avg_vol = volumes[0..=index].iter().sum::<i64>() as f64 / (index + 1) as f64;
                obv_val / (avg_vol * (index + 1) as f64)
            } else {
                0.0
            }
        }
        "momentum" => {
            if index >= 10 {
                prices[index] / prices[index - 10] - 1.0
            } else {
                0.0
            }
        }
        "stochastic_k" => {
            if index >= 13 {
                kdj::calculate_stochastic_k(&prices[index - 13..=index], prices[index])
            } else {
                0.5
            }
        }
        _ => 0.0,
    }
}

/// 获取特征所需的历史天数
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

