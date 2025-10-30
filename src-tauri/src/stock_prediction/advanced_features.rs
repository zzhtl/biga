//! 高级特征工程模块
//!
//! 提供更多维度的特征提取，以提升模型预测准确性
//! 包括：
//! - 高级技术指标特征
//! - 时间序列分解特征
//! - 统计特征（偏度、峰度等）
//! - 市场微观结构特征
//! - 动量与反转特征
//! - 波动率聚类特征
//! - 相对强度特征
//! - 价量关系特征

use serde::{Deserialize, Serialize};

/// 高级特征集合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatures {
    // 动量特征
    pub momentum_features: MomentumFeatures,

    // 波动率特征
    pub volatility_features: VolatilityFeatures,

    // 统计特征
    pub statistical_features: StatisticalFeatures,

    // 价量特征
    pub price_volume_features: PriceVolumeFeatures,

    // 趋势特征
    pub trend_features: TrendFeatures,

    // 市场结构特征
    pub market_structure_features: MarketStructureFeatures,
}

/// 动量特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumFeatures {
    pub roc_5: f64,          // 5日变化率
    pub roc_10: f64,         // 10日变化率
    pub roc_20: f64,         // 20日变化率
    pub momentum_7: f64,     // 7日动量
    pub momentum_14: f64,    // 14日动量
    pub momentum_28: f64,    // 28日动量
    pub rsi_divergence: f64, // RSI背离度
    pub macd_momentum: f64,  // MACD动量
    pub acceleration: f64,   // 价格加速度
    pub jerk: f64,           // 价格急动度（三阶导数）
}

/// 波动率特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityFeatures {
    pub realized_volatility: f64,     // 已实现波动率
    pub parkinson_volatility: f64,    // Parkinson波动率（基于高低价）
    pub garman_klass_volatility: f64, // Garman-Klass波动率
    pub volatility_ratio: f64,        // 波动率比率（短期/长期）
    pub volatility_trend: f64,        // 波动率趋势
    pub volatility_persistence: f64,  // 波动率持续性
    pub atr_normalized: f64,          // 标准化ATR
    pub volatility_regime: i8,        // 波动率状态：-1低，0中，1高
}

/// 统计特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatures {
    pub returns_mean: f64,      // 收益率均值
    pub returns_std: f64,       // 收益率标准差
    pub returns_skewness: f64,  // 收益率偏度
    pub returns_kurtosis: f64,  // 收益率峰度
    pub hurst_exponent: f64,    // Hurst指数（趋势强度）
    pub autocorr_1: f64,        // 1阶自相关
    pub autocorr_5: f64,        // 5阶自相关
    pub entropy: f64,           // 信息熵
    pub fractal_dimension: f64, // 分形维数
}

/// 价量特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceVolumeFeatures {
    pub vwap: f64,                      // 成交量加权平均价
    pub volume_price_correlation: f64,  // 价量相关性
    pub volume_trend: f64,              // 成交量趋势
    pub volume_breakout: f64,           // 成交量突破信号
    pub buying_pressure: f64,           // 买盘压力
    pub selling_pressure: f64,          // 卖盘压力
    pub volume_momentum: f64,           // 成交量动量
    pub money_flow_index: f64,          // 资金流量指标
    pub accumulation_distribution: f64, // 累积/派发指标
}

/// 趋势特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendFeatures {
    pub trend_strength: f64,      // 趋势强度
    pub trend_consistency: f64,   // 趋势一致性
    pub adx: f64,                 // 平均趋向指标
    pub ma_alignment: f64,        // 均线排列度
    pub price_position: f64,      // 价格相对位置（布林带）
    pub support_distance: f64,    // 距离支撑位
    pub resistance_distance: f64, // 距离阻力位
    pub channel_position: f64,    // 通道位置
}

/// 市场结构特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketStructureFeatures {
    pub higher_highs: i32,        // 连续更高高点数
    pub lower_lows: i32,          // 连续更低低点数
    pub swing_strength: f64,      // 摆动强度
    pub range_expansion: f64,     // 区间扩张度
    pub gap_ratio: f64,           // 跳空比例
    pub tail_ratio: f64,          // 上下影线比例
    pub body_ratio: f64,          // 实体比例
    pub consolidation_score: f64, // 盘整得分
}

/// 计算高级特征
pub fn calculate_advanced_features(
    prices: &[f64],
    volumes: &[i64],
    highs: &[f64],
    lows: &[f64],
    opens: &[f64],
) -> AdvancedFeatures {
    AdvancedFeatures {
        momentum_features: calculate_momentum_features(prices),
        volatility_features: calculate_volatility_features(prices, highs, lows),
        statistical_features: calculate_statistical_features(prices),
        price_volume_features: calculate_price_volume_features(prices, volumes, highs, lows),
        trend_features: calculate_trend_features(prices, highs, lows),
        market_structure_features: calculate_market_structure_features(prices, highs, lows, opens),
    }
}

/// 计算动量特征
fn calculate_momentum_features(prices: &[f64]) -> MomentumFeatures {
    let n = prices.len();

    // ROC（变化率）
    let roc_5 = if n >= 6 {
        (prices[n - 1] - prices[n - 6]) / prices[n - 6]
    } else {
        0.0
    };

    let roc_10 = if n >= 11 {
        (prices[n - 1] - prices[n - 11]) / prices[n - 11]
    } else {
        0.0
    };

    let roc_20 = if n >= 21 {
        (prices[n - 1] - prices[n - 21]) / prices[n - 21]
    } else {
        0.0
    };

    // 动量（Momentum）
    let momentum_7 = if n >= 8 {
        prices[n - 1] - prices[n - 8]
    } else {
        0.0
    };

    let momentum_14 = if n >= 15 {
        prices[n - 1] - prices[n - 15]
    } else {
        0.0
    };

    let momentum_28 = if n >= 29 {
        prices[n - 1] - prices[n - 29]
    } else {
        0.0
    };

    // 计算价格的一阶和二阶导数（加速度）
    let acceleration = if n >= 3 {
        let v1 = prices[n - 1] - prices[n - 2];
        let v2 = prices[n - 2] - prices[n - 3];
        v1 - v2
    } else {
        0.0
    };

    // 三阶导数（急动度）
    let jerk = if n >= 4 {
        let a1 = prices[n - 1] - 2.0 * prices[n - 2] + prices[n - 3];
        let a2 = prices[n - 2] - 2.0 * prices[n - 3] + prices[n - 4];
        a1 - a2
    } else {
        0.0
    };

    MomentumFeatures {
        roc_5,
        roc_10,
        roc_20,
        momentum_7,
        momentum_14,
        momentum_28,
        rsi_divergence: 0.0, // 简化处理
        macd_momentum: 0.0,  // 简化处理
        acceleration: acceleration / prices[n - 1],
        jerk: jerk / prices[n - 1],
    }
}

/// 计算波动率特征
fn calculate_volatility_features(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
) -> VolatilityFeatures {
    let n = prices.len();
    let window = 20.min(n);

    // 已实现波动率（基于收益率标准差）
    let realized_volatility = if n >= 2 {
        let returns: Vec<f64> = (1..window)
            .map(|i| {
                let idx = n - window + i;
                (prices[idx] / prices[idx - 1]).ln()
            })
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt() * (252.0_f64).sqrt() // 年化
    } else {
        0.0
    };

    // Parkinson波动率（基于高低价，更高效）
    let parkinson_volatility = if n >= window && highs.len() >= n && lows.len() >= n {
        let hl_ratios: Vec<f64> = (n - window..n)
            .map(|i| ((highs[i] / lows[i]).ln()).powi(2))
            .collect();

        let sum: f64 = hl_ratios.iter().sum();
        (sum / (4.0 * window as f64 * (2.0_f64).ln())).sqrt() * (252.0_f64).sqrt()
    } else {
        realized_volatility
    };

    // Garman-Klass波动率（考虑开盘价和收盘价）
    let garman_klass_volatility = parkinson_volatility * 1.1; // 简化估算

    // 波动率比率（短期/长期）
    let short_vol = if n >= 6 {
        let returns: Vec<f64> = (1..6)
            .map(|i| (prices[n - i] / prices[n - i - 1]).ln())
            .collect();
        let mean = returns.iter().sum::<f64>() / 5.0;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / 5.0;
        variance.sqrt()
    } else {
        realized_volatility / (252.0_f64).sqrt()
    };

    let long_vol = realized_volatility / (252.0_f64).sqrt();
    let volatility_ratio = if long_vol > 1e-8 {
        short_vol / long_vol
    } else {
        1.0
    };

    // 波动率趋势（波动率是在上升还是下降）
    let volatility_trend = if n >= 40 {
        let recent_vol = calculate_period_volatility(&prices[n - 20..n]);
        let past_vol = calculate_period_volatility(&prices[n - 40..n - 20]);
        if past_vol > 1e-8 {
            (recent_vol - past_vol) / past_vol
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 标准化ATR
    let atr_normalized = if n >= 14 && highs.len() >= n && lows.len() >= n {
        let atr = calculate_atr_simple(highs, lows, prices, 14);
        atr / prices[n - 1]
    } else {
        0.02 // 默认2%
    };

    // 波动率状态
    let volatility_regime = if atr_normalized > 0.03 {
        1 // 高波动
    } else if atr_normalized < 0.015 {
        -1 // 低波动
    } else {
        0 // 中等波动
    };

    VolatilityFeatures {
        realized_volatility,
        parkinson_volatility,
        garman_klass_volatility,
        volatility_ratio,
        volatility_trend,
        volatility_persistence: 0.5, // 简化处理
        atr_normalized,
        volatility_regime,
    }
}

/// 计算统计特征
fn calculate_statistical_features(prices: &[f64]) -> StatisticalFeatures {
    let n = prices.len();
    let window = 30.min(n);

    // 计算收益率
    let returns: Vec<f64> = if n >= 2 {
        (1..window)
            .map(|i| {
                let idx = n - window + i;
                (prices[idx] - prices[idx - 1]) / prices[idx - 1]
            })
            .collect()
    } else {
        vec![0.0]
    };

    // 均值
    let returns_mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // 标准差
    let variance = returns
        .iter()
        .map(|r| (r - returns_mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let returns_std = variance.sqrt();

    // 偏度（Skewness）
    let returns_skewness = if returns_std > 1e-8 {
        let skew_sum = returns
            .iter()
            .map(|r| ((r - returns_mean) / returns_std).powi(3))
            .sum::<f64>();
        skew_sum / returns.len() as f64
    } else {
        0.0
    };

    // 峰度（Kurtosis）
    let returns_kurtosis = if returns_std > 1e-8 {
        let kurt_sum = returns
            .iter()
            .map(|r| ((r - returns_mean) / returns_std).powi(4))
            .sum::<f64>();
        kurt_sum / returns.len() as f64 - 3.0 // 超额峰度
    } else {
        0.0
    };

    // Hurst指数（简化版R/S分析）
    let hurst_exponent = calculate_hurst_exponent(prices);

    // 自相关
    let autocorr_1 = calculate_autocorrelation(&returns, 1);
    let autocorr_5 = calculate_autocorrelation(&returns, 5);

    // 信息熵（简化计算）
    let entropy = calculate_entropy(&returns);

    // 分形维数（Fractal Dimension）
    let fractal_dimension = 2.0 - hurst_exponent;

    StatisticalFeatures {
        returns_mean,
        returns_std,
        returns_skewness,
        returns_kurtosis,
        hurst_exponent,
        autocorr_1,
        autocorr_5,
        entropy,
        fractal_dimension,
    }
}

/// 计算价量特征
fn calculate_price_volume_features(
    prices: &[f64],
    volumes: &[i64],
    highs: &[f64],
    lows: &[f64],
) -> PriceVolumeFeatures {
    let n = prices.len();
    let window = 20.min(n);

    // VWAP（成交量加权平均价）
    let vwap = if n >= window && volumes.len() >= n {
        let total_pv: f64 = (n - window..n).map(|i| prices[i] * volumes[i] as f64).sum();
        let total_v: i64 = volumes[n - window..n].iter().sum();
        if total_v > 0 {
            total_pv / total_v as f64
        } else {
            prices[n - 1]
        }
    } else {
        prices[n - 1]
    };

    // 价量相关性
    let volume_price_correlation = if n >= window && volumes.len() >= n {
        let price_returns: Vec<f64> = (1..window)
            .map(|i| {
                let idx = n - window + i;
                (prices[idx] - prices[idx - 1]) / prices[idx - 1]
            })
            .collect();

        let vol_changes: Vec<f64> = (1..window)
            .map(|i| {
                let idx = n - window + i;
                let v1 = volumes[idx] as f64;
                let v2 = volumes[idx - 1] as f64;
                if v2 > 0.0 {
                    (v1 - v2) / v2
                } else {
                    0.0
                }
            })
            .collect();

        calculate_correlation(&price_returns, &vol_changes)
    } else {
        0.0
    };

    // 成交量趋势
    let volume_trend = if n >= 20 && volumes.len() >= n {
        let recent_avg: f64 = volumes[n - 10..n].iter().map(|&v| v as f64).sum::<f64>() / 10.0;
        let past_avg: f64 = volumes[n - 20..n - 10]
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>()
            / 10.0;
        if past_avg > 0.0 {
            (recent_avg - past_avg) / past_avg
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 成交量突破
    let volume_breakout = if n >= 20 && volumes.len() >= n {
        let avg_vol: f64 = volumes[n - 20..n - 1]
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>()
            / 19.0;
        let current_vol = volumes[n - 1] as f64;
        if avg_vol > 0.0 {
            (current_vol - avg_vol) / avg_vol
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 买卖压力（简化计算）
    let (buying_pressure, selling_pressure) = if n >= 1 && highs.len() >= n && lows.len() >= n {
        let close = prices[n - 1];
        let high = highs[n - 1];
        let low = lows[n - 1];
        let range = high - low;

        if range > 1e-8 {
            let buying = (close - low) / range;
            let selling = (high - close) / range;
            (buying, selling)
        } else {
            (0.5, 0.5)
        }
    } else {
        (0.5, 0.5)
    };

    // 资金流量指标（MFI）简化版
    let money_flow_index = calculate_mfi_simple(prices, volumes, highs, lows);

    PriceVolumeFeatures {
        vwap,
        volume_price_correlation,
        volume_trend,
        volume_breakout,
        buying_pressure,
        selling_pressure,
        volume_momentum: volume_trend,
        money_flow_index,
        accumulation_distribution: 0.0, // 简化处理
    }
}

/// 计算趋势特征
fn calculate_trend_features(prices: &[f64], highs: &[f64], lows: &[f64]) -> TrendFeatures {
    let n = prices.len();

    // 趋势强度（基于线性回归R²）
    let trend_strength = if n >= 20 {
        let recent_prices = &prices[n - 20..n];
        calculate_linear_regression_r2(recent_prices)
    } else {
        0.5
    };

    // 趋势一致性（连续同向变化）
    let trend_consistency = if n >= 10 {
        let mut consistent_moves = 0;
        let mut total_moves = 0;
        let mut prev_direction = 0;

        for i in n - 9..n {
            let direction = if prices[i] > prices[i - 1] {
                1
            } else if prices[i] < prices[i - 1] {
                -1
            } else {
                0
            };

            if i > n - 9 && direction == prev_direction && direction != 0 {
                consistent_moves += 1;
            }
            if direction != 0 {
                total_moves += 1;
            }
            prev_direction = direction;
        }

        if total_moves > 0 {
            consistent_moves as f64 / total_moves as f64
        } else {
            0.5
        }
    } else {
        0.5
    };

    // ADX（简化计算）
    let adx = trend_strength * 100.0;

    // 均线排列度
    let ma_alignment = if n >= 60 {
        let ma5 = prices[n - 5..n].iter().sum::<f64>() / 5.0;
        let ma10 = prices[n - 10..n].iter().sum::<f64>() / 10.0;
        let ma20 = prices[n - 20..n].iter().sum::<f64>() / 20.0;
        let ma60 = prices[n - 60..n].iter().sum::<f64>() / 60.0;

        let mut score = 0.0;
        if ma5 > ma10 {
            score += 0.25;
        }
        if ma10 > ma20 {
            score += 0.25;
        }
        if ma20 > ma60 {
            score += 0.25;
        }
        if prices[n - 1] > ma5 {
            score += 0.25;
        }

        score * 2.0 - 1.0 // 转换到[-1, 1]
    } else {
        0.0
    };

    // 价格相对位置（布林带）
    let price_position = if n >= 20 {
        let ma20 = prices[n - 20..n].iter().sum::<f64>() / 20.0;
        let variance = prices[n - 20..n]
            .iter()
            .map(|p| (p - ma20).powi(2))
            .sum::<f64>()
            / 20.0;
        let std_dev = variance.sqrt();

        if std_dev > 1e-8 {
            (prices[n - 1] - ma20) / (2.0 * std_dev)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 距离支撑位和阻力位
    let (support_distance, resistance_distance) = if n >= 20 {
        let recent_lows: Vec<f64> = lows[n - 20..n].to_vec();
        let recent_highs: Vec<f64> = highs[n - 20..n].to_vec();

        let support = recent_lows.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let resistance = recent_highs
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let current = prices[n - 1];
        let sup_dist = (current - support) / current;
        let res_dist = (resistance - current) / current;

        (sup_dist, res_dist)
    } else {
        (0.05, 0.05)
    };

    TrendFeatures {
        trend_strength,
        trend_consistency,
        adx,
        ma_alignment,
        price_position,
        support_distance,
        resistance_distance,
        channel_position: price_position,
    }
}

/// 计算市场结构特征
fn calculate_market_structure_features(
    prices: &[f64],
    highs: &[f64],
    lows: &[f64],
    opens: &[f64],
) -> MarketStructureFeatures {
    let n = prices.len();

    // 连续更高高点和更低低点
    let (higher_highs, lower_lows) = if n >= 10 && highs.len() >= n && lows.len() >= n {
        let mut hh = 0;
        let mut ll = 0;

        for i in n - 9..n {
            if highs[i] > highs[i - 1] {
                hh += 1;
            }
            if lows[i] < lows[i - 1] {
                ll += 1;
            }
        }

        (hh, ll)
    } else {
        (0, 0)
    };

    // 摆动强度
    let swing_strength = if n >= 5 && highs.len() >= n && lows.len() >= n {
        let recent_range: f64 = (n - 5..n).map(|i| highs[i] - lows[i]).sum::<f64>() / 5.0;

        recent_range / prices[n - 1]
    } else {
        0.02
    };

    // 区间扩张度
    let range_expansion = if n >= 10 && highs.len() >= n && lows.len() >= n {
        let recent_range = (highs[n - 5..n]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - lows[n - 5..n].iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        let past_range = (highs[n - 10..n - 5]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - lows[n - 10..n - 5]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b)));

        if past_range > 1e-8 {
            (recent_range - past_range) / past_range
        } else {
            0.0
        }
    } else {
        0.0
    };

    // 跳空比例
    let gap_ratio = if n >= 10 && opens.len() >= n {
        let mut gaps = 0;
        for i in n - 9..n {
            let gap = (opens[i] - prices[i - 1]).abs() / prices[i - 1];
            if gap > 0.02 {
                // 超过2%算跳空
                gaps += 1;
            }
        }
        gaps as f64 / 9.0
    } else {
        0.0
    };

    // 上下影线比例
    let (tail_ratio, body_ratio) =
        if n >= 1 && opens.len() >= n && highs.len() >= n && lows.len() >= n {
            let open = opens[n - 1];
            let close = prices[n - 1];
            let high = highs[n - 1];
            let low = lows[n - 1];

            let body = (close - open).abs();
            let upper_shadow = high - open.max(close);
            let lower_shadow = open.min(close) - low;
            let total_range = high - low;

            if total_range > 1e-8 {
                let tail = (upper_shadow + lower_shadow) / total_range;
                let body_pct = body / total_range;
                (tail, body_pct)
            } else {
                (0.5, 0.5)
            }
        } else {
            (0.5, 0.5)
        };

    // 盘整得分（低波动 + 窄幅震荡）
    let consolidation_score = if n >= 20 {
        let recent_std = calculate_period_volatility(&prices[n - 20..n]);
        let price_range = prices[n - 20..n]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - prices[n - 20..n]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
        let range_pct = price_range / prices[n - 1];

        // 波动小且区间窄说明在盘整
        if recent_std < 0.015 && range_pct < 0.05 {
            0.8
        } else if recent_std < 0.025 && range_pct < 0.1 {
            0.5
        } else {
            0.2
        }
    } else {
        0.5
    };

    MarketStructureFeatures {
        higher_highs,
        lower_lows,
        swing_strength,
        range_expansion,
        gap_ratio,
        tail_ratio,
        body_ratio,
        consolidation_score,
    }
}

// ========== 辅助函数 ==========

/// 计算周期波动率
fn calculate_period_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = (1..prices.len())
        .map(|i| (prices[i] - prices[i - 1]) / prices[i - 1])
        .collect();

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}

/// 计算Hurst指数（简化版R/S分析）
fn calculate_hurst_exponent(prices: &[f64]) -> f64 {
    if prices.len() < 20 {
        return 0.5; // 默认随机游走
    }

    let n = prices.len().min(100);
    let data = &prices[prices.len() - n..];

    // 计算对数收益率
    let returns: Vec<f64> = (1..data.len())
        .map(|i| (data[i] / data[i - 1]).ln())
        .collect();

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // 计算累积离差
    let mut cumulative_dev = 0.0;
    let mut max_dev = f64::NEG_INFINITY;
    let mut min_dev = f64::INFINITY;

    for ret in &returns {
        cumulative_dev += ret - mean;
        max_dev = max_dev.max(cumulative_dev);
        min_dev = min_dev.min(cumulative_dev);
    }

    let range = max_dev - min_dev;
    let std_dev = calculate_period_volatility(data);

    // R/S值
    let rs = if std_dev > 1e-8 { range / std_dev } else { 1.0 };

    // Hurst = log(R/S) / log(n)
    let hurst = if rs > 0.0 && n > 1 {
        rs.ln() / (n as f64).ln()
    } else {
        0.5
    };

    hurst.clamp(0.0, 1.0)
}

/// 计算自相关系数
fn calculate_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    if variance < 1e-8 {
        return 0.0;
    }

    let covariance = (lag..data.len())
        .map(|i| (data[i] - mean) * (data[i - lag] - mean))
        .sum::<f64>()
        / (data.len() - lag) as f64;

    covariance / variance
}

/// 计算信息熵
fn calculate_entropy(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // 将数据分箱
    let bins = 10;
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    if range < 1e-8 {
        return 0.0;
    }

    let bin_width = range / bins as f64;
    let mut counts = vec![0; bins];

    for &value in data {
        let bin = ((value - min_val) / bin_width).floor() as usize;
        let bin = bin.min(bins - 1);
        counts[bin] += 1;
    }

    // 计算熵
    let total = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// 计算两个序列的相关系数
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-8 || var_y < 1e-8 {
        return 0.0;
    }

    cov / (var_x * var_y).sqrt()
}

/// 计算ATR（简化版）
fn calculate_atr_simple(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if highs.len() < period || lows.len() < period || closes.len() < period {
        return 0.0;
    }

    let n = highs.len();
    let mut trs = Vec::new();

    for i in (n - period)..n {
        let high_low = highs[i] - lows[i];
        let high_close = if i > 0 {
            (highs[i] - closes[i - 1]).abs()
        } else {
            high_low
        };
        let low_close = if i > 0 {
            (lows[i] - closes[i - 1]).abs()
        } else {
            high_low
        };

        let tr = high_low.max(high_close).max(low_close);
        trs.push(tr);
    }

    trs.iter().sum::<f64>() / trs.len() as f64
}

/// 计算资金流量指标（MFI）简化版
fn calculate_mfi_simple(prices: &[f64], volumes: &[i64], highs: &[f64], lows: &[f64]) -> f64 {
    let n = prices.len();
    if n < 14 || volumes.len() < n || highs.len() < n || lows.len() < n {
        return 50.0; // 默认中性
    }

    let mut positive_flow = 0.0;
    let mut negative_flow = 0.0;

    for i in (n - 14)..n {
        if i == 0 {
            continue;
        }

        let typical_price = (highs[i] + lows[i] + prices[i]) / 3.0;
        let prev_typical_price = (highs[i - 1] + lows[i - 1] + prices[i - 1]) / 3.0;
        let money_flow = typical_price * volumes[i] as f64;

        if typical_price > prev_typical_price {
            positive_flow += money_flow;
        } else if typical_price < prev_typical_price {
            negative_flow += money_flow;
        }
    }

    if negative_flow < 1e-8 {
        return 100.0;
    }

    let money_ratio = positive_flow / negative_flow;
    let mfi = 100.0 - (100.0 / (1.0 + money_ratio));

    mfi.clamp(0.0, 100.0)
}

/// 计算线性回归R²
fn calculate_linear_regression_r2(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let x_mean = (data.len() - 1) as f64 / 2.0;
    let y_mean = data.iter().sum::<f64>() / n;

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    // 计算斜率
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    let slope = if denominator > 1e-8 {
        numerator / denominator
    } else {
        0.0
    };

    let intercept = y_mean - slope * x_mean;

    // 计算R²
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        let y_pred = slope * x + intercept;
        ss_res += (y - y_pred).powi(2);
        ss_tot += (y - y_mean).powi(2);
    }

    if ss_tot < 1e-8 {
        return 0.0;
    }

    let r2 = 1.0 - (ss_res / ss_tot);
    r2.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_momentum_features() {
        let prices = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
        ];
        let features = calculate_momentum_features(&prices);

        assert!(features.roc_5 > 0.0);
        assert!(features.momentum_7 > 0.0);
    }

    #[test]
    fn test_calculate_volatility_features() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0];
        let highs = vec![101.0, 102.0, 100.0, 103.0, 99.0, 104.0, 98.0, 105.0];
        let lows = vec![99.0, 100.0, 98.0, 101.0, 97.0, 102.0, 96.0, 103.0];

        let features = calculate_volatility_features(&prices, &highs, &lows);
        assert!(features.realized_volatility >= 0.0);
        assert!(features.atr_normalized > 0.0);
    }

    #[test]
    fn test_hurst_exponent() {
        let trending_prices = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        ];
        let hurst = calculate_hurst_exponent(&trending_prices);
        assert!(hurst > 0.0 && hurst <= 1.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = calculate_correlation(&x, &y);
        assert!(corr > 0.99); // 应该接近1（完全正相关）
    }
}
