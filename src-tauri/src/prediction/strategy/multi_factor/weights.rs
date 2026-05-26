//! 市场状态自适应权重

use crate::config::weights::*;
use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};

/// 动态权重结构
pub(super) struct AdaptiveWeights {
    pub(super) trend: f64,
    pub(super) volume_price: f64,
    pub(super) momentum: f64,
    pub(super) pattern: f64,
    pub(super) support_resistance: f64,
    pub(super) sentiment: f64,
    pub(super) volatility: f64,
}

/// 根据市场状态获取自适应权重
pub(super) fn get_adaptive_weights(
    regime: Option<&MarketRegime>,
    volatility: Option<&VolatilityLevel>,
) -> AdaptiveWeights {
    let base = AdaptiveWeights {
        trend: TREND_FACTOR_WEIGHT,
        volume_price: VOLUME_PRICE_FACTOR_WEIGHT,
        momentum: MOMENTUM_FACTOR_WEIGHT,
        pattern: PATTERN_FACTOR_WEIGHT,
        support_resistance: SUPPORT_RESISTANCE_FACTOR_WEIGHT,
        sentiment: SENTIMENT_FACTOR_WEIGHT,
        volatility: VOLATILITY_FACTOR_WEIGHT,
    };

    // 根据市场状态调整
    match regime {
        Some(MarketRegime::StrongUptrend) | Some(MarketRegime::StrongDowntrend) => {
            // 强趋势市场：增加趋势和动量权重
            AdaptiveWeights {
                trend: base.trend * 1.4,
                volume_price: base.volume_price * 1.2,
                momentum: base.momentum * 1.3,
                pattern: base.pattern * 0.8,
                support_resistance: base.support_resistance * 0.7,
                sentiment: base.sentiment * 0.9,
                volatility: base.volatility * 0.8,
            }
        }
        Some(MarketRegime::Ranging) => {
            // 震荡市场：增加支撑阻力和形态权重
            AdaptiveWeights {
                trend: base.trend * 0.7,
                volume_price: base.volume_price * 1.1,
                momentum: base.momentum * 0.8,
                pattern: base.pattern * 1.4,
                support_resistance: base.support_resistance * 1.5,
                sentiment: base.sentiment * 1.2,
                volatility: base.volatility * 1.3,
            }
        }
        Some(MarketRegime::PotentialTop) | Some(MarketRegime::PotentialBottom) => {
            // 转折点：增加情绪和形态权重
            AdaptiveWeights {
                trend: base.trend * 0.8,
                volume_price: base.volume_price * 1.3,
                momentum: base.momentum * 1.1,
                pattern: base.pattern * 1.5,
                support_resistance: base.support_resistance * 1.3,
                sentiment: base.sentiment * 1.4,
                volatility: base.volatility * 1.2,
            }
        }
        _ => {
            // 根据波动率微调
            match volatility {
                Some(VolatilityLevel::VeryHigh) | Some(VolatilityLevel::High) => AdaptiveWeights {
                    volatility: base.volatility * 1.5,
                    support_resistance: base.support_resistance * 1.3,
                    ..base
                },
                Some(VolatilityLevel::VeryLow) | Some(VolatilityLevel::Low) => AdaptiveWeights {
                    trend: base.trend * 1.2,
                    momentum: base.momentum * 1.2,
                    ..base
                },
                _ => base,
            }
        }
    }
}
