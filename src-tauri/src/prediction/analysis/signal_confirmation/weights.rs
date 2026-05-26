//! 动态信号权重

use super::SignalSource;
use crate::prediction::analysis::market_regime::{MarketRegime, VolatilityLevel};

/// 应用动态权重
pub(super) fn apply_dynamic_weights(
    signals: &[SignalSource],
    regime: &MarketRegime,
    volatility: &VolatilityLevel,
) -> Vec<(f64, f64)> {
    let volatility_factor = match volatility {
        VolatilityLevel::VeryHigh => 0.70,
        VolatilityLevel::High => 0.85,
        VolatilityLevel::Normal => 1.0,
        VolatilityLevel::Low => 1.05,
        VolatilityLevel::VeryLow => 0.95,
    };

    signals
        .iter()
        .map(|s| {
            let base_weight = s.signal_type.base_weight(regime);
            let regime_bonus = if s.regime_aligned { 1.1 } else { 0.9 };
            let weight = base_weight * s.reliability * regime_bonus * volatility_factor;
            (s.direction * s.strength, weight)
        })
        .collect()
}
