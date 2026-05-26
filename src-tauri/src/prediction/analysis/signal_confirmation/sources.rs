//! 收集各技术指标信号

use super::{SignalSource, SignalType};
use crate::prediction::analysis::market_regime::MarketRegime;
use crate::prediction::analysis::{TrendState, VolumePriceSignal};
use crate::prediction::indicators::TechnicalIndicatorValues;

/// 收集所有技术信号
pub(super) fn collect_all_signals(
    ind: &TechnicalIndicatorValues,
    trend: &TrendState,
    volume: &VolumePriceSignal,
    regime: &MarketRegime,
) -> Vec<SignalSource> {
    let mut signals = Vec::new();

    // 趋势信号
    let trend_direction = match trend {
        TrendState::StrongBullish => 1.0,
        TrendState::Bullish => 0.6,
        TrendState::Neutral => 0.0,
        TrendState::Bearish => -0.6,
        TrendState::StrongBearish => -1.0,
    };
    signals.push(SignalSource {
        name: "趋势".to_string(),
        direction: trend_direction,
        strength: trend_direction.abs(),
        reliability: 0.85,
        signal_type: SignalType::Trend,
        regime_aligned: regime.is_trending() == trend.is_bullish()
            || regime.is_trending() == trend.is_bearish(),
    });

    // MACD 信号
    let macd_direction = if ind.macd_golden_cross {
        0.8
    } else if ind.macd_death_cross {
        -0.8
    } else if ind.macd_histogram > 0.0 {
        0.4
    } else if ind.macd_histogram < 0.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "MACD".to_string(),
        direction: macd_direction,
        strength: macd_direction.abs(),
        reliability: 0.80,
        signal_type: SignalType::Momentum,
        regime_aligned: true,
    });

    // RSI 信号（逆向指标在极端区域）
    let rsi_direction = if ind.rsi < 25.0 {
        0.7 // 极度超卖，看涨
    } else if ind.rsi < 35.0 {
        0.4
    } else if ind.rsi > 75.0 {
        -0.7 // 极度超买，看跌
    } else if ind.rsi > 65.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "RSI".to_string(),
        direction: rsi_direction,
        strength: rsi_direction.abs(),
        reliability: 0.75,
        signal_type: SignalType::Oscillator,
        regime_aligned: !regime.is_trending() || rsi_direction.abs() < 0.5,
    });

    // KDJ 信号
    let kdj_direction = if ind.kdj_golden_cross && ind.kdj_oversold {
        0.85
    } else if ind.kdj_golden_cross {
        0.5
    } else if ind.kdj_death_cross && ind.kdj_overbought {
        -0.85
    } else if ind.kdj_death_cross {
        -0.5
    } else if ind.kdj_j < 20.0 {
        0.6
    } else if ind.kdj_j > 80.0 {
        -0.6
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "KDJ".to_string(),
        direction: kdj_direction,
        strength: kdj_direction.abs(),
        reliability: 0.70,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });

    // Williams %R 信号
    let wr_direction = if ind.williams_oversold {
        0.65 // 超卖看涨
    } else if ind.williams_overbought {
        -0.65 // 超买看跌
    } else if ind.williams_r < -70.0 {
        0.35
    } else if ind.williams_r > -30.0 {
        -0.35
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "Williams%R".to_string(),
        direction: wr_direction,
        strength: wr_direction.abs(),
        reliability: 0.68,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });

    // ROC 动量信号
    let roc_direction = if ind.roc > 8.0 {
        0.7
    } else if ind.roc > 3.0 {
        0.4
    } else if ind.roc < -8.0 {
        -0.7
    } else if ind.roc < -3.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "ROC".to_string(),
        direction: roc_direction,
        strength: roc_direction.abs(),
        reliability: 0.72,
        signal_type: SignalType::Momentum,
        regime_aligned: regime.is_trending(),
    });

    // CCI 信号
    let cci_direction = if ind.cci < -200.0 {
        0.75
    } else if ind.cci < -100.0 {
        0.4
    } else if ind.cci > 200.0 {
        -0.75
    } else if ind.cci > 100.0 {
        -0.4
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "CCI".to_string(),
        direction: cci_direction,
        strength: cci_direction.abs(),
        reliability: 0.65,
        signal_type: SignalType::Oscillator,
        regime_aligned: true,
    });

    // 量能信号
    let vol_direction = match volume.direction.as_str() {
        "上涨" => volume.confidence * 0.8,
        "下跌" => -volume.confidence * 0.8,
        _ => 0.0,
    };
    signals.push(SignalSource {
        name: "量价".to_string(),
        direction: vol_direction,
        strength: vol_direction.abs(),
        reliability: 0.78,
        signal_type: SignalType::Volume,
        regime_aligned: true,
    });

    // BRAR 人气信号
    let brar_direction = if ind.br > 150.0 && ind.ar > 130.0 {
        -0.5 // 过热
    } else if ind.br < 50.0 || ind.ar < 70.0 {
        0.5 // 超卖
    } else if ind.br > ind.ar && ind.br > 100.0 {
        0.3
    } else if ind.br < ind.ar && ind.br < 100.0 {
        -0.3
    } else {
        0.0
    };
    signals.push(SignalSource {
        name: "BRAR".to_string(),
        direction: brar_direction,
        strength: brar_direction.abs(),
        reliability: 0.60,
        signal_type: SignalType::Sentiment,
        regime_aligned: true,
    });

    signals
}
