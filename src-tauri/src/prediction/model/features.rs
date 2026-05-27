//! 特征工程
//!
//! 从历史数据构造机器学习模型的输入特征与标签。特征显式包含**量比**与**换手率**。
//! 标签为次日收益率（回归目标），其符号即方向。

use crate::db::models::HistoricalData;

/// 特征维度
pub const FEATURE_DIM: usize = 10;
/// 特征所需的最小回看窗口
const LOOKBACK: usize = 20;

/// 各特征名称（用于模型元数据）
pub fn feature_names() -> Vec<String> {
    [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ma5_ma20_ratio",
        "rsi14",
        "volatility10",
        "volume_ratio",
        "turnover_rate",
        "range_position20",
        "amplitude",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// 计算索引 `i` 处的特征向量（仅使用 ≤ i 的数据）
fn features_at(h: &[HistoricalData], i: usize) -> [f32; FEATURE_DIM] {
    let close = |k: usize| h[k].close;
    let ret = |a: usize, b: usize| -> f32 {
        if close(b) > 0.0 {
            ((close(a) - close(b)) / close(b)) as f32
        } else {
            0.0
        }
    };

    let ret_1 = ret(i, i - 1);
    let ret_5 = ret(i, i - 5);
    let ret_10 = ret(i, i - 10);

    let ma5 = (i - 4..=i).map(close).sum::<f64>() / 5.0;
    let ma20 = (i - 19..=i).map(close).sum::<f64>() / 20.0;
    let ma_ratio = if ma20 > 0.0 {
        ((ma5 / ma20) - 1.0) as f32
    } else {
        0.0
    };

    // RSI(14)：基于最近14日涨跌
    let rsi = {
        let (mut gain, mut loss) = (0.0f64, 0.0f64);
        for k in (i - 13)..=i {
            let diff = close(k) - close(k - 1);
            if diff >= 0.0 {
                gain += diff;
            } else {
                loss -= diff;
            }
        }
        if gain + loss > 0.0 {
            (gain / (gain + loss) * 100.0) as f32
        } else {
            50.0
        }
    };

    // 10日收益率波动
    let volatility = {
        let rets: Vec<f64> = (i - 9..=i)
            .map(|k| {
                if close(k - 1) > 0.0 {
                    (close(k) - close(k - 1)) / close(k - 1)
                } else {
                    0.0
                }
            })
            .collect();
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rets.len() as f64;
        var.sqrt() as f32
    };

    // 量比：当日成交量 / 过去5日均量
    let vavg5 = (i - 4..=i).map(|k| h[k].volume as f64).sum::<f64>() / 5.0;
    let volume_ratio = if vavg5 > 0.0 {
        (h[i].volume as f64 / vavg5) as f32
    } else {
        1.0
    };

    // 换手率（缩放）
    let turnover = (h[i].turnover_rate / 10.0) as f32;

    // 20日区间位置
    let max20 = (i - 19..=i).map(|k| h[k].high).fold(f64::MIN, f64::max);
    let min20 = (i - 19..=i).map(|k| h[k].low).fold(f64::MAX, f64::min);
    let pos = if max20 > min20 {
        ((close(i) - min20) / (max20 - min20)) as f32 - 0.5
    } else {
        0.0
    };

    let amplitude = (h[i].amplitude / 10.0) as f32;

    // 确定性量纲缩放：把各特征拉到 ~±1 同一量级，避免收益率被量比淹没。
    // 训练与推理使用同一变换，无需存储统计量。
    [
        ret_1 * 25.0,
        ret_5 * 12.0,
        ret_10 * 8.0,
        ma_ratio * 15.0,
        ((rsi / 100.0) - 0.5) * 2.0,
        volatility * 30.0,
        (volume_ratio - 1.0).clamp(-2.0, 3.0),
        turnover, // 换手率/10，缺数据时为 0
        pos * 2.0,
        amplitude * 2.0,
    ]
}

/// 构造训练数据集：返回 (扁平特征 n×FEATURE_DIM, 标签 n, 样本数 n)。
///
/// 标签为次日收益率（%），符号即涨跌方向。
pub fn build_dataset(historical: &[HistoricalData]) -> (Vec<f32>, Vec<f32>, usize) {
    let len = historical.len();
    if len < LOOKBACK + 2 {
        return (Vec::new(), Vec::new(), 0);
    }

    let mut features = Vec::new();
    let mut labels = Vec::new();
    // i 从 LOOKBACK 到 len-2（需要 i+1 作为标签）
    for i in LOOKBACK..(len - 1) {
        let feat = features_at(historical, i);
        let base = historical[i].close;
        if base <= 0.0 {
            continue;
        }
        let next_ret = ((historical[i + 1].close - base) / base * 100.0) as f32;
        features.extend_from_slice(&feat);
        labels.push(next_ret);
    }
    let n = labels.len();
    (features, labels, n)
}

/// 带日期的样本（用于截面相对强弱建模）
pub struct DatedSample {
    pub date: chrono::NaiveDate,
    pub features: [f32; FEATURE_DIM],
    /// 未来 horizon 日收益率（小数）
    pub fwd_return: f64,
}

/// 构造带日期、带未来 horizon 日收益的样本序列（用于跨股票按日对齐做截面去均值）。
pub fn build_samples(historical: &[HistoricalData], horizon: usize) -> Vec<DatedSample> {
    let len = historical.len();
    let h = horizon.max(1);
    if len < LOOKBACK + h + 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in LOOKBACK..(len - h) {
        let base = historical[i].close;
        if base <= 0.0 {
            continue;
        }
        let fwd = (historical[i + h].close - base) / base;
        out.push(DatedSample {
            date: historical[i].date,
            features: features_at(historical, i),
            fwd_return: fwd,
        });
    }
    out
}

/// 提取最新一个交易日（无标签）的特征向量，用于实时预测。
pub fn latest_features(historical: &[HistoricalData]) -> Option<Vec<f32>> {
    let len = historical.len();
    if len < LOOKBACK + 1 {
        return None;
    }
    Some(features_at(historical, len - 1).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make(n: usize) -> Vec<HistoricalData> {
        (0..n)
            .map(|k| HistoricalData {
                symbol: "test".to_string(),
                date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap() + chrono::Duration::days(k as i64),
                open: 10.0 + k as f64 * 0.1,
                close: 10.0 + k as f64 * 0.1,
                high: 10.5 + k as f64 * 0.1,
                low: 9.5 + k as f64 * 0.1,
                volume: 1000 + (k as i64 % 7) * 100,
                amount: 10000.0,
                amplitude: 2.0,
                turnover_rate: 3.0,
                volume_ratio: 1.0,
                change_percent: 1.0,
                change: 0.1,
            })
            .collect()
    }

    #[test]
    fn test_build_dataset_shape() {
        let h = make(60);
        let (features, labels, n) = build_dataset(&h);
        assert_eq!(n, labels.len());
        assert_eq!(features.len(), n * FEATURE_DIM);
        assert!(n > 0);
    }

    #[test]
    fn test_insufficient_data() {
        let h = make(10);
        let (_, _, n) = build_dataset(&h);
        assert_eq!(n, 0);
    }
}
