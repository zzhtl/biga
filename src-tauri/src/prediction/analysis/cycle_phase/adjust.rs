//! 前复权：用接口给的 `change` 还原除权因子。
//!
//! 行情源是不复权日线，10 送 10 当天价格腰斩。拿原始收盘价量回撤会凭空多出一次
//! "跌 50%"——恰好落在周期模型的下跌目标区里，是这个模型最怕的假信号。本库 131 只票
//! 2000 年以来有 325 次单日基准缩水超过 20% 的除权。
//!
//! 接口的 `change = close − pre_close`，而 `pre_close` 是**除权后**的昨收，所以
//! `close / (close − change) − 1` 就是剔除了除权缺口的真实日收益。从最新一根往回连乘，
//! 得到以最新收盘为基准的前复权序列：近期价格与行情软件的前复权显示一致，界面上的关键
//! 价位可以直接对照盘面。

use chrono::NaiveDate;

use crate::db::models::HistoricalData;

/// 单日真实收益的物理上限，超过即视为 `change` 字段异常，退回原始收盘价之比。
///
/// 主板新股首日上限 44%，其余情形都更低。创业板/科创板上市前 5 日不设涨跌幅，但那几天
/// 在周期模型的量级里可以忽略。
const MAX_ABS_DAILY_RETURN: f64 = 0.45;

/// 判定为基准错位的相对偏差。比 `discipline::facts` 的 2% 严：这里小额分红也要复权，
/// 否则长历史上的累计分红会系统性地放大回撤。
const BASIS_MISMATCH_TOLERANCE: f64 = 0.001;

/// 基准错位天数占比超过它，就判定 `change` 字段不可信、整段不复权。
///
/// 正常数据只在除权除息日错位（每年 1～2 次，占比 <1%）。`change` 缺失（恒为 0）时
/// 每个有涨跌的交易日都会"错位"，硬做复权会把整条价格线压平——宁可不复权。
const MAX_BASIS_MISMATCH_RATIO: f64 = 0.05;

/// 错位次数不超过它时一律视为可信：短序列上比例没有意义，一次除权就能占到 50%。
const TOLERATED_MISMATCHES: usize = 5;

/// 前复权后的一根 K 线。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjustedBar {
    pub date: NaiveDate,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// 前复权序列。
#[derive(Debug, Clone, PartialEq)]
pub struct AdjustedSeries {
    pub bars: Vec<AdjustedBar>,
    /// 是否实际做了复权（`change` 字段不可信时为 false，价格为原始口径）
    pub adjusted: bool,
    /// 识别到的除权除息次数
    pub corporate_actions: usize,
}

/// 把时间正序的日线转成前复权序列，并剔除价格非正或非有限的坏 K 线。
pub fn forward_adjust(bars: &[HistoricalData]) -> AdjustedSeries {
    let valid: Vec<&HistoricalData> = bars.iter().filter(|bar| is_valid(bar)).collect();
    let n = valid.len();

    // returns[i] 是第 i 根相对第 i−1 根的真实收益；returns[0] 不使用
    let mut returns = vec![0.0; n];
    let mut mismatches = 0usize;
    for i in 1..n {
        let (prev, current) = (valid[i - 1], valid[i]);
        let raw = current.close / prev.close - 1.0;
        let pre_close = current.close - current.change;
        let implied = current.close / pre_close - 1.0;
        returns[i] =
            if pre_close > 0.0 && implied.is_finite() && implied.abs() <= MAX_ABS_DAILY_RETURN {
                if (pre_close / prev.close - 1.0).abs() > BASIS_MISMATCH_TOLERANCE {
                    mismatches += 1;
                }
                implied
            } else {
                raw
            };
    }

    let pairs = n.saturating_sub(1);
    let adjusted = mismatches <= TOLERATED_MISMATCHES
        || (mismatches as f64) <= MAX_BASIS_MISMATCH_RATIO * pairs as f64;
    if !adjusted {
        returns = (0..n)
            .map(|i| {
                if i == 0 {
                    0.0
                } else {
                    valid[i].close / valid[i - 1].close - 1.0
                }
            })
            .collect();
    }

    // 以最新收盘为锚往回连乘；factor = 复权价 / 原始价，同一天的 OHLC 共用
    let mut factors = vec![1.0; n];
    if n > 0 {
        let mut adjusted_close = valid[n - 1].close;
        for i in (1..n).rev() {
            let previous_close = adjusted_close / (1.0 + returns[i]);
            factors[i - 1] = previous_close / valid[i - 1].close;
            adjusted_close = previous_close;
        }
    }

    AdjustedSeries {
        bars: valid
            .iter()
            .zip(&factors)
            .map(|(bar, &factor)| AdjustedBar {
                date: bar.date,
                open: bar.open * factor,
                high: bar.high * factor,
                low: bar.low * factor,
                close: bar.close * factor,
            })
            .collect(),
        adjusted,
        corporate_actions: if adjusted { mismatches } else { 0 },
    }
}

fn is_valid(bar: &HistoricalData) -> bool {
    [bar.open, bar.high, bar.low, bar.close]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
        && bar.high >= bar.low
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(day: u32, close: f64, change: f64) -> HistoricalData {
        HistoricalData {
            symbol: "600000".to_string(),
            date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + chrono::Duration::days(day as i64),
            open: close,
            close,
            high: close * 1.01,
            low: close * 0.99,
            volume: 1000,
            amount: 0.0,
            amplitude: 0.0,
            turnover_rate: 0.0,
            volume_ratio: 0.0,
            change_percent: 0.0,
            change,
        }
    }

    #[test]
    fn bonus_share_split_does_not_create_a_fake_crash() {
        // 10 送 10：昨收 20，除权后基准 10，当天收 10.2（真实 +2%）
        let bars = vec![bar(0, 19.0, 0.0), bar(1, 20.0, 1.0), bar(2, 10.2, 0.2)];
        let series = forward_adjust(&bars);
        assert!(series.adjusted);
        assert_eq!(series.corporate_actions, 1);
        let closes: Vec<f64> = series.bars.iter().map(|b| b.close).collect();
        // 最新一根保持真实价
        assert!((closes[2] - 10.2).abs() < 1e-12);
        // 除权日真实涨幅 +2%，没有腰斩
        assert!((closes[2] / closes[1] - 1.02).abs() < 1e-9, "{closes:?}");
        // 更早的价格按同一因子缩放：20 → 10，19 → 9.5
        assert!((closes[1] - 10.0).abs() < 1e-9);
        assert!((closes[0] - 9.5).abs() < 1e-9);
        // 高低开同步缩放
        assert!((series.bars[0].high - 19.0 * 1.01 * 0.5).abs() < 1e-9);
    }

    #[test]
    fn small_cash_dividend_is_adjusted_too() {
        // 派息 0.3：昨收 10，除息基准 9.7，当天收 9.7（真实平盘）
        let bars = vec![bar(0, 10.0, 0.0), bar(1, 9.7, 0.0)];
        let series = forward_adjust(&bars);
        assert_eq!(series.corporate_actions, 1);
        assert!((series.bars[0].close - 9.7).abs() < 1e-9);
    }

    #[test]
    fn missing_change_field_falls_back_to_raw_prices() {
        // change 恒为 0：若照单全收，整条线会被压平成常数
        let bars: Vec<HistoricalData> = (0..40)
            .map(|i| bar(i, 10.0 + i as f64 * 0.1, 0.0))
            .collect();
        let series = forward_adjust(&bars);
        assert!(!series.adjusted, "change 缺失时必须放弃复权");
        assert_eq!(series.corporate_actions, 0);
        for (adjusted, raw) in series.bars.iter().zip(&bars) {
            assert!((adjusted.close - raw.close).abs() < 1e-9);
        }
    }

    #[test]
    fn absurd_implied_return_uses_raw_ratio() {
        // pre_close 异常（1.0）会推出 +1300% 的"真实收益"，必须拒绝
        let bars = vec![bar(0, 14.0, 0.0), bar(1, 14.5, 13.5)];
        let series = forward_adjust(&bars);
        let closes: Vec<f64> = series.bars.iter().map(|b| b.close).collect();
        assert!((closes[1] / closes[0] - 14.5 / 14.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_bars_are_dropped() {
        let mut zero = bar(1, 0.0, 0.0);
        zero.high = 0.0;
        zero.low = 0.0;
        zero.open = 0.0;
        let bars = vec![bar(0, 10.0, 0.0), zero, bar(2, 10.5, 0.5)];
        let series = forward_adjust(&bars);
        assert_eq!(series.bars.len(), 2);
        assert!(series.bars.iter().all(|b| b.close > 0.0));
    }

    #[test]
    fn empty_and_single_bar_inputs() {
        assert!(forward_adjust(&[]).bars.is_empty());
        let one = forward_adjust(&[bar(0, 10.0, 0.0)]);
        assert_eq!(one.bars.len(), 1);
        assert!(one.adjusted);
        assert!((one.bars[0].close - 10.0).abs() < 1e-12);
    }
}
