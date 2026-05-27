//! 截面因子库
//!
//! 提供原始因子值（不做缩放，交由截面标准化处理），用于滚动多因子排序。
//! 因子方向不在此处硬编码——由滚动 IC 自适应定权。
//!
//! 重点：包含**量比 × 换手率**的交互/组合因子，验证两者结合的预测力。

use crate::db::models::HistoricalData;

/// 因子所需的最小回看窗口
pub const FACTOR_LOOKBACK: usize = 20;

/// 因子名称（顺序与 [`compute_factor_row`] 一致）
pub fn factor_names() -> Vec<String> {
    [
        "volatility20",
        "turnover5",
        "amplitude5",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ma20_gap",
        "range_pos20",
        "rsi14",
        "volume_ratio",
        "vr_x_turnover",  // 量比 × 换手率（交互）
        "vr_over_turnover", // 量比 / 换手率（低换手上的量比异动）
        // === 量 × 价 交叉因子 ===
        "vr_x_ret5",       // 量比 × 5日动量（放量上涨 vs 放量下跌，方向性区分）
        "volprice_corr10", // 10日量价配合度（量变与价变的相关系数）
        "obv_slope10",     // OBV 10日归一化斜率（资金流方向）
        "vr_x_rangepos",   // 量比 × 区间位置去中心（高位放量出货 / 低位放量吸筹）
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// 因子维度
pub fn factor_dim() -> usize {
    factor_names().len()
}

/// 皮尔逊相关系数（样本不足或零方差时返回 0，保证因子有限）
fn corr(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return 0.0;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        let (dx, dy) = (x - mx, y - my);
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 1e-12 || vy <= 1e-12 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// 计算索引 `i` 处的原始因子向量（仅用 ≤ i 的数据）。i 须 ≥ FACTOR_LOOKBACK。
pub fn compute_factor_row(h: &[HistoricalData], i: usize) -> Option<Vec<f64>> {
    if i < FACTOR_LOOKBACK || i >= h.len() {
        return None;
    }
    let close = |k: usize| h[k].close;
    let ret = |a: usize, b: usize| -> f64 {
        if close(b) > 0.0 {
            (close(a) - close(b)) / close(b)
        } else {
            0.0
        }
    };

    // 20日收益波动
    let rets: Vec<f64> = ((i - 19)..=i).map(|k| ret(k, k - 1)).collect();
    let mean_r = rets.iter().sum::<f64>() / rets.len() as f64;
    let volatility = (rets.iter().map(|x| (x - mean_r).powi(2)).sum::<f64>() / rets.len() as f64).sqrt();

    // 近5日平均换手率 / 振幅（更稳）
    let turnover5 = ((i - 4)..=i).map(|k| h[k].turnover_rate).sum::<f64>() / 5.0;
    let amplitude5 = ((i - 4)..=i).map(|k| h[k].amplitude).sum::<f64>() / 5.0;

    let ret_5 = ret(i, i - 5);
    let ret_10 = ret(i, i - 10);
    let ret_20 = ret(i, i - 20);

    let ma20 = ((i - 19)..=i).map(close).sum::<f64>() / 20.0;
    let ma_gap = if ma20 > 0.0 { close(i) / ma20 - 1.0 } else { 0.0 };

    let max20 = ((i - 19)..=i).map(|k| h[k].high).fold(f64::MIN, f64::max);
    let min20 = ((i - 19)..=i).map(|k| h[k].low).fold(f64::MAX, f64::min);
    let range_pos = if max20 > min20 {
        (close(i) - min20) / (max20 - min20)
    } else {
        0.5
    };

    // RSI(14)
    let (mut gain, mut loss) = (0.0, 0.0);
    for k in (i - 13)..=i {
        let d = close(k) - close(k - 1);
        if d >= 0.0 {
            gain += d;
        } else {
            loss -= d;
        }
    }
    let rsi = if gain + loss > 0.0 {
        gain / (gain + loss) * 100.0
    } else {
        50.0
    };

    // 量比：当日成交量 / 过去5日均量
    let vavg5 = ((i - 4)..=i).map(|k| h[k].volume as f64).sum::<f64>() / 5.0;
    let volume_ratio = if vavg5 > 0.0 {
        h[i].volume as f64 / vavg5
    } else {
        1.0
    };

    // 量比 × 换手率 组合因子（用户假设：两者结合更有效）
    let vr_x_turnover = volume_ratio * turnover5;
    let vr_over_turnover = volume_ratio / (turnover5 + 0.5); // 低换手上的量比异动

    // === 量 × 价 交叉因子 ===
    // 放量方向：量比 × 5日动量。放量上涨为正、放量下跌为负，区分量能背后的方向。
    let vr_x_ret5 = volume_ratio * ret_5;

    // 量价配合度：近10日「成交量日变化率」与「价格日收益」的相关系数。
    // 正相关 = 涨放量/跌缩量（健康），负相关 = 量价背离。
    let volprice_corr10 = {
        let vchg: Vec<f64> = ((i - 9)..=i)
            .map(|k| {
                let pv = h[k - 1].volume as f64;
                if pv > 0.0 {
                    (h[k].volume as f64 - pv) / pv
                } else {
                    0.0
                }
            })
            .collect();
        let pret: Vec<f64> = ((i - 9)..=i).map(|k| ret(k, k - 1)).collect();
        corr(&vchg, &pret)
    };

    // 资金流方向：OBV(=累积 sign(Δclose)×成交量) 近10日的归一化斜率。
    let obv_slope10 = {
        let mut obv = 0.0;
        let mut first = None;
        let mut last = 0.0;
        for k in (i - 9)..=i {
            let d = close(k) - close(k - 1);
            let s = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            obv += s * h[k].volume as f64;
            first.get_or_insert(obv);
            last = obv;
        }
        let avgv = ((i - 9)..=i).map(|k| h[k].volume as f64).sum::<f64>() / 10.0;
        if avgv > 0.0 {
            (last - first.unwrap()) / (10.0 * avgv)
        } else {
            0.0
        }
    };

    // 高低位放量：量比 × 区间位置去中心。高位(>0.5)放量为正(出货风险)、低位放量为负(吸筹)。
    let vr_x_rangepos = volume_ratio * (range_pos - 0.5);

    Some(vec![
        volatility,
        turnover5,
        amplitude5,
        ret_5,
        ret_10,
        ret_20,
        ma_gap,
        range_pos,
        rsi,
        volume_ratio,
        vr_x_turnover,
        vr_over_turnover,
        vr_x_ret5,
        volprice_corr10,
        obv_slope10,
        vr_x_rangepos,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn make(n: usize) -> Vec<HistoricalData> {
        (0..n)
            .map(|k| HistoricalData {
                symbol: "t".to_string(),
                date: NaiveDate::from_ymd_opt(2020, 1, 1).unwrap() + chrono::Duration::days(k as i64),
                open: 10.0 + k as f64 * 0.1,
                close: 10.0 + k as f64 * 0.1,
                high: 10.5 + k as f64 * 0.1,
                low: 9.5 + k as f64 * 0.1,
                volume: 1000 + (k as i64 % 5) * 200,
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
    fn test_factor_row_dim() {
        let h = make(40);
        let row = compute_factor_row(&h, 30).expect("应有因子");
        assert_eq!(row.len(), factor_dim());
        assert_eq!(factor_names().len(), factor_dim());
        // 所有因子有限
        assert!(row.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_factor_row_insufficient() {
        let h = make(40);
        assert!(compute_factor_row(&h, 10).is_none());
    }

    #[test]
    fn test_vr_x_turnover_present() {
        let h = make(40);
        let row = compute_factor_row(&h, 30).unwrap();
        let names = factor_names();
        let idx = names.iter().position(|n| n == "vr_x_turnover").unwrap();
        // 量比≈1，换手5日均=3 → 交互≈3
        assert!((row[idx] - 3.0).abs() < 1.0);
    }
}
