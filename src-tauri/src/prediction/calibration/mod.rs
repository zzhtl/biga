//! 概率校准与概率评分。
//!
//! 本模块回答一个问题：一个被渲染成百分比的数，配不配叫「概率」。
//!
//! 判据是 proper scoring rule 加无技能基准对照，两件事必须分开看：
//! - **校准（calibration）**：报 60% 的那批样本，实际是否约 60% 发生。看 [`ReliabilityBin`]
//!   与 [`ProbabilityScore::reliability`]。
//! - **技能（skill）**：相对「永远报气候基率」的参照预测有没有增量信息。看
//!   [`ProbabilityScore::brier_skill_score`] 与 [`ProbabilityScore::resolution`]。
//!
//! 两者互相独立：永远报基率的预测完全校准但零技能（BSS=0）；有判别力却整体偏移的预测
//! 有技能但不校准。界面上只给其中一个都会误导——本项目单股方向已被实证为无技能
//! （见 README 与 `backtest::metrics::BacktestMetrics::edge`），所以「校准但无技能」
//! 正是预期结果，必须能被如实表达出来。
//!
//! [`isotonic_fit`] 把任意单调分数映射成校准概率（PAVA 保序回归）。
//! **必须在走步样本外拟合**：同一批样本拟合又自评，得到的校准是自欺。

use serde::{Deserialize, Serialize};

/// 可靠性图的分桶数（预测概率按 [0,0.1) … [0.9,1.0] 十等分）
pub const RELIABILITY_BINS: usize = 10;

/// log loss 的概率截断，避免报 0/1 时出现无穷大
const LOG_LOSS_EPS: f64 = 1e-12;

/// 可靠性图的一个分桶。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// 桶左边界（含）
    pub lo: f64,
    /// 桶右边界（末桶含右端，其余不含）
    pub hi: f64,
    /// 落入该桶的样本数
    pub count: usize,
    /// 桶内预测概率均值；空桶为 0
    pub mean_predicted: f64,
    /// 桶内事件实际发生频率；空桶为 0
    pub realized_freq: f64,
}

impl ReliabilityBin {
    /// 校准缺口 |预测均值 − 实际频率|。空桶返回 `None`。
    pub fn gap(&self) -> Option<f64> {
        (self.count > 0).then(|| (self.mean_predicted - self.realized_freq).abs())
    }
}

/// 一组「预测概率 vs 二元结果」的评分结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityScore {
    /// 有效样本数
    pub n: usize,
    /// Brier 分数（均方误差，越小越好；范围 [0,1]）
    pub brier: f64,
    /// 对数损失（越小越好）
    pub log_loss: f64,
    /// 参照预测使用的气候基率
    pub climatology: f64,
    /// Brier Skill Score = 1 − brier/brier_ref。**>0 才说明有技能**。
    /// 基率退化为 0 或 1 时参照分数为 0，技能无定义，返回 `None`。
    pub brier_skill_score: Option<f64>,
    /// 预测概率均值（与 `realized_freq` 差得远＝整体偏移）
    pub mean_predicted: f64,
    /// 事件实际发生频率
    pub realized_freq: f64,
    /// Murphy 分解之可靠性项，越小越好（0 = 完美校准）。
    ///
    /// 注意：分解 `Brier = reliability − resolution + uncertainty` 只对**分桶后**的预测
    /// 精确成立。用连续概率时还有一项桶内离散度被丢掉，所以三项重建出的值与
    /// [`ProbabilityScore::brier`] 会差一个不超过 (桶宽/2)² 的量。三项之间自洽，
    /// 拿来做诊断没问题；不要用它们去反推 Brier。
    pub reliability: f64,
    /// Murphy 分解之分辨率项，越大越好（0 = 与永远报基率无异，即无技能）
    pub resolution: f64,
    /// Murphy 分解之不确定性项 = c(1−c)，只由数据本身决定，与预测无关
    pub uncertainty: f64,
    /// 期望校准误差：各桶 |预测均值−实际频率| 的样本量加权平均
    pub ece: f64,
    /// 可靠性图分桶，始终 [`RELIABILITY_BINS`] 个（含空桶，便于跨次对比）
    pub bins: Vec<ReliabilityBin>,
}

impl ProbabilityScore {
    /// 样本量达到 `min_count` 的分桶里，最大的校准缺口。
    ///
    /// 这是验收判据用的量：小样本桶的频率噪声极大，不设门槛会被噪声支配。
    pub fn max_gap_over(&self, min_count: usize) -> Option<f64> {
        self.bins
            .iter()
            .filter(|b| b.count >= min_count.max(1))
            .filter_map(|b| b.gap())
            .max_by(f64::total_cmp)
    }

    /// 是否有技能（BSS > 0）。无定义时视为无技能。
    pub fn has_skill(&self) -> bool {
        self.brier_skill_score.is_some_and(|s| s > 0.0)
    }
}

/// 用样本自身的基率作参照，给一组 (预测概率, 是否发生) 打分。
pub fn score_probabilities(pairs: &[(f64, bool)]) -> Option<ProbabilityScore> {
    let clim = climatology(pairs)?;
    score_probabilities_against(pairs, clim)
}

/// 用外部给定的气候基率作参照打分。
///
/// 跨样本比较时必须用同一个参照——否则 BSS 之间不可比。样本外评估尤其要用**训练期**
/// 的基率，而不是评估期自己的基率。
pub fn score_probabilities_against(
    pairs: &[(f64, bool)],
    climatology: f64,
) -> Option<ProbabilityScore> {
    let valid: Vec<(f64, bool)> = pairs
        .iter()
        .filter(|(p, _)| p.is_finite())
        .map(|&(p, y)| (p.clamp(0.0, 1.0), y))
        .collect();
    if valid.is_empty() || !climatology.is_finite() {
        return None;
    }
    let n = valid.len();
    let n_f = n as f64;

    let mut brier = 0.0;
    let mut log_loss = 0.0;
    let mut brier_ref = 0.0;
    let mut pred_sum = 0.0;
    let mut event_count = 0usize;

    let mut bin_count = [0usize; RELIABILITY_BINS];
    let mut bin_pred_sum = [0.0_f64; RELIABILITY_BINS];
    let mut bin_event = [0usize; RELIABILITY_BINS];

    for &(p, y) in &valid {
        let outcome = if y { 1.0 } else { 0.0 };
        brier += (p - outcome).powi(2);
        brier_ref += (climatology - outcome).powi(2);
        let safe = p.clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS);
        log_loss -= if y { safe.ln() } else { (1.0 - safe).ln() };
        pred_sum += p;
        event_count += usize::from(y);

        let idx = bin_index(p);
        bin_count[idx] += 1;
        bin_pred_sum[idx] += p;
        bin_event[idx] += usize::from(y);
    }

    brier /= n_f;
    log_loss /= n_f;
    brier_ref /= n_f;

    let realized_freq = event_count as f64 / n_f;

    let bins: Vec<ReliabilityBin> = (0..RELIABILITY_BINS)
        .map(|i| {
            let c = bin_count[i];
            let w = c as f64;
            ReliabilityBin {
                lo: i as f64 / RELIABILITY_BINS as f64,
                hi: (i + 1) as f64 / RELIABILITY_BINS as f64,
                count: c,
                mean_predicted: if c > 0 { bin_pred_sum[i] / w } else { 0.0 },
                realized_freq: if c > 0 { bin_event[i] as f64 / w } else { 0.0 },
            }
        })
        .collect();

    // Murphy 分解：Brier = reliability − resolution + uncertainty
    let mut reliability = 0.0;
    let mut resolution = 0.0;
    let mut ece = 0.0;
    for b in &bins {
        if b.count == 0 {
            continue;
        }
        let w = b.count as f64 / n_f;
        reliability += w * (b.mean_predicted - b.realized_freq).powi(2);
        resolution += w * (b.realized_freq - realized_freq).powi(2);
        ece += w * (b.mean_predicted - b.realized_freq).abs();
    }

    Some(ProbabilityScore {
        n,
        brier,
        log_loss,
        climatology,
        brier_skill_score: (brier_ref > 0.0).then(|| 1.0 - brier / brier_ref),
        mean_predicted: pred_sum / n_f,
        realized_freq,
        reliability,
        resolution,
        uncertainty: realized_freq * (1.0 - realized_freq),
        ece,
        bins,
    })
}

/// 样本的气候基率（事件发生频率）。空样本返回 `None`。
pub fn climatology(pairs: &[(f64, bool)]) -> Option<f64> {
    let valid: Vec<bool> = pairs
        .iter()
        .filter(|(p, _)| p.is_finite())
        .map(|&(_, y)| y)
        .collect();
    if valid.is_empty() {
        return None;
    }
    Some(valid.iter().filter(|&&y| y).count() as f64 / valid.len() as f64)
}

fn bin_index(p: f64) -> usize {
    let idx = (p * RELIABILITY_BINS as f64).floor() as isize;
    idx.clamp(0, RELIABILITY_BINS as isize - 1) as usize
}

// =============================================================================
// 区间预测的评分
// =============================================================================

/// Pinball loss（分位损失）：`tau` 分位预测 `quantile` 对实际值 `actual` 的损失。
///
/// 这是分位数预测的 proper scoring rule。区间评估必须配它，**光看覆盖率会被骗**：
/// 把带无限放宽就能让覆盖率达到 100%，但那条带毫无用处；pinball 会因带宽惩罚它。
pub fn pinball_loss(actual: f64, quantile: f64, tau: f64) -> f64 {
    let diff = actual - quantile;
    if diff >= 0.0 {
        tau * diff
    } else {
        (tau - 1.0) * diff
    }
}

/// 一组区间预测的评分。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IntervalScore {
    /// 有效样本数
    pub n: usize,
    /// 名义覆盖率（如 0.80）
    pub nominal: f64,
    /// 经验覆盖率
    pub coverage: f64,
    /// 平均带宽（与输入同单位）
    pub mean_width: f64,
    /// 上下两个分位的 pinball 之和，越小越好
    pub pinball: f64,
    /// 落在带下方的比例——与落在上方的比例严重不等说明带偏了，而不只是宽窄问题
    pub below_rate: f64,
    /// 落在带上方的比例
    pub above_rate: f64,
}

impl IntervalScore {
    /// 覆盖率相对名义值的偏差（正数表示过覆盖）
    pub fn coverage_gap(&self) -> f64 {
        self.coverage - self.nominal
    }
}

/// 给一组 `(实际值, 下界, 上界)` 打分。`nominal` 是名义覆盖率，如 0.80。
pub fn score_intervals(observations: &[(f64, f64, f64)], nominal: f64) -> Option<IntervalScore> {
    if !(0.0..1.0).contains(&nominal) {
        return None;
    }
    let valid: Vec<(f64, f64, f64)> = observations
        .iter()
        .filter(|(a, l, u)| a.is_finite() && l.is_finite() && u.is_finite() && u >= l)
        .copied()
        .collect();
    if valid.is_empty() {
        return None;
    }
    let n = valid.len();
    let n_f = n as f64;
    // 名义 80% 带对应 10%/90% 两个分位
    let tau_low = (1.0 - nominal) / 2.0;
    let tau_high = 1.0 - tau_low;

    let mut covered = 0usize;
    let mut below = 0usize;
    let mut above = 0usize;
    let mut width_sum = 0.0;
    let mut pinball = 0.0;
    for (actual, lower, upper) in valid {
        if actual < lower {
            below += 1;
        } else if actual > upper {
            above += 1;
        } else {
            covered += 1;
        }
        width_sum += upper - lower;
        pinball += pinball_loss(actual, lower, tau_low) + pinball_loss(actual, upper, tau_high);
    }

    Some(IntervalScore {
        n,
        nominal,
        coverage: covered as f64 / n_f,
        mean_width: width_sum / n_f,
        pinball: pinball / n_f,
        below_rate: below as f64 / n_f,
        above_rate: above as f64 / n_f,
    })
}

/// 保序回归拟出的「分数 → 概率」单调映射。
///
/// 内部按 PAVA 的块存：块内是平台（就是该块的拟合概率），块与块之间线性过渡。
/// 这样训练点上取到的值与 PAVA 解**完全一致**——若改成在块重心之间插值，训练点上的
/// 取值会偏离拟合解，样本内 Brier 甚至可能劣于常数基率，校准层反而把信息弄丢。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsotonicMap {
    segments: Vec<IsotonicSegment>,
    /// 拟合样本的整体基率，用于非法输入兜底
    base_rate: f64,
}

/// PAVA 的一个块：分数落在 `[lo, hi]` 内时概率取 `y`。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct IsotonicSegment {
    lo: f64,
    hi: f64,
    y: f64,
}

impl IsotonicMap {
    /// 把一个原始分数映射成概率。区间外做水平外推（clip），不外插。
    ///
    /// 外推刻意不做线性延伸：训练期没见过的极端分数外插出 0% 或 100% 是最典型的
    /// 过拟合表现，压在端点上更诚实。
    pub fn apply(&self, x: f64) -> f64 {
        let Some(first) = self.segments.first() else {
            return self.base_rate;
        };
        if !x.is_finite() {
            return self.base_rate;
        }
        if x <= first.lo {
            return first.y;
        }
        let last = self.segments.last().expect("已确认非空");
        if x >= last.hi {
            return last.y;
        }
        // 各块的 hi 严格递增，且此处已保证 x < last.hi，下标必然合法
        let i = self.segments.partition_point(|s| s.hi < x);
        let seg = self.segments[i];
        if x >= seg.lo {
            return seg.y; // 落在块内：平台
        }
        let prev = self.segments[i - 1]; // 落在两块之间：线性过渡
        let span = seg.lo - prev.hi;
        if span <= 0.0 {
            return seg.y;
        }
        prev.y + (x - prev.hi) / span * (seg.y - prev.y)
    }

    /// 拟合样本的整体基率——映射退化成常数时，它就是唯一能报的数。
    pub fn base_rate(&self) -> f64 {
        self.base_rate
    }

    /// 分段数。为 1 说明保序回归把整条曲线压平了，即分数对结果无单调信息。
    ///
    /// 注意反过来不成立：纯噪声上分段数按 O(n^(1/3)) 增长，**分段多不等于有信息**，
    /// 判断有没有信息一律看样本外 BSS。
    pub fn segments(&self) -> usize {
        self.segments.len()
    }
}

/// PAVA 保序回归：拟合「分数 → 事件概率」的单调非减映射。
///
/// **必须用走步样本外的数据拟合。** 本函数不做任何切分，切分是调用方的责任。
///
/// 返回 `None` 仅当没有任何有限分数样本。
pub fn isotonic_fit(pairs: &[(f64, bool)]) -> Option<IsotonicMap> {
    let mut pts: Vec<(f64, f64)> = pairs
        .iter()
        .filter(|(x, _)| x.is_finite())
        .map(|&(x, y)| (x, if y { 1.0 } else { 0.0 }))
        .collect();
    if pts.is_empty() {
        return None;
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let base_rate = pts.iter().map(|(_, y)| y).sum::<f64>() / pts.len() as f64;

    // 先按相同分数并桶：同一个分数必须拿到同一个概率
    let mut pooled: Vec<(f64, f64, f64)> = Vec::new(); // (x, y_sum, weight)
    for (x, y) in pts {
        match pooled.last_mut() {
            Some(last) if last.0 == x => {
                last.1 += y;
                last.2 += 1.0;
            }
            _ => pooled.push((x, y, 1.0)),
        }
    }

    // PAVA：块为 (x_lo, x_hi, y 和, 权重)，违反单调就与前一块合并
    let mut blocks: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(pooled.len());
    for (x, y_sum, w) in pooled {
        blocks.push((x, x, y_sum, w));
        while blocks.len() >= 2 {
            let n = blocks.len();
            let (lo1, _, ys1, w1) = blocks[n - 2];
            let (_, hi2, ys2, w2) = blocks[n - 1];
            if ys1 / w1 <= ys2 / w2 {
                break;
            }
            blocks.truncate(n - 2);
            blocks.push((lo1, hi2, ys1 + ys2, w1 + w2));
        }
    }

    // 合并概率相同的相邻块：PAVA 只在"违反单调"时合并，等值块会留成多段，
    // 让 segments() 失去"有几个不同概率档位"的含义。
    let mut merged: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(blocks.len());
    for b in blocks {
        match merged.last_mut() {
            Some(last) if (last.2 / last.3 - b.2 / b.3).abs() < 1e-12 => {
                last.1 = b.1;
                last.2 += b.2;
                last.3 += b.3;
            }
            _ => merged.push(b),
        }
    }

    // 把块概率夹离 0/1。理由有二：
    // 1. 保序回归在两端有已知的 spiking——端块常被拟成 0.0/1.0，而样本外那批样本的
    //    真实频率并不是 0/1，这是端点上最大的校准缺口来源。
    // 2. 报 0% / 100% 会让 log loss 爆成无穷，界面上也从不该出现"必然"。
    // 夹取是逐点单调变换，不会破坏 PAVA 解的单调性。
    let n_total = merged.iter().map(|(_, _, _, w)| w).sum::<f64>().max(1.0);
    let floor = 1.0 / (2.0 * n_total);

    Some(IsotonicMap {
        segments: merged
            .into_iter()
            .map(|(lo, hi, y_sum, w)| IsotonicSegment {
                lo,
                hi,
                y: (y_sum / w).clamp(floor, 1.0 - floor),
            })
            .collect(),
        base_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性伪随机，避免为测试引入 rand 依赖
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64)
    }

    #[test]
    fn test_isotonic_output_is_monotone() {
        let mut seed = 42u64;
        // 分数越高越容易发生，但带噪声
        let pairs: Vec<(f64, bool)> = (0..500)
            .map(|i| {
                let x = i as f64 / 500.0;
                (x, lcg(&mut seed) < x)
            })
            .collect();
        let map = isotonic_fit(&pairs).expect("应能拟合");
        assert!(
            map.segments.windows(2).all(|w| w[0].y <= w[1].y + 1e-12),
            "块概率必须非递减"
        );
        assert!(
            map.segments.windows(2).all(|w| w[0].hi < w[1].lo),
            "块的分数区间必须两两不交且递增"
        );
        // 单调性也必须体现在 apply 上
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=20 {
            let y = map.apply(i as f64 / 20.0);
            assert!(y >= prev - 1e-12, "apply 必须单调: {y} < {prev}");
            assert!((0.0..=1.0).contains(&y));
            prev = y;
        }
    }

    #[test]
    fn test_isotonic_on_noise_has_no_out_of_sample_skill() {
        // 保序回归在纯噪声上照样能拟出样本内"技能"（块数按 O(n^(1/3)) 增长，压不平是
        // 正常的）。这正是本模块反复强调必须样本外评估的原因——本测试把这条契约钉死。
        let mut seed = 7u64;
        let all: Vec<(f64, bool)> = (0..2000)
            .map(|_| (lcg(&mut seed), lcg(&mut seed) < 0.53))
            .collect();
        let (train, holdout) = all.split_at(1000);

        let map = isotonic_fit(train).expect("应能拟合");
        let in_sample: Vec<(f64, bool)> = train.iter().map(|&(x, y)| (map.apply(x), y)).collect();
        let oos: Vec<(f64, bool)> = holdout.iter().map(|&(x, y)| (map.apply(x), y)).collect();

        let bss_in = score_probabilities(&in_sample)
            .unwrap()
            .brier_skill_score
            .unwrap();
        let bss_out = score_probabilities(&oos).unwrap().brier_skill_score.unwrap();

        assert!(bss_in > 0.0, "样本内必然出现虚假技能，否则本测试失去意义: {bss_in}");
        assert!(bss_out <= 0.01, "纯噪声的样本外 BSS 必须 ≈0 或为负，得到 {bss_out}");
    }

    #[test]
    fn test_isotonic_recovers_real_signal_out_of_sample() {
        // 与上一个测试对照：真有信号时，样本外必须看得见技能，且大致校准。
        let mut seed = 99u64;
        let all: Vec<(f64, bool)> = (0..2000)
            .map(|_| {
                let x = lcg(&mut seed);
                (x, lcg(&mut seed) < x)
            })
            .collect();
        let (train, holdout) = all.split_at(1000);

        let map = isotonic_fit(train).expect("应能拟合");
        let oos: Vec<(f64, bool)> = holdout.iter().map(|&(x, y)| (map.apply(x), y)).collect();
        let s = score_probabilities(&oos).expect("应能打分");

        assert!(s.has_skill(), "真实信号的样本外 BSS 必须 >0: {:?}", s.brier_skill_score);
        // ECE 是样本量加权的平均缺口，小桶噪声不会支配它——这是该拿来做验收的聚合量。
        assert!(s.ece < 0.06, "样本外整体应校准: ece={}", s.ece);
        // 单桶最大缺口只在大桶上才有意义：n=30 时仅二项抽样噪声就有 ±0.18（2σ）。
        assert!(
            s.max_gap_over(150).unwrap_or(0.0) < 0.06,
            "大桶上不该有明显缺口: {:?}",
            s.max_gap_over(150)
        );
    }

    #[test]
    fn test_isotonic_clips_outside_training_range() {
        let pairs = vec![(0.2, false), (0.4, false), (0.6, true), (0.8, true)];
        let map = isotonic_fit(&pairs).expect("应能拟合");
        // 区间外做水平外推，不外插出 0%/100%
        assert_eq!(map.apply(-99.0), map.apply(0.2));
        assert_eq!(map.apply(99.0), map.apply(0.8));
        // 0/0/1/1 → 两个档位，且被夹离 0/1：n=4 时 floor = 1/(2*4) = 0.125
        assert_eq!(map.segments(), 2, "等值相邻块应被合并成一档");
        assert!((map.apply(0.2) - 0.125).abs() < 1e-12, "得到 {}", map.apply(0.2));
        assert!((map.apply(0.8) - 0.875).abs() < 1e-12, "得到 {}", map.apply(0.8));
        assert!(map.apply(0.2) > 0.0 && map.apply(0.8) < 1.0, "永远不该报 0% 或 100%");
        // 两块之间线性过渡
        assert!((map.apply(0.5) - 0.5).abs() < 1e-12);
        assert!((map.apply(f64::NAN) - map.base_rate()).abs() < 1e-12);
    }

    #[test]
    fn test_isotonic_empty_input() {
        assert!(isotonic_fit(&[]).is_none());
        assert!(isotonic_fit(&[(f64::NAN, true)]).is_none());
    }

    #[test]
    fn test_perfect_forecast_has_bss_one() {
        let pairs = vec![(1.0, true), (0.0, false), (1.0, true), (0.0, false), (1.0, true)];
        let s = score_probabilities(&pairs).expect("应能打分");
        assert!(s.brier.abs() < 1e-12, "完美预测 Brier 应为 0，得到 {}", s.brier);
        assert!((s.brier_skill_score.unwrap() - 1.0).abs() < 1e-12);
        assert!(s.has_skill());
        assert!(s.reliability.abs() < 1e-12, "完美预测应完全校准");
    }

    #[test]
    fn test_climatology_forecast_has_bss_zero() {
        // 7 涨 3 跌，全部报基率 0.7
        let mut pairs: Vec<(f64, bool)> = (0..7).map(|_| (0.7, true)).collect();
        pairs.extend((0..3).map(|_| (0.7, false)));
        let s = score_probabilities(&pairs).expect("应能打分");
        assert!((s.climatology - 0.7).abs() < 1e-12);
        assert!(
            s.brier_skill_score.unwrap().abs() < 1e-12,
            "报基率的 BSS 必须是 0，得到 {:?}",
            s.brier_skill_score
        );
        assert!(!s.has_skill(), "报基率不算技能");
        // 完全校准（reliability=0）但零分辨率——这正是本项目单股方向的预期形态
        assert!(s.reliability.abs() < 1e-12);
        assert!(s.resolution.abs() < 1e-12);
        assert!((s.brier - s.uncertainty).abs() < 1e-12);
    }

    #[test]
    fn test_reliability_bins_conserve_count() {
        let mut seed = 1234u64;
        let pairs: Vec<(f64, bool)> = (0..977)
            .map(|_| {
                let p = lcg(&mut seed);
                (p, lcg(&mut seed) < p)
            })
            .collect();
        let s = score_probabilities(&pairs).expect("应能打分");
        assert_eq!(s.n, 977);
        assert_eq!(s.bins.len(), RELIABILITY_BINS);
        assert_eq!(s.bins.iter().map(|b| b.count).sum::<usize>(), 977, "分桶必须计数守恒");
        // Murphy 分解：Brier = reliability − resolution + uncertainty，只对分桶后的预测
        // 精确成立；连续概率下差一项桶内离散度，上界是 (桶宽/2)² = 0.0025。
        let rebuilt = s.reliability - s.resolution + s.uncertainty;
        let within_bin_slack = (0.5 / RELIABILITY_BINS as f64).powi(2);
        assert!(
            (s.brier - rebuilt).abs() < within_bin_slack,
            "Murphy 三项应自洽到桶内离散度以内: brier={} rebuilt={} 容差={}",
            s.brier,
            rebuilt,
            within_bin_slack
        );
        // 由 p 生成的结果本身就是校准的 → ECE 应该很小
        assert!(s.ece < 0.08, "自洽生成的概率 ECE 不该大: {}", s.ece);
    }

    #[test]
    fn test_bin_edges_include_one() {
        // p = 1.0 必须落进末桶而不是越界
        let s = score_probabilities(&[(1.0, true), (0.0, false)]).expect("应能打分");
        assert_eq!(s.bins[RELIABILITY_BINS - 1].count, 1);
        assert_eq!(s.bins[0].count, 1);
    }

    #[test]
    fn test_bss_undefined_when_base_rate_degenerate() {
        // 全部上涨 → 参照分数为 0，技能无定义
        let pairs = vec![(0.6, true), (0.8, true)];
        let s = score_probabilities(&pairs).expect("应能打分");
        assert_eq!(s.climatology, 1.0);
        assert!(s.brier_skill_score.is_none(), "退化基率下 BSS 必须是 None 而不是编造一个数");
        assert!(!s.has_skill());
    }

    #[test]
    fn test_max_gap_over_filters_small_bins() {
        // 0.05 桶只有 1 个样本却缺口巨大；0.95 桶有 40 个样本且完全校准
        let mut pairs = vec![(0.05, true)];
        pairs.extend((0..38).map(|_| (0.95, true)));
        pairs.extend((0..2).map(|_| (0.95, false)));
        let s = score_probabilities(&pairs).expect("应能打分");
        let unfiltered = s.max_gap_over(1).unwrap();
        let filtered = s.max_gap_over(10).unwrap();
        assert!(unfiltered > 0.9, "不设门槛会被单样本桶支配: {unfiltered}");
        assert!(filtered < 0.05, "设了门槛后应只剩校准良好的大桶: {filtered}");
    }

    #[test]
    fn test_pinball_is_minimised_at_the_true_quantile() {
        // 样本取自 U[0,1]，tau=0.3 的最优预测是 0.3
        let sample: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
        let tau = 0.3;
        let loss = |q: f64| sample.iter().map(|&a| pinball_loss(a, q, tau)).sum::<f64>();
        let best = loss(0.3);
        for q in [0.1_f64, 0.2, 0.4, 0.5, 0.8] {
            assert!(best < loss(q), "tau=0.3 的最优分位应是 0.3，但 q={q} 更优");
        }
    }

    #[test]
    fn test_pinball_penalises_both_directions_asymmetrically() {
        // tau=0.9：低估（实际值高于预测）罚得比高估重
        assert!((pinball_loss(1.0, 0.0, 0.9) - 0.9).abs() < 1e-12);
        assert!((pinball_loss(-1.0, 0.0, 0.9) - 0.1).abs() < 1e-12);
        // tau=0.1 反过来
        assert!((pinball_loss(1.0, 0.0, 0.1) - 0.1).abs() < 1e-12);
        assert!((pinball_loss(-1.0, 0.0, 0.1) - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_wider_band_buys_coverage_but_loses_on_pinball() {
        // 这条性质正是"不能只看覆盖率"的理由：宽带覆盖率更高，pinball 却更差
        let actuals: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 2.0 - 1.0).collect();
        let tight: Vec<(f64, f64, f64)> =
            actuals.iter().map(|&a| (a, -0.8, 0.8)).collect();
        let huge: Vec<(f64, f64, f64)> =
            actuals.iter().map(|&a| (a, -50.0, 50.0)).collect();

        let t = score_intervals(&tight, 0.8).expect("应能打分");
        let h = score_intervals(&huge, 0.8).expect("应能打分");

        assert!(h.coverage > t.coverage, "宽带覆盖率更高");
        assert!((h.coverage - 1.0).abs() < 1e-12);
        assert!(h.pinball > t.pinball, "但 pinball 必须罚它: tight={} huge={}", t.pinball, h.pinball);
        assert!((t.coverage - 0.8).abs() < 0.01, "紧带应恰好覆盖 ~80%: {}", t.coverage);
    }

    #[test]
    fn test_interval_score_detects_a_shifted_band() {
        // 带整体偏高：覆盖率可能还凑合，但越界全落在下方——只看覆盖率发现不了
        let obs: Vec<(f64, f64, f64)> = (0..100)
            .map(|i| (i as f64 / 100.0, 0.5, 1.5))
            .collect();
        let s = score_intervals(&obs, 0.8).expect("应能打分");
        assert!(s.below_rate > 0.4, "越界应集中在下方: {}", s.below_rate);
        assert_eq!(s.above_rate, 0.0);
        assert!(s.coverage_gap() < 0.0, "此处应是覆盖不足");
    }

    #[test]
    fn test_score_intervals_rejects_bad_input() {
        assert!(score_intervals(&[], 0.8).is_none());
        assert!(score_intervals(&[(0.0, -1.0, 1.0)], 1.0).is_none());
        // 上界小于下界的样本直接丢弃
        assert!(score_intervals(&[(0.0, 1.0, -1.0)], 0.8).is_none());
    }

    #[test]
    fn test_score_empty_input() {
        assert!(score_probabilities(&[]).is_none());
        assert!(climatology(&[]).is_none());
    }
}
