# 股票预测系统 - 权重配置完全指南

> 📅 最后更新：2025年9月30日
> 
> 📖 本指南涵盖预测系统的权重配置、调优策略和使用说明

---

## 📂 一、配置文件结构

### 1.1 核心权重配置 `core_weights.rs` ⭐⭐⭐⭐⭐

**位置**: `src-tauri/src/stock_prediction/core_weights.rs`

**内容**: 直接影响预测准确率的核心权重（8大类，约90个常量）

**重要程度**: 最高 - 这些权重直接决定预测准确率

**包含类别**:

| 类别 | 数量 | 优先级 | 说明 |
|-----|------|-------|------|
| 多因子评分权重 | 8个 | ⭐⭐⭐⭐⭐ | 趋势、量价、动量等因子权重，总和=100% |
| 预测算法核心权重 | 5个 | ⭐⭐⭐⭐⭐ | 趋势、技术指标等在预测中的占比 |
| 趋势一致性增强 | 8个 | ⭐⭐⭐⭐⭐ | 反向抑制系数和趋势偏置值 |
| 方向投票权重 | 9个 | ⭐⭐⭐⭐ | MACD、KDJ、RSI等指标的投票权重 |
| 技术指标影响权重 | 5个 | ⭐⭐⭐⭐ | 不同市场状态下技术指标权重 |
| 趋势衰减系数 | 7个 | ⭐⭐⭐⭐ | 随天数增加的影响力衰减 |
| 均值回归参数 | 5个 | ⭐⭐⭐ | 价格偏离均线的回归力量 |
| 量价预测权重 | 6个 | ⭐⭐⭐ | 方向与价格准确率权重比例 |

### 1.2 技术参数配置 `constants.rs` ⭐⭐⭐

**位置**: `src-tauri/src/stock_prediction/constants.rs`

**内容**: 技术指标周期、阈值、评分参数（21大类，约120个常量）

**重要程度**: 中等 - 影响计算细节，通常不需要频繁调整

**主要类别**: 技术指标周期、评分参数、阈值设置、风险控制参数等

---

## 🎯 二、核心权重详解

### 2.1 多因子评分权重（⭐⭐⭐⭐⭐ 最高优先级）

**总和必须=1.0** | 直接影响买卖点识别和操作建议

```rust
趋势因子            TREND_FACTOR_WEIGHT = 0.22         (22%)
量价因子            VOLUME_PRICE_FACTOR_WEIGHT = 0.18   (18%)
多周期共振          MULTI_TIMEFRAME_FACTOR_WEIGHT = 0.15 (15%)
动量因子            MOMENTUM_FACTOR_WEIGHT = 0.13       (13%)
K线形态            PATTERN_FACTOR_WEIGHT = 0.12         (12%)
支撑压力            SUPPORT_RESISTANCE_FACTOR_WEIGHT = 0.10 (10%)
市场情绪            SENTIMENT_FACTOR_WEIGHT = 0.07       (7%)
波动率              VOLATILITY_FACTOR_WEIGHT = 0.03      (3%)
```

**调整建议**:
- 趋势市：提高趋势、多周期共振权重
- 震荡市：提高支撑压力、形态权重
- 每次±5%，保持总和=1.0

### 2.2 预测算法核心权重（⭐⭐⭐⭐⭐ 最高优先级）

**决定最终价格预测的计算方式**

```rust
趋势因子占比        PREDICTION_TREND_RATIO = 0.40       (40%)
技术指标占比        PREDICTION_TECHNICAL_RATIO = 0.30   (30%)
均线量能占比        PREDICTION_MA_VOLUME_RATIO = 0.20   (20%)
市场波动占比        PREDICTION_MARKET_FLUCTUATION_RATIO = 0.12 (12%)
基础模型占比        PREDICTION_BASE_MODEL_RATIO = 0.10  (10%)
```

**典型问题与调整**:
- 追涨杀跌严重 → 降低趋势占比(40%→35%)，提高技术指标(30%→35%)
- 预测过于保守 → 提高趋势占比，降低波动占比

### 2.3 趋势一致性增强系数（⭐⭐⭐⭐⭐ 最高优先级）

**控制趋势与技术指标一致时的预测强度**

```rust
强一致性反向抑制    STRONG_ALIGNMENT_OPPOSITE_SUPPRESS = 0.25   (25%)
强一致性偏置值      STRONG_ALIGNMENT_BIAS = 0.010               (1.0%)
中等一致性反向抑制  MEDIUM_ALIGNMENT_OPPOSITE_SUPPRESS = 0.40   (40%)
中等一致性偏置值    MEDIUM_ALIGNMENT_BIAS = 0.006               (0.6%)
```

**调整逻辑**:
- 反向抑制系数 **越小** = 趋势持续性越强
- 偏置值 **越大** = 预测越激进

---

## 🔧 三、快速调优指南

### 3.1 常见问题诊断表

| 问题现象 | 问题定位 | 调整参数 | 调整方向 |
|---------|---------|---------|---------|
| ❌ 方向预测频繁错误 | 方向判断逻辑 | `方向投票权重`<br>`趋势一致性系数` | 加强MACD/KDJ投票分<br>提高反向抑制系数 |
| ❌ 追涨杀跌严重 | 趋势权重过高 | `PREDICTION_TREND_RATIO`<br>`STRONG_ALIGNMENT_BIAS` | 40%→35%<br>1.0%→0.6% |
| ❌ 涨跌幅偏差大 | 预测算法权重 | `预测算法核心权重` | 调整各占比平衡 |
| ❌ 预测信号迟钝 | 技术指标权重低 | `TECH_STRONG_ALIGNED_WEIGHT`<br>`PREDICTION_TECHNICAL_RATIO` | 3.5%→4.0%<br>30%→35% |
| ❌ 震荡市表现差 | 因子权重不适 | `SUPPORT_RESISTANCE_WEIGHT`<br>`PATTERN_FACTOR_WEIGHT` | 10%→12%<br>12%→14% |
| ❌ 趋势市错失机会 | 趋势权重不足 | `TREND_FACTOR_WEIGHT`<br>`MULTI_TIMEFRAME_WEIGHT` | 22%→25%<br>15%→18% |
| ⚠️ 信号触发过于频繁 | 阈值设置 | `constants.rs` 阈值参数 | 提高阈值 |

### 3.2 调优步骤流程

```
第1步：基准回测
├─ 记录当前准确率（整体、方向、价格）
├─ 记录各因子得分分布
└─ 记录问题场景（上涨、下跌、震荡）

第2步：问题诊断
├─ 使用上表定位问题类型
├─ 确定调整文件（core_weights.rs 或 constants.rs）
└─ 确定调整参数

第3步：调整权重
├─ 只修改 1-2 个相关参数
├─ 小幅调整：±5%（如 0.40→0.42）
├─ 中幅调整：±10%（如 0.40→0.44）
└─ 大幅调整：±20%（如 0.40→0.48，谨慎使用）

第4步：验证效果
├─ cargo build --release
├─ cargo run --example test_backtest
└─ 对比调整前后数据

第5步：记录归档
├─ 记录修改的参数名和值
├─ 记录修改原因和预期
├─ 记录实际效果
└─ 效果不佳则使用 git 回滚
```

### 3.3 调优示例

#### 示例1：预测追涨杀跌（滞后问题）

**现象**: 总是在涨了之后才预测上涨，跌了之后才预测下跌

**原因**: 趋势权重过高，导致过度依赖已形成的趋势

**解决方案**:
```rust
// core_weights.rs

// 1. 降低趋势在预测中的占比
pub const PREDICTION_TREND_RATIO: f64 = 0.35;  // 原值 0.40 → 0.35

// 2. 提高技术指标占比
pub const PREDICTION_TECHNICAL_RATIO: f64 = 0.35;  // 原值 0.30 → 0.35

// 3. 降低强一致性偏置（减少追涨杀跌）
pub const STRONG_ALIGNMENT_BIAS: f64 = 0.006;  // 原值 0.010 → 0.006
```

**预期效果**: 预测更注重技术指标变化，减少对趋势的滞后跟随

#### 示例2：方向预测频繁错误

**现象**: 预测方向经常与实际相反

**原因**: 反向预测未被充分抑制，或投票权重不合理

**解决方案**:
```rust
// core_weights.rs

// 1. 加强反向预测抑制（从25%降到20%）
pub const STRONG_ALIGNMENT_OPPOSITE_SUPPRESS: f64 = 0.20;  // 原值 0.25 → 0.20

// 2. 提高MACD金叉死叉的投票权重
pub const MACD_CROSS_VOTE_SCORE: f64 = 1.5;  // 原值 1.2 → 1.5

// 3. 提高趋势因子权重
pub const TREND_FACTOR_WEIGHT: f64 = 0.25;  // 原值 0.22 → 0.25
pub const VOLUME_PRICE_FACTOR_WEIGHT: f64 = 0.15;  // 原值 0.18 → 0.15（保持总和=1.0）
```

**预期效果**: 趋势判断更坚定，方向错误率降低

#### 示例3：震荡市表现差

**现象**: 在震荡市频繁止损，无法盈利

**原因**: 趋势权重过高，支撑压力和形态权重不足

**解决方案**:
```rust
// core_weights.rs

// 调整多因子评分权重（保持总和=1.0）
pub const TREND_FACTOR_WEIGHT: f64 = 0.20;  // 22% → 20%
pub const SUPPORT_RESISTANCE_FACTOR_WEIGHT: f64 = 0.12;  // 10% → 12%
pub const PATTERN_FACTOR_WEIGHT: f64 = 0.14;  // 12% → 14%
pub const MULTI_TIMEFRAME_FACTOR_WEIGHT: f64 = 0.13;  // 15% → 13%
```

**预期效果**: 更关注支撑压力位和K线形态，在震荡市表现更好

---

## 📊 四、权重配置速查表

### 4.1 多因子评分权重速查

| 权重名称 | 默认值 | 适合场景 | 调高时机 | 调低时机 |
|---------|-------|---------|---------|---------|
| TREND_FACTOR_WEIGHT | 22% | 趋势市 | 牛市/熊市 | 震荡市 |
| VOLUME_PRICE_FACTOR_WEIGHT | 18% | 量价配合好 | 放量明显 | 量能噪音大 |
| MULTI_TIMEFRAME_FACTOR_WEIGHT | 15% | 共振明确 | 多周期一致 | 周期冲突多 |
| MOMENTUM_FACTOR_WEIGHT | 13% | 动量明显 | 指标有效 | 指标钝化 |
| PATTERN_FACTOR_WEIGHT | 12% | 短期交易 | 形态准确 | 关注趋势 |
| SUPPORT_RESISTANCE_FACTOR_WEIGHT | 10% | 关键位 | 位置有效 | 频繁突破 |
| SENTIMENT_FACTOR_WEIGHT | 7% | 情绪极端 | 恐慌/狂热 | 情绪平稳 |
| VOLATILITY_FACTOR_WEIGHT | 3% | 风险控制 | 控制风险 | 正常波动 |

### 4.2 预测算法权重速查

| 权重名称 | 默认值 | 建议范围 | 调整影响 |
|---------|-------|---------|---------|
| PREDICTION_TREND_RATIO | 40% | 30-50% | 过高→追涨杀跌<br>过低→错失趋势 |
| PREDICTION_TECHNICAL_RATIO | 30% | 20-40% | 影响短期波动捕捉 |
| PREDICTION_MA_VOLUME_RATIO | 20% | 15-25% | 影响中期趋势判断 |
| PREDICTION_MARKET_FLUCTUATION_RATIO | 12% | 8-15% | 影响预测平滑度 |
| PREDICTION_BASE_MODEL_RATIO | 10% | 5-15% | 过高→过拟合<br>过低→浪费模型 |

### 4.3 趋势一致性系数速查

| 系数名称 | 默认值 | 建议范围 | 调整逻辑 |
|---------|-------|---------|---------|
| STRONG_ALIGNMENT_OPPOSITE_SUPPRESS | 0.25 | 0.20-0.35 | 越小→趋势越强 |
| STRONG_ALIGNMENT_BIAS | 0.010 | 0.005-0.015 | 越大→越激进 |
| MEDIUM_ALIGNMENT_OPPOSITE_SUPPRESS | 0.40 | 0.35-0.50 | 中等强度控制 |
| WEAK_ALIGNMENT_OPPOSITE_SUPPRESS | 0.50 | 0.45-0.60 | 弱趋势控制 |

---

## 🎓 五、使用建议

### 5.1 针对不同用户

#### 新手用户
1. ✅ **先熟悉** `core_weights.rs` 中的核心权重含义
2. ✅ **从示例开始**，参考"调优示例"进行调整
3. ✅ **小幅调整**，每次只改1个参数，±5%
4. ✅ **记录每次调整**，建立调优日志
5. ❌ **避免调整** `constants.rs` 中的技术参数

#### 进阶用户
1. ✅ 可同时调整2-3个相关权重
2. ✅ 针对不同市场环境设计权重组合
3. ✅ 建立回测对比表格，量化效果
4. ✅ 使用 git 分支管理不同权重版本

#### 专家用户
1. ✅ 组合调整多个相关权重
2. ✅ 设计自适应权重切换机制
3. ✅ 使用自动化脚本批量回测
4. ✅ 建立权重优化算法

### 5.2 调优注意事项

#### ✅ 应该做的
- ✅ 每次调整前做基准回测
- ✅ 每次只调整1-2个参数
- ✅ 调整幅度从小到大（5%→10%→20%）
- ✅ 调整后立即回测验证
- ✅ 记录每次调整的原因和结果
- ✅ 使用 git 管理权重版本
- ✅ 建立调优日志和对比表

#### ❌ 不应该做的
- ❌ 同时大幅调整多个不相关参数
- ❌ 不做回测就直接应用到实盘
- ❌ 频繁调整技术参数（constants.rs）
- ❌ 调整后不记录原因和效果
- ❌ 没有备份就修改权重
- ❌ 凭感觉调整，不看数据
- ❌ 调整后总和≠1.0（多因子权重）

---

## 📁 六、文件引用关系

```
core_weights.rs (核心权重)
     ↓ 被引用
├── prediction.rs           (预测算法)
├── multi_factor_scoring.rs (多因子评分)
└── utils.rs               (工具函数)

constants.rs (技术参数)
     ↓ 被引用
├── prediction.rs           (预测算法)
├── multi_factor_scoring.rs (多因子评分)
├── technical_indicators.rs (技术指标)
├── technical_analysis.rs   (技术分析)
└── utils.rs               (工具函数)
```

### 如何引用

```rust
// 在需要使用的文件中
use crate::stock_prediction::core_weights::*;  // 核心权重
use crate::stock_prediction::constants::*;     // 技术参数
```

---

## 🔍 七、回测验证

### 7.1 回测命令

```bash
# 进入项目目录
cd src-tauri

# 编译项目
cargo build --release

# 运行回测示例
cargo run --example test_backtest

# 查看结果
# 关注：整体准确率、方向准确率、各场景表现
```

### 7.2 关键指标

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| 整体预测准确率 | >70% | 综合评估 |
| 方向预测准确率 | >65% | 最重要指标 |
| 价格预测准确率 | >60% | 辅助指标 |
| 上涨场景准确率 | >65% | 看涨能力 |
| 下跌场景准确率 | >65% | 看跌能力 |
| 震荡场景准确率 | >55% | 震荡市能力 |

### 7.3 效果评估

**调整成功的标志**:
- ✅ 目标场景准确率提升 >5%
- ✅ 其他场景准确率下降 <3%
- ✅ 整体准确率提升或持平

**需要回滚的情况**:
- ❌ 整体准确率下降 >3%
- ❌ 某个场景准确率暴跌 >10%
- ❌ 出现系统性方向错误

---

## 📞 八、问题排查

### 8.1 编译错误

**现象**: 修改后编译失败

**排查步骤**:
```bash
# 1. 检查语法
cargo check

# 2. 查看详细错误
cargo build 2>&1 | less

# 3. 常见错误
- 权重总和≠1.0（多因子评分权重）
- 引用了不存在的常量
- 数值类型不匹配
```

### 8.2 预测异常

**现象**: 预测结果明显不合理

**排查步骤**:
1. 检查多因子评分权重总和是否=1.0
2. 检查预测算法权重是否在合理范围
3. 检查趋势一致性系数是否过极端
4. 使用 git diff 查看修改了什么
5. 回滚到上一个稳定版本

### 8.3 性能问题

**现象**: 预测速度变慢

**原因**: 通常不是权重问题，可能是：
- 数据量过大
- 技术指标计算周期过长
- 内存不足

**解决**: 优化代码逻辑，而非调整权重

---

## ✅ 附录：权重调整记录模板

建议创建 `WEIGHT_TUNING_LOG.md` 记录每次调整：

```markdown
## 调整记录 #001

**日期**: 2025-09-30
**问题**: 预测追涨杀跌严重
**调整参数**:
- PREDICTION_TREND_RATIO: 0.40 → 0.35
- PREDICTION_TECHNICAL_RATIO: 0.30 → 0.35
- STRONG_ALIGNMENT_BIAS: 0.010 → 0.006

**回测结果对比**:
| 指标 | 调整前 | 调整后 | 变化 |
|-----|-------|-------|------|
| 整体准确率 | 68% | 71% | +3% ✅ |
| 方向准确率 | 63% | 67% | +4% ✅ |
| 追涨杀跌次数 | 15次 | 8次 | -7次 ✅ |

**结论**: 调整有效，保留此版本
**Git Commit**: abc123def456
```

---

## 🎯 总结

### 核心理念
> **核心权重决定预测准确率，技术参数决定计算细节**
> 
> **优先调整 core_weights.rs，谨慎调整 constants.rs**
> 
> **每次调整必须回测验证，记录完整数据**

### 快速上手
1. 📖 熟悉 `core_weights.rs` 的8大类权重
2. 🔍 使用"常见问题诊断表"定位问题
3. 🔧 参考"调优示例"进行小幅调整
4. 🧪 运行回测验证效果
5. 📝 记录调整过程和结果

### 相关文件
- `src-tauri/src/stock_prediction/core_weights.rs` - 核心权重（含详细注释）
- `src-tauri/src/stock_prediction/constants.rs` - 技术参数
- `src-tauri/src/stock_prediction/prediction.rs` - 预测算法实现
- `src-tauri/src/stock_prediction/multi_factor_scoring.rs` - 多因子评分实现

---

祝调优顺利！🚀 如有疑问，请参考 `core_weights.rs` 中的详细注释。 