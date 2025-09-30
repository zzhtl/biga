# 核心权重配置重组说明

## 📅 更新时间
2025年9月30日

## 🎯 重组目的

将**影响涨跌幅预测的核心权重**放在最前面，方便快速调整和定位。

---

## 📂 新的文件结构

### `src-tauri/src/stock_prediction/core_weights.rs`

```
一、预测算法核心权重 ⭐⭐⭐⭐⭐ (影响涨跌幅)
   ├─ PREDICTION_TREND_RATIO = 0.40              (40%)
   ├─ PREDICTION_TECHNICAL_RATIO = 0.30          (30%)
   ├─ PREDICTION_MA_VOLUME_RATIO = 0.20          (20%)
   ├─ PREDICTION_MARKET_FLUCTUATION_RATIO = 0.12 (12%)
   └─ PREDICTION_BASE_MODEL_RATIO = 0.10         (10%)
   
二、多因子综合评分权重 ⭐⭐⭐⭐ (影响操作建议)
   ├─ TREND_FACTOR_WEIGHT = 0.22                 (22%)
   ├─ VOLUME_PRICE_FACTOR_WEIGHT = 0.18          (18%)
   ├─ MULTI_TIMEFRAME_FACTOR_WEIGHT = 0.15       (15%)
   ├─ MOMENTUM_FACTOR_WEIGHT = 0.13              (13%)
   ├─ PATTERN_FACTOR_WEIGHT = 0.12               (12%)
   ├─ SUPPORT_RESISTANCE_FACTOR_WEIGHT = 0.10    (10%)
   ├─ SENTIMENT_FACTOR_WEIGHT = 0.07             (7%)
   └─ VOLATILITY_FACTOR_WEIGHT = 0.03            (3%)
   总和 = 1.0 ✅

三、趋势一致性增强系数 ⭐⭐⭐⭐⭐
四、方向投票权重 ⭐⭐⭐⭐
五、技术指标影响权重 ⭐⭐⭐⭐
六、趋势衰减系数 ⭐⭐⭐⭐
七、均值回归参数 ⭐⭐⭐
八、量价预测权重 ⭐⭐⭐
```

---

## 🔑 核心变更

### 1. 权重顺序调整

**之前**：多因子评分权重在最前面
**现在**：预测算法核心权重在最前面 ⭐

### 2. 多因子评分权重恢复默认值

| 权重名称 | 之前 | 现在 | 变化 |
|---------|-----|------|------|
| TREND_FACTOR_WEIGHT | 0.15 | 0.22 | +7% ✅ 恢复默认 |
| VOLUME_PRICE_FACTOR_WEIGHT | 0.30 | 0.18 | -12% ✅ 恢复默认 |
| MULTI_TIMEFRAME_FACTOR_WEIGHT | 0.20 | 0.15 | -5% ✅ 恢复默认 |
| MOMENTUM_FACTOR_WEIGHT | 0.11 | 0.13 | +2% ✅ 恢复默认 |
| PATTERN_FACTOR_WEIGHT | 0.08 | 0.12 | +4% ✅ 恢复默认 |
| SUPPORT_RESISTANCE_FACTOR_WEIGHT | 0.10 | 0.10 | 不变 |
| SENTIMENT_FACTOR_WEIGHT | 0.01 | 0.07 | +6% ✅ 恢复默认 |
| VOLATILITY_FACTOR_WEIGHT | 0.05 | 0.03 | -2% ✅ 恢复默认 |

**总和检查**：1.0 ✅

---

## 💡 使用建议

### 调整涨跌幅预测 → 修改第一部分

**位置**：`core_weights.rs` 第19-54行

**影响**：直接改变未来价格预测和涨跌幅

**示例**：
```rust
// 降低追涨杀跌
pub const PREDICTION_TREND_RATIO: f64 = 0.35;  // 40% → 35%
pub const PREDICTION_TECHNICAL_RATIO: f64 = 0.35;  // 30% → 35%
```

### 调整操作建议 → 修改第二部分

**位置**：`core_weights.rs` 第56-126行

**影响**：改变"买入/卖出"建议和综合评分

**示例**：
```rust
// 强调趋势因子
pub const TREND_FACTOR_WEIGHT: f64 = 0.25;  // 22% → 25%
pub const SENTIMENT_FACTOR_WEIGHT: f64 = 0.04;  // 7% → 4%
// 其他权重微调保持总和=1.0
```

---

## ✅ 验证结果

- ✅ 编译通过
- ✅ 所有常量引用正确
- ✅ 多因子评分权重总和 = 1.0
- ✅ 权重恢复为优化后的默认值

---

## 📖 相关文档

- `PREDICTION_WEIGHT_CONFIG_GUIDE.md` - 完整的权重配置指南
- `src-tauri/src/stock_prediction/core_weights.rs` - 权重配置文件（含详细注释）

---

## 🎯 快速记忆

```
想改涨跌幅   → 第一部分 PREDICTION_xxx_RATIO
想改操作建议  → 第二部分 xxx_FACTOR_WEIGHT
``` 