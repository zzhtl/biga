# 金融级预测系统 - 快速参考

## 新增指标速查 (8大专业指标)

### 1. ATR (平均真实波幅) - 波动率核心
```rust
calculate_atr(highs, lows, closes, 14)
```
- **用途**: 衡量价格波动幅度
- **阈值**: <1.5%稳定, 1.5-3%中等, >3%高波动
- **应用**: 动态止损、风险评估

### 2. 布林带 (Bollinger Bands) - 超买超卖
```rust
calculate_bollinger_bands(prices, 20, 2.0) -> (上轨, 中轨, 下轨)
```
- **用途**: 价格通道,判断超买超卖
- **规则**: 价格触上轨超买,触下轨超卖
- **应用**: 支撑压力位、反转信号

### 3. SAR (抛物线指标) - 趋势反转
```rust
calculate_sar_signal(highs, lows, 0.02, 0.2) -> (SAR值, 看涨趋势)
```
- **用途**: 跟踪趋势,提供止损位
- **规则**: SAR在价格下方看涨,上方看跌
- **应用**: 动态止损、趋势跟踪

### 4. DMI/ADX - 趋势强度量化
```rust
calculate_dmi_adx(highs, lows, closes, 14) -> (DI+, DI-, ADX)
```
- **用途**: 判断趋势强度和方向
- **阈值**: ADX>40强趋势, 25-40中等, <20震荡
- **应用**: 策略切换(趋势vs震荡)

### 5. 威廉指标 (%R) - 短期超买超卖
```rust
calculate_williams_r(highs, lows, closes, 14)
```
- **用途**: 超买超卖判断
- **阈值**: <-80超卖区, >-20超买区
- **应用**: 短期买卖点

### 6. ROC (变动率) - 动量分析
```rust
calculate_roc(prices, 12)
```
- **用途**: 价格变化速度
- **规则**: >0上涨动能, <0下跌动能
- **应用**: 动量评分

### 7. 市场情绪指标 - 恐惧贪婪
```rust
calculate_market_sentiment(prices, volumes, highs, lows)
```
- **输出**: 
  - `sentiment_score`: 0-100情绪得分
  - `fear_greed_index`: 0-100恐惧贪婪指数
  - `market_phase`: 市场阶段(过热/上升/震荡/下跌/恐慌)
- **逆向思维**: 极度恐惧时机会,极度贪婪时风险

### 8. 多因子综合评分 - 8维度智能评分
```rust
calculate_multi_factor_score(factors) -> MultiFactorScore
```
- **8大维度** (v2.1权重优化):
  1. 趋势因子 (22%) ⬇️
  2. 量价因子 (18%) ⬇️
  3. 多周期共振 (15%)
  4. 动量因子 (13%) ⬇️
  5. 形态因子 (12%) ⬇️
  6. 支撑压力 (10%)
  7. **市场情绪 (7%)** ✨新增 ⬇️优化
  8. **波动率 (3%)** ✨新增 ⬇️优化
  
  总和: 100% ✅ (v2.1修正)
  
- **动态权重调整**:
  - 强趋势市场: 趋势因子+30%
  - 震荡市场: 支撑压力+30%
  - 高波动市场: 整体权重-10%
  - 市场恐慌期: 情绪+30%, 支撑压力+30%

## 核心功能调用

### 后端 Rust
```rust
// 金融级专业预测
let (prediction_response, professional_prediction) = 
    predict_with_professional_strategy(request).await?;

// 获取多因子评分
let score = professional_prediction.multi_factor_score;
println!("综合评分: {:.1}分", score.total_score);
println!("信号质量: {}", score.signal_quality);
println!("操作建议: {}", score.operation_suggestion);
```

### 前端 TypeScript
```typescript
// 调用金融级预测
const result = await invoke<ProfessionalPredictionResponse>(
    'predict_with_professional_strategy',
    { request }
);

// 解构结果
const { predictions, professional_analysis } = result;
const { 
    buy_points,           // 买入点
    sell_points,          // 卖出点
    multi_factor_score    // 多因子评分 (含新增情绪&波动率)
} = professional_analysis;
```

## 智能策略核心逻辑

### 1. 市场环境自适应
```
ADX > 40 → 强趋势策略 (趋势跟踪)
ADX 25-40 → 中等趋势策略
ADX < 20 → 震荡策略 (支撑压力)
```

### 2. 波动率风险控制
```
波动率 < 1.5% → 稳定市场 → 加分
波动率 1.5-3% → 中等波动 → 正常
波动率 > 3% → 高风险 → 减分
波动率 > 5% → 极端风险 → 大幅减分
```

### 3. 市场情绪逆向思维
```
恐惧贪婪指数 < 20 → 极度恐惧 → 机会 (逆向加分)
恐惧贪婪指数 20-40 → 恐惧 → 谨慎乐观
恐惧贪婪指数 60-80 → 贪婪 → 警惕
恐惧贪婪指数 > 80 → 极度贪婪 → 风险 (限制高分)
```

### 4. 市场阶段策略切换
```
过热期 → 情绪+20%, 波动率+30%, 趋势-10%
上升期 → 趋势+20%, 多周期+20%, 动量+10%
震荡期 → 支撑压力+30%, 形态+20%, 趋势-20%
恐慌期 → 情绪+30%, 支撑压力+30%, 波动率+20%
```

## 评分系统

### 多因子总分 (0-100分)
- **90-100**: 极强信号 🔥
- **80-89**: 强信号 ✅
- **70-79**: 较强信号 📈
- **60-69**: 中等信号 ⚖️
- **50-59**: 偏弱信号 📉
- **0-49**: 弱信号 ⚠️

### 风险等级
- **1-3分**: 低风险 ✅
- **4-6分**: 中等风险 ⚠️
- **7-8分**: 较高风险 🔶
- **9-10分**: 高风险 ⛔

## 关键文件

```
src-tauri/src/stock_prediction/
├── prediction.rs              # 核心预测逻辑 (主要修改)
├── technical_indicators.rs    # 新增7大专业指标
├── multi_factor_scoring.rs    # 多因子评分+动态权重
├── volume_analysis.rs         # 增强量价分析
├── candlestick_patterns.rs    # K线形态识别
└── multi_timeframe_analysis.rs # 多周期共振

src/routes/components/
└── stock_prediction.svelte    # 前端展示 (需进一步完善)
```

## 测试验证

```bash
# 编译检查 (✅ 已通过,无警告)
cd src-tauri && cargo check

# 运行专业预测示例
cargo run --example test_professional_prediction

# 回测验证
cargo run --example test_backtest
```

## 下一步优化方向

1. ✅ 完成: 增强技术指标体系
2. ✅ 完成: 优化多因子评分权重
3. ✅ 完成: 增加市场情绪分析
4. ✅ 完成: 修正权重配置和评分逻辑 (v2.1)
5. ⏳ 待完成: 实现自适应预测算法 (根据回测自动优化)
6. ⏳ 待完成: 增强风险控制 (VaR计算、仓位管理)

---

**版本**: v2.1 权重修复版  
**更新时间**: 2025-09-30  
**编译状态**: ✅ 通过 (0 errors, 0 warnings)  
**测试状态**: ✅ 通过 (2 passed)

## v2.1 重要修复
- 修正基础权重总和为100% (原120%)
- 修正各因子评分逻辑,确保0-100范围
- 修正RSI超买逻辑 (扣分而非加分)
- 优化动态权重调整,降低调整幅度
- 增加权重上下限保护 (2%-30%) 