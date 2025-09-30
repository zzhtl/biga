# 📊 买卖点准确率统计系统 - 开发计划

## 🎯 目标

实现买卖点信号的历史准确率统计和评估系统，提供量化的策略有效性证明。

## 📋 核心功能

### 1. 信号记录系统
```sql
-- 新增数据表：信号记录表
CREATE TABLE signal_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    signal_type TEXT NOT NULL,  -- '买入点'/'卖出点'
    signal_subtype TEXT,         -- '多周期共振买入点'等
    signal_date TEXT NOT NULL,
    signal_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit_1 REAL,
    take_profit_2 REAL,
    confidence REAL,
    signal_strength REAL,
    reasons TEXT,
    
    -- 实际结果跟踪
    actual_result TEXT,          -- 'pending'/'hit_stop_loss'/'hit_take_profit'/'expired'
    result_date TEXT,
    result_price REAL,
    profit_loss_pct REAL,
    days_to_result INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signal_stock ON signal_records(stock_code);
CREATE INDEX idx_signal_type ON signal_records(signal_type);
CREATE INDEX idx_signal_date ON signal_records(signal_date);
```

### 2. 自动回测评估
```rust
/// 回测评估买卖点信号的准确率
pub async fn backtest_signal_accuracy(
    stock_code: &str,
    start_date: &str,
    end_date: &str,
) -> Result<SignalAccuracyReport, String> {
    // 1. 获取历史数据
    // 2. 在每个时间点生成买卖点信号
    // 3. 跟踪后续价格走势，判断是否触发止损/止盈
    // 4. 统计准确率、盈亏比、平均持仓时间等
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalAccuracyReport {
    pub signal_type: String,
    pub total_signals: usize,
    pub hit_take_profit: usize,
    pub hit_stop_loss: usize,
    pub expired: usize,
    pub accuracy_rate: f64,          // 止盈次数/总次数
    pub avg_profit_pct: f64,         // 平均盈利百分比
    pub avg_loss_pct: f64,           // 平均亏损百分比
    pub profit_loss_ratio: f64,      // 盈亏比
    pub avg_holding_days: f64,       // 平均持仓天数
    pub confidence_accuracy: Vec<(f64, f64)>,  // (置信度区间, 实际准确率)
}
```

### 3. 前端展示

```typescript
interface SignalWithAccuracy extends BuySellPoint {
    historical_accuracy?: {
        total_count: number;
        success_rate: number;
        avg_profit: number;
        avg_loss: number;
    };
}
```

界面展示：
```
💚 多周期共振买入点 (强度: 89分)
建议价格: 85.50元
止损位: 82.71元 (-3.26%)
止盈位: 88.60元(+3.63%)
风险收益比: 1:2.08
置信度: 80%
📊 历史准确率: 72% (基于156次信号)
   平均盈利: +5.2% | 平均亏损: -3.1%
   盈亏比: 1.68 | 平均持仓: 4.3天
```

## 🔧 实现步骤

### 第一阶段：数据库和基础结构
- [ ] 创建信号记录表
- [ ] 实现信号保存接口
- [ ] 实现信号查询接口

### 第二阶段：回测系统
- [ ] 实现单个信号的结果跟踪逻辑
- [ ] 实现批量历史回测功能
- [ ] 计算各类准确率指标
- [ ] 按信号类型、置信度分组统计

### 第三阶段：实时跟踪
- [ ] 每次生成信号时自动保存到数据库
- [ ] 定时任务检查历史信号的实际结果
- [ ] 更新信号状态（pending → hit_stop_loss/hit_take_profit）

### 第四阶段：前端集成
- [ ] 在买卖点卡片中展示历史准确率
- [ ] 创建准确率分析页面
- [ ] 展示不同股票、不同信号类型的表现对比
- [ ] 提供可视化图表（准确率趋势、盈亏分布）

### 第五阶段：优化和预警
- [ ] 识别高质量信号的特征
- [ ] 自动过滤准确率低的信号类型
- [ ] 根据历史表现调整置信度
- [ ] 提供信号质量评分

## 📊 评估指标

### 买入点准确率
- **触达止盈率**：触发止盈位的信号比例
- **平均持仓时间**：从买入到止盈/止损的天数
- **盈亏比**：平均盈利幅度 / 平均亏损幅度
- **夏普比率**：风险调整后收益

### 卖出点准确率
- **避险成功率**：卖出后股价确实下跌的比例
- **错失涨幅**：卖出后股价上涨的平均幅度
- **及时性**：信号发出时机的有效性

## 🎯 目标指标

| 指标 | 目标值 | 优秀值 |
|------|--------|--------|
| 买入点止盈率 | >60% | >75% |
| 盈亏比 | >1.5:1 | >2:1 |
| 卖出点避险成功率 | >70% | >85% |
| 平均持仓时间 | 3-7天 | 3-5天 |

## 📝 使用示例

```rust
// 回测特定股票的买卖点准确率
let report = backtest_signal_accuracy("600519.SH", "2024-01-01", "2024-10-01").await?;

println!("买入点准确率: {:.1}%", report.accuracy_rate * 100.0);
println!("平均盈利: {:.2}%", report.avg_profit_pct);
println!("盈亏比: 1:{:.2}", report.profit_loss_ratio);

// 获取信号时包含历史准确率
let signals = identify_buy_points_with_accuracy(stock_data).await?;
for signal in signals {
    if let Some(accuracy) = signal.accuracy_rate {
        println!("历史准确率: {:.1}%", accuracy * 100.0);
    }
}
```

## ⚠️ 注意事项

1. **数据充分性**：至少需要3个月的历史信号数据才能计算可靠的准确率
2. **市场环境**：牛市和熊市的准确率会有显著差异，需要分别统计
3. **样本偏差**：避免过度拟合，准确率应基于足够大的样本量（>30次）
4. **时效性**：最近3个月的准确率权重应该更高
5. **风险提示**：历史表现不代表未来收益，仅供参考

---

**优先级**：高
**预计工时**：4-6天
**依赖**：现有预测系统和回测框架
**负责人**：待分配 