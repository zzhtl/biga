# 🇨🇳 中国A股专业级预测策略 v2.0

## 📋 概述

针对中国股市特点设计的高精度预测系统，融合技术分析、资金流向、市场微观结构和政策因素。

## 🎯 A股市场特有因素

### 1. 交易制度
```rust
pub struct ChinaMarketRules {
    pub limit_up_down: f64,        // 涨跌停限制 10%
    pub st_limit: f64,              // ST股票限制 5%
    pub trading_mode: String,       // T+1 交易制度
    pub min_lot: i32,               // 最小交易单位 100股
    pub trading_hours: Vec<(String, String)>,  // 交易时段
}

impl ChinaMarketRules {
    // 检查是否接近涨停
    pub fn near_limit_up(&self, current_price: f64, prev_close: f64) -> bool {
        let change_pct = (current_price - prev_close) / prev_close;
        change_pct >= 0.09  // 接近9%视为临近涨停
    }
    
    // 检查是否接近跌停
    pub fn near_limit_down(&self, current_price: f64, prev_close: f64) -> bool {
        let change_pct = (current_price - prev_close) / prev_close;
        change_pct <= -0.09
    }
}
```

### 2. 北向资金流向分析
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NorthboundCapital {
    pub date: String,
    pub net_inflow: f64,           // 净流入（亿元）
    pub total_buy: f64,            // 买入金额
    pub total_sell: f64,           // 卖出金额
    pub active_stocks: Vec<String>, // 主力买入股票
    pub market_sentiment: f64,     // 市场情绪指数 0-100
}

// 分析北向资金对个股的影响
pub fn analyze_northbound_impact(
    stock_code: &str,
    northbound_data: &[NorthboundCapital]
) -> NorthboundSignal {
    // 1. 连续净流入天数
    // 2. 流入强度（相对于个股流通市值）
    // 3. 与指数背离情况
}
```

### 3. 龙虎榜数据分析
```rust
#[derive(Debug, Clone)]
pub struct LonghubangData {
    pub date: String,
    pub stock_code: String,
    pub reason: String,            // 上榜原因（涨停、换手率等）
    pub buy_seats: Vec<TradingSeat>,
    pub sell_seats: Vec<TradingSeat>,
    pub institutional_net: f64,    // 机构净买入
    pub hot_money_net: f64,        // 游资净买入
}

#[derive(Debug, Clone)]
pub struct TradingSeat {
    pub name: String,
    pub buy_amount: f64,
    pub sell_amount: f64,
    pub seat_type: SeatType,       // 机构/游资/普通席位
}

pub enum SeatType {
    Institution,    // 机构
    HotMoney,       // 知名游资
    Unknown,
}
```

## 💡 增强型技术分析

### 1. 筹码分布分析
```rust
#[derive(Debug, Clone)]
pub struct ChipDistribution {
    pub price_levels: Vec<f64>,
    pub chip_concentration: Vec<f64>,  // 各价位筹码集中度
    pub avg_cost: f64,                 // 平均持仓成本
    pub profit_ratio: f64,             // 获利盘比例
    pub trapped_ratio: f64,            // 套牢盘比例
    pub chip_peak: Vec<f64>,           // 筹码峰位置
}

impl ChipDistribution {
    // 计算筹码集中度（90%筹码的价格区间占比）
    pub fn concentration_degree(&self) -> f64 {
        // 筹码越集中，主力控盘越强
    }
    
    // 判断筹码是否充分换手
    pub fn is_fully_exchanged(&self, current_price: f64) -> bool {
        // 当前价与筹码峰的距离，判断是否充分换手
    }
}
```

### 2. 量价结构分析
```rust
#[derive(Debug, Clone)]
pub struct VolumepriceStructure {
    pub obv: Vec<f64>,                 // 能量潮
    pub volume_ratio: Vec<f64>,        // 量比
    pub turnover_rate: Vec<f64>,       // 换手率
    pub price_volume_correlation: f64, // 价量相关性
    pub volume_momentum: f64,          // 成交量动能
}

pub fn analyze_volume_structure(prices: &[f64], volumes: &[i64]) -> VolumeSignal {
    // 1. 放量突破信号
    // 2. 缩量回调信号
    // 3. 量能衰竭信号
    // 4. 底部放量信号
}
```

### 3. 缠论形态识别（核心）
```rust
#[derive(Debug, Clone)]
pub struct ChanTheory {
    pub pens: Vec<Pen>,           // 笔
    pub segments: Vec<Segment>,   // 线段
    pub centers: Vec<Center>,     // 中枢
    pub trends: Vec<Trend>,       // 趋势类型
}

#[derive(Debug, Clone)]
pub struct Pen {
    pub start_idx: usize,
    pub end_idx: usize,
    pub direction: Direction,      // 向上笔/向下笔
    pub high: f64,
    pub low: f64,
}

#[derive(Debug, Clone)]
pub struct Center {
    pub price_range: (f64, f64),   // 中枢价格区间
    pub time_range: (usize, usize),
    pub level: i32,                // 中枢级别
    pub strength: f64,             // 中枢强度
}

pub fn identify_buy_points_by_chan(data: &[Candle]) -> Vec<ChanBuyPoint> {
    // 缠论买卖点：
    // 第一类买点：下跌趋势背驰
    // 第二类买点：回抽中枢不破
    // 第三类买点：突破中枢向上
}
```

### 4. 波浪理论应用
```rust
#[derive(Debug, Clone)]
pub struct ElliottWave {
    pub current_wave: WaveType,
    pub wave_level: i32,
    pub extension_wave: Option<i32>,  // 延长浪
    pub fibonacci_levels: Vec<f64>,   // 斐波那契回调/扩展
}

pub enum WaveType {
    Impulse1,
    Corrective2,
    Impulse3,        // 主升浪
    Corrective4,
    Impulse5,
    CorrectiveA,
    CorrectiveB,
    CorrectiveC,
}

impl ElliottWave {
    // 判断当前所处的浪型位置
    pub fn identify_current_position(&self, prices: &[f64]) -> WavePosition {
        // 通过形态、幅度、时间判断浪型
    }
    
    // 预测目标价位
    pub fn predict_target_price(&self) -> Vec<f64> {
        // 根据斐波那契扩展计算目标价
    }
}
```

## 📊 市场微观结构分析

### 1. 大单流向追踪
```rust
#[derive(Debug, Clone)]
pub struct BigOrderFlow {
    pub date: String,
    pub super_buy: f64,      // 超大单买入（>100万）
    pub super_sell: f64,     // 超大单卖出
    pub big_buy: f64,        // 大单买入（20-100万）
    pub big_sell: f64,       // 大单卖出
    pub medium_buy: f64,     // 中单买入（5-20万）
    pub medium_sell: f64,    // 中单卖出
    pub small_buy: f64,      // 小单买入（<5万）
    pub small_sell: f64,     // 小单卖出
}

pub fn analyze_smart_money(flows: &[BigOrderFlow]) -> SmartMoneySignal {
    // 1. 大单持续流入 = 主力吸筹
    // 2. 大单流出 + 小单接盘 = 主力派发
    // 3. 超大单异动 = 特殊事件
}
```

### 2. 市场情绪指标
```rust
#[derive(Debug, Clone)]
pub struct MarketSentiment {
    pub advance_decline: f64,        // 涨跌家数比
    pub new_high_low: f64,           // 新高新低比
    pub limit_up_count: i32,         // 涨停家数
    pub limit_down_count: i32,       // 跌停家数
    pub turnover_rate: f64,          // 市场换手率
    pub margin_balance: f64,         // 两融余额
    pub margin_change_pct: f64,      // 两融余额变化
    pub etf_net_inflow: f64,         // ETF净流入
}

impl MarketSentiment {
    // 计算市场温度（0-100）
    pub fn market_temperature(&self) -> f64 {
        // 多指标综合评分
        // <20: 极度冰冷，可能底部
        // 20-40: 偏冷，谨慎乐观
        // 40-60: 中性
        // 60-80: 偏热，注意风险
        // >80: 过热，警惕顶部
    }
}
```

### 3. 板块轮动识别
```rust
#[derive(Debug, Clone)]
pub struct SectorRotation {
    pub hot_sectors: Vec<String>,    // 热门板块
    pub cold_sectors: Vec<String>,   // 冷门板块
    pub rotation_phase: RotationPhase,
    pub sector_momentum: HashMap<String, f64>,
}

pub enum RotationPhase {
    DefensiveToValue,      // 防御到价值
    ValueToGrowth,         // 价值到成长
    GrowthToCyclical,      // 成长到周期
    CyclicalToDefensive,   // 周期到防御
}

pub fn predict_sector_rotation(
    historical_data: &[SectorData]
) -> Vec<String> {
    // 预测下一个可能热起来的板块
}
```

## 🎯 综合买卖点策略

### 策略1：主力吸筹买点
```rust
pub struct MainForceAccumulationBuyPoint {
    pub conditions: Vec<String>,
    pub confidence: f64,
}

impl MainForceAccumulationBuyPoint {
    pub fn identify(data: &StockData) -> Option<Self> {
        // 必要条件：
        // 1. 筹码集中度上升
        // 2. 大单持续净流入（5天以上）
        // 3. 缩量横盘或小幅震荡
        // 4. 均线多头排列或即将形成
        // 5. MACD底背离或即将金叉
        
        // 加分项：
        // 6. 北向资金流入
        // 7. 机构龙虎榜买入
        // 8. 行业处于上升周期
        // 9. 突破重要筹码密集区
        
        // 置信度：80-95%
    }
}
```

### 策略2：突破回踩买点
```rust
pub struct BreakoutPullbackBuyPoint {
    pub breakout_price: f64,
    pub support_level: f64,
    pub volume_confirmation: bool,
}

impl BreakoutPullbackBuyPoint {
    pub fn identify(data: &StockData) -> Option<Self> {
        // 必要条件：
        // 1. 放量突破重要压力位
        // 2. 回踩突破位获得支撑
        // 3. 回踩时缩量
        // 4. 未跌破5日或10日均线
        
        // 加分项：
        // 5. 突破时有大单买入
        // 6. 回踩时小单抛售
        // 7. MACD保持多头
        
        // 风险收益比：≥3:1
        // 置信度：70-85%
    }
}
```

### 策略3：超跌反弹买点
```rust
pub struct OversoldRebound BuyPoint {
    pub rsi: f64,
    pub deviation_rate: f64,
    pub support_distance: f64,
}

impl OversoldRebound BuyPoint {
    pub fn identify(data: &StockData) -> Option<Self> {
        // 必要条件：
        // 1. RSI < 30（超卖）
        // 2. 价格远离5日均线（乖离率< -8%）
        // 3. 出现止跌K线形态（锤子线、启明星等）
        // 4. 成交量开始温和放大
        
        // 风险提示：
        // - 只做反弹不做反转
        // - 严格止损（3-5%）
        // - 快进快出
        
        // 置信度：60-70%
    }
}
```

### 策略4：趋势加速买点
```rust
pub struct TrendAccelerationBuyPoint {
    pub trend_strength: f64,
    pub acceleration_factor: f64,
}

impl TrendAccelerationBuyPoint {
    pub fn identify(data: &StockData) -> Option<Self> {
        // 必要条件：
        // 1. 明确的上升趋势（20日均线向上）
        // 2. 突破前期高点
        // 3. 成交量持续放大
        // 4. 均线呈多头排列且发散
        
        // 特征：
        // - 涨幅开始加速
        // - 阳线实体增大
        // - 回调幅度减小
        
        // 操作：追涨策略，设置移动止损
        // 置信度：75-85%
    }
}
```

## 🚫 卖出信号识别

### 危险信号1：量价背离
```rust
pub fn detect_volume_price_divergence_sell(data: &StockData) -> Option<SellSignal> {
    // 价格创新高但成交量萎缩
    // 或价格滞涨但成交量激增
    // 危险等级：★★★★
}
```

### 危险信号2：破位出局
```rust
pub fn detect_breakdown_sell(data: &StockData) -> Option<SellSignal> {
    // 跌破重要支撑位（如10日/20日均线）
    // 且放量下跌
    // 危险等级：★★★★★
    // 操作：立即止损
}
```

### 危险信号3：顶部形态
```rust
pub fn detect_top_pattern_sell(data: &StockData) -> Option<SellSignal> {
    // 头肩顶、双顶、圆弧顶等
    // 配合MACD顶背离
    // 危险等级：★★★★
    // 操作：分批减仓
}
```

### 危险信号4：主力派发
```rust
pub fn detect_distribution_sell(data: &StockData) -> Option<SellSignal> {
    // 大单持续流出
    // 小单接盘
    // 筹码集中度下降
    // 危险等级：★★★★★
    // 操作：跟随主力离场
}
```

## 📈 风险控制系统

### 1. 动态仓位管理
```rust
pub struct PositionManager {
    pub max_position: f64,           // 最大仓位 70%
    pub single_stock_limit: f64,     // 单股上限 20%
    pub current_position: f64,
}

impl PositionManager {
    pub fn calculate_position(&self, signal_strength: f64, market_temp: f64) -> f64 {
        // 信号强度 × 市场温度 × 风险系数 = 建议仓位
        
        // 市场温度 < 30: 可以适当加仓
        // 市场温度 30-70: 正常仓位
        // 市场温度 > 70: 降低仓位
    }
}
```

### 2. 止损止盈策略
```rust
pub struct StopLossStrategy {
    pub fixed_stop_loss: f64,        // 固定止损 -5%
    pub trailing_stop: f64,          // 移动止损
    pub time_stop: i32,              // 时间止损 10个交易日
}

pub struct TakeProfitStrategy {
    pub targets: Vec<f64>,           // 多目标止盈
    pub position_reduction: Vec<f64>, // 分批止盈比例
}
```

## 🎓 实现步骤

### 阶段1：数据采集增强（1-2天）
- [ ] 集成北向资金API
- [ ] 采集龙虎榜数据
- [ ] 获取大单流向数据
- [ ] 板块资金流向

### 阶段2：技术指标增强（2-3天）
- [ ] 实现筹码分布计算
- [ ] 缠论形态识别
- [ ] 波浪理论分析
- [ ] 量价结构分析

### 阶段3：策略整合（2-3天）
- [ ] 多因子综合评分模型
- [ ] 买卖点决策树
- [ ] 信号置信度计算
- [ ] 风险评估系统

### 阶段4：回测优化（2-3天）
- [ ] 历史数据回测
- [ ] 参数优化
- [ ] 准确率统计
- [ ] 策略迭代

## 📊 预期提升

| 指标 | 当前 | 目标 | 提升 |
|------|------|------|------|
| 买点准确率 | ~60% | >75% | +25% |
| 盈亏比 | ~1.5:1 | >2.5:1 | +67% |
| 平均收益率 | ~3% | >5% | +67% |
| 最大回撤 | -8% | <-5% | +37% |

## ⚠️ 重要提示

1. **数据依赖**：需要接入专业金融数据源（如Tushare Pro、Wind、同花顺等）
2. **计算复杂度**：高级策略需要更多计算资源
3. **实盘验证**：策略需要经过至少3个月的实盘验证
4. **持续优化**：市场环境变化，策略需要动态调整
5. **风险警示**：任何策略都有失效风险，严格控制仓位

---

**当前状态**：设计阶段
**实现优先级**：高
**预计完成时间**：10-15个工作日
**需要资源**：专业金融数据API、计算资源 