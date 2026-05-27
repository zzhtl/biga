# BigA — 项目工作指南（给 Claude / 后续优化）

Tauri + Rust 的沪深股票分析与预测系统。后端在 `src-tauri/`，前端 Svelte。
**所有命令在 `src-tauri/` 下执行**，并按本仓库约定加 `rtk` 前缀（token 优化）。

## 常用命令
```bash
cd src-tauri
rtk cargo build
rtk cargo test                      # 单元 + 集成测试（应全绿，当前 56）
rtk cargo clippy --all-targets
# 量化评测 / 数据工具（见下）务必用 --release，否则训练/回测很慢
cargo run --release --example cross_sectional   # 截面多因子评测（核心）
```
- 数据库：`src-tauri/db/stock_data.db`（SQLite + sqlx）。**不要提交 DB 二进制**（含 -wal/-shm）。
- 迁移：`migrations/*.sql`，需在 `lib.rs` 的 `migration_files` 数组登记；运行器对 `duplicate column` 容错（SQLite 无 `ADD COLUMN IF NOT EXISTS`）。

## 模块地图（src-tauri/src/）
- `db/`：models、repository（CRUD + `backfill_volume_metrics`、`get_symbols_with_min_bars`、`upsert/get_stock_capital`）、connection。
- `api/stock.rs`：zhitu 接口。历史 `hs/history/{code}/d/n`；股本/实时 `hs/real/ssjy/{code}`（**只认纯 6 位代码**，`fetch_stock_capital` 内已用 `normalize_quote_symbol` 归一化 `000002.SZ→000002`）。
- `prediction/`
  - `indicators/`：技术指标（含 `volume_metrics` 由 utils 提供）。
  - `analysis/`：trend / volume / pattern / support_resistance，以及**文件夹模块** `divergence/`、`market_regime/`、`signal_confirmation/`（大文件已按职责拆分，公共 API 不变）。
  - `strategy/`：`multi_factor/`、`professional_engine/`（均为文件夹模块）、adaptive_weights、price_model、multi_timeframe。
  - `model/`：features（含量比/换手率特征 + `build_samples`/`latest_features`）、network（candle MLP + `train_and_save`/`train_eval`/`train_predict`）、ml_inference、inference（规则引擎 13 阶段 + 抽出的 `analyze`）、training、management。
  - `backtest/`：走步回测 + `metrics`（含 `baseline_accuracy`/`edge()`/高置信子集）。
  - **`factor.rs`**：截面因子库（12 因子，含 `vr_x_turnover`、`vr_over_turnover` 量比×换手率组合）。
  - **`cross_section.rs`**：截面标准化 + 滚动 IC 加权多因子 + `orthogonalize_panel` 正交化 + `walk_forward` 评估 + `rank_latest` 生产排名。
- `commands/stock_prediction.rs`：Tauri 命令。`predict_with_candle`(有模型走 ML 否则规则引擎)、`run_model_backtest`(真实走步)、**`cross_sectional_ranking`**(相对强弱排名)。
- `utils/volume_metrics.rs`：量比 = 当日量/过去N日均量；换手率 = 成交额/流通市值×100（单位安全，规避手/股）。

## 关键实证结论（重要——决定优化方向，勿重复踩坑）
1. **单股"次日涨跌方向"预测无 alpha**：规则引擎 ~46–48%、单股/池化 ML ~47%，均**低于朴素基准**（总猜多数方向 ~56%）。天花板=基准，调参/换模型都突破不了。**不要再在这个方向投入。**
2. **截面相对强弱（市场中性）有真实 alpha**：滚动多因子前向 Rank IC ≈ **+0.0497**，多空 top20%-bot20% 毛 +0.57%/5日，扣 0.3%/期双边成本后**净 +0.27%/5日（≈年化 ~13.5%）**。这是产出 alpha 的正确框架。
3. **量比×换手率"组合"远强于单独**：volume_ratio 单独 IC≈+0.004；`vr_x_turnover`≈−0.016、`vr_over_turnover`≈+0.011（强 3–4 倍）。主力因子：低振幅、低换手、低波动。
4. **广度是最大杠杆**：股票数 95→130，Rank IC +0.0355→+0.0480。继续扩到全市场 IC 还会上行。
5. 评测一律用 `cross_section::walk_forward` 的 **`edge()` / 前向 Rank IC** 当裁判，杜绝单次切分偶然与训练方差自欺（MLP 有随机初始化，需多次平均）。

## 数据与评测工具（examples/）
- `fetch_more_data.rs`：批量拉取大盘股历史+股本入库（改 `CODES` 列表，纯 6 位代码）。注意每日 API 额度。
- `batch_refresh.rs`：对全库回填股本/量比/换手率。
- `cross_sectional.rs`：**主评测**——单因子 IC、前向滚动 Rank IC、正交化前后对比、加成本净多空、最新排名。
- `tune_backtest.rs` / `pooled_ml.rs`：规则引擎走步 / 池化 ML 评测（已证无 alpha，留作对照）。

## 后续优化方向（按性价比）
1. **扩票池到全市场**（最高杠杆，需分批拉数据控额度）。
2. 行业/市值中性化（剔除风格暴露，纯化 alpha）。
3. 因子库扩展（已正交化去冗余；可加资金流、基本面等非技术维度——技术因子 IC 上限有限）。
4. 更长调仓周期 / 降换手以减成本拖累（5 日调仓成本敏感）。

## 约定
- 注释/文档优先跟随仓库现状（中文）；不主动格式化既有代码。
- 改动力求外科手术式；新迁移记得登记到 `lib.rs`。
- 提交信息中文；推送默认分支需用户明确授权。
