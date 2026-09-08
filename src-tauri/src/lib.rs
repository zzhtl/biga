//! Biga 股票预测应用
//! 
//! 基于 Tauri 的股票分析和预测工具

// 核心模块
pub mod config;
pub mod error;
pub mod utils;

// 数据层
pub mod db;
pub mod api;

// 业务模块
pub mod discipline;
pub mod prediction;
pub mod services;

// 命令层
mod commands;

// CSV 处理
mod csv;

use db::connection::{find_database_path, open_pool, run_migrations};
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir { file_name: None }),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Webview),
                ])
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            // 股票列表命令
            commands::stock_list::get_stock_list,
            // 股票信息命令
            commands::stock::get_stock_infos,
            commands::stock::refresh_stock_infos,
            // 实时数据命令
            commands::stock_realtime::get_realtime_data,
            // 历史数据命令
            commands::stock_historical::get_historical_data,
            commands::stock_historical::refresh_historical_data,
            // 预测命令
            commands::stock_prediction::train_stock_prediction_model,
            commands::stock_prediction::predict_stock_price,
            commands::stock_prediction::list_stock_prediction_models,
            commands::stock_prediction::delete_stock_prediction_model,
            commands::stock_prediction::train_candle_model,
            commands::stock_prediction::predict_with_candle,
            commands::stock_prediction::predict_candle_price_simple,
            commands::stock_prediction::retrain_candle_model,
            commands::stock_prediction::evaluate_candle_model,
            commands::stock_prediction::run_model_backtest,
            commands::stock_prediction::get_optimization_suggestions,
            commands::stock_prediction::get_multi_timeframe_signals,
            commands::stock_prediction::get_latest_multi_timeframe_signal,
            commands::stock_prediction::analyze_multi_timeframe_prediction_value,
            commands::stock_prediction::predict_with_professional_strategy,
            commands::stock_prediction::predict_with_technical_only,
            commands::stock_prediction::cross_sectional_ranking,
            commands::stock_prediction::get_valuation_context,
            // 收藏池命令
            commands::watchlist::get_watchlist_overview,
            commands::watchlist::add_to_watchlist,
            commands::watchlist::remove_from_watchlist,
            commands::watchlist::get_watchlist_symbols,
            commands::watchlist::comprehensive_predict,
            // 交易纪律命令
            commands::discipline::get_discipline_account,
            commands::discipline::save_discipline_account,
            commands::discipline::get_discipline_positions,
            commands::discipline::check_buy_discipline,
            commands::discipline::open_discipline_position,
            commands::discipline::record_discipline_trade,
            commands::discipline::resolve_discipline_event,
            commands::discipline::get_discipline_review,
            // 安全设置命令
            commands::settings::get_api_token_status,
            commands::settings::save_api_token,
            commands::settings::clear_api_token,
            commands::settings::test_api_token
        ])
        .setup(|app| {
            tauri::async_runtime::block_on(async {
                // 库文件位置：已有库优先（开发时就在 src-tauri/db 下，不搬家）。
                // 都没有时——debug 构建落在 CWD 的 db/，保持 `bun run tauri dev` 的习惯；
                // release 构建落到系统应用数据目录，因为装机后进程 CWD 是用户双击时
                // 所在的目录，往那儿写会到处撒 db 文件夹，换个地方启动还看不到自己的持仓。
                let db_path = match find_database_path() {
                    Some(path) => path,
                    None if cfg!(debug_assertions) => std::env::current_dir()
                        .expect("应能读取当前工作目录")
                        .join("db/stock_data.db"),
                    None => app
                        .path()
                        .app_data_dir()
                        .expect("应能解析应用数据目录")
                        .join("stock_data.db"),
                };

                let pool = open_pool(&db_path)
                    .await
                    .expect("Failed to create database pool");

                // 迁移脚本已编进二进制（见 db::connection::MIGRATIONS），不再读盘——
                // 装机后 migrations/ 不在 CWD 里，原来的读盘写法会静默跳过、一张表都不建。
                run_migrations(&pool).await.expect("数据库迁移失败");

                app.manage(pool);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
