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
pub mod prediction;
pub mod services;

// 命令层
mod commands;

// CSV 处理
mod csv;

use db::connection::create_pool;
use std::path::Path;
use std::fs;
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
            commands::stock_prediction::predict_with_technical_only
        ])
        .setup(|app| {
            tauri::async_runtime::block_on(async {
                let pool = create_pool().await
                    .expect("Failed to create database pool");
                
                // 执行迁移脚本
                let migration_files = ["01_create_tables.sql", "02_stock_prediction_model.sql"];
                for file in &migration_files {
                    let path = Path::new("migrations").join(file);
                    if path.exists() {
                        let sql = fs::read_to_string(&path)
                            .expect("Failed to read migration file");
                        sqlx::query(&sql).execute(&pool).await
                            .expect("Failed to execute migration");
                    }
                }
                
                app.manage(pool);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
