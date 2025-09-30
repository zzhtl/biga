pub mod api;
mod commands;
mod constants;
mod csv;
mod db;
pub mod error;
pub mod stock_prediction;

use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
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
            commands::stock_list::get_stock_list,
            commands::stock::get_stock_infos,
            commands::stock_realtime::get_realtime_data,
            commands::stock::refresh_stock_infos,
            commands::stock_historical::get_historical_data,
            commands::stock_historical::refresh_historical_data,
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
                let pool = create_optimized_pool().await
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

async fn create_optimized_pool() -> Result<Pool<Sqlite>, sqlx::Error> {
    // 获取当前工作目录并构建数据库路径
    let current_dir = std::env::current_dir()
        .map_err(sqlx::Error::Io)?;
    
    // 尝试多个可能的数据库路径
    let possible_paths = [
        current_dir.join("db/stock_data.db"),
        current_dir.join("src-tauri/db/stock_data.db"),
    ];
    
    let mut db_path = None;
    for path in &possible_paths {
        println!("检查数据库路径: {}", path.display());
        if path.exists() {
            db_path = Some(path);
            println!("✅ 找到数据库文件: {}", path.display());
            break;
        }
    }
    
    // 如果找不到数据库文件，创建一个新的
    let final_db_path = match db_path {
        Some(path) => path.clone(),
        None => {
            // 优先在项目根目录的db文件夹中创建
            let preferred_path = if current_dir.join("src-tauri").exists() {
                // 在项目根目录
                current_dir.join("db/stock_data.db")
            } else {
                // 在src-tauri目录
                current_dir.join("db/stock_data.db")
            };
            
            // 确保db目录存在
            if let Some(parent) = preferred_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(sqlx::Error::Io)?;
            }
            
            println!("📁 创建新数据库文件: {}", preferred_path.display());
            preferred_path
        }
    };
    
    let connection_string = format!("sqlite://{}", final_db_path.display());
    println!("🔗 数据库连接字符串: {connection_string}");
    
    let pool = SqlitePoolOptions::new()
        .max_connections(5) // 最大连接数
        .min_connections(2) // 最小保持的空闲连接数
        .acquire_timeout(std::time::Duration::from_secs(30)) // 获取连接超时时间
        .connect(&connection_string)
        .await?;

    // 启用 WAL 模式
    sqlx::query("PRAGMA journal_mode=WAL;")
        .execute(&pool)
        .await?;

    println!("✅ 数据库连接池创建成功");
    Ok(pool)
}
