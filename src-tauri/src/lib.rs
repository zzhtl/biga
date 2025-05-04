mod api;
mod commands;
mod constants;
mod csv;
mod db;
mod error;
mod prediction;
use commands::stock::{get_stock_infos, refresh_stock_infos};
use commands::stock_historical::{get_historical_data, refresh_historical_data};
use commands::stock_list::get_stock_list;
use commands::stock_realtime::get_realtime_data;
use commands::stock_prediction::{train_stock_prediction_model, predict_stock_price};
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use std::env;
use std::error::Error;
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};
use std::path::Path;
use std::fs;
use tauri::{
    plugin::TauriPlugin, AppHandle, Runtime, State, Window
};

mod models;
mod stock_prediction;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    Target::new(TargetKind::Stdout),
                    Target::new(TargetKind::LogDir { file_name: None }),
                    Target::new(TargetKind::Webview),
                ])
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            get_stock_list,
            get_stock_infos,
            get_realtime_data,
            refresh_stock_infos,
            get_historical_data,
            refresh_historical_data,
            train_stock_prediction_model,
            predict_stock_price,
            list_models,
            delete_model,
            train_candle_model,
            predict_with_candle,
            retrain_candle_model,
            evaluate_candle_model
        ])
        .setup(|app| -> Result<(), Box<dyn Error>> {
            tauri::async_runtime::block_on(async {
                //dotenv().ok(); // 函数会尝试从项目根目录的 .env`` 文件中读取键值对，并将其注入到进程的环境变量中
                //let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
                let pool = create_optimized_pool().await?;
                
                // 确保应用数据目录存在
                let app_dir = app.path().app_data_dir().unwrap_or_else(|_| {
                    Path::new("./data").to_path_buf()
                });
                
                // 确保 migrations 目录存在且存放迁移脚本
                let migrations_dir = app_dir.join("migrations");
                if !migrations_dir.exists() {
                    fs::create_dir_all(&migrations_dir)?;
                }

                // 复制迁移文件
                let migration_files = ["01_create_tables.sql", "02_stock_prediction_model.sql"];
                for file in &migration_files {
                    let source_path = Path::new("migrations").join(file);
                    if source_path.exists() {
                        let target_path = migrations_dir.join(file);
                        if !target_path.exists() {
                            fs::copy(&source_path, &target_path)?;
                        }
                    }
                }

                // 执行迁移脚本
                for file in &migration_files {
                    let path = migrations_dir.join(file);
                    if path.exists() {
                        let sql = fs::read_to_string(&path)?;
                        sqlx::query(&sql).execute(&pool).await?;
                    }
                }
                
                app.manage(pool);
                Ok(())
            })
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

async fn create_optimized_pool() -> Result<Pool<Sqlite>, sqlx::Error> {
    let pool = SqlitePoolOptions::new()
        .max_connections(5) // 最大连接数
        .min_connections(2) // 最小保持的空闲连接数
        .acquire_timeout(std::time::Duration::from_secs(30)) // 获取连接超时时间
        .connect("sqlite://db/stock_data.db")
        .await?;

    // 启用 WAL 模式（类似你提供的 rusqlite 示例）
    sqlx::query("PRAGMA journal_mode=WAL;")
        .execute(&pool)
        .await?;

    Ok(pool)
}

// 股票预测模型列表
#[tauri::command]
async fn list_models(
    state: State<'_, Pool<Sqlite>>, 
    symbol: String
) -> Result<Vec<commands::stock_prediction::ModelMetadata>, String> {
    commands::stock_prediction::list_prediction_models(state, symbol).await
}

// 删除股票预测模型
#[tauri::command]
async fn delete_model(
    state: State<'_, Pool<Sqlite>>, 
    model_id: String
) -> Result<(), String> {
    commands::stock_prediction::delete_prediction_model(state, model_id).await
}

// Candle模型相关命令
#[tauri::command]
async fn train_candle_model(request: stock_prediction::TrainingRequest) -> Result<stock_prediction::TrainingResult, String> {
    stock_prediction::train_candle_model(request).await
}

#[tauri::command]
async fn predict_with_candle(request: stock_prediction::PredictionRequest) -> Result<Vec<stock_prediction::Prediction>, String> {
    stock_prediction::predict_with_candle(request).await
}

#[tauri::command]
async fn retrain_candle_model(
    model_id: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    stock_prediction::retrain_candle_model(model_id, epochs, batch_size, learning_rate).await
}

#[tauri::command]
async fn evaluate_candle_model(model_id: String) -> Result<stock_prediction::EvaluationResult, String> {
    stock_prediction::evaluate_candle_model(model_id).await
}

// 创建Tauri插件
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    tauri::plugin::Builder::new("biga")
        .invoke_handler(tauri::generate_handler![
            train_candle_model,
            predict_with_candle,
            retrain_candle_model,
            evaluate_candle_model
        ])
        .build()
}
