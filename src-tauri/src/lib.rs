mod api;
mod commands;
mod db;
mod error;
use commands::stock_historical::get_historical_data;
use dotenv::dotenv;
use sqlx::sqlite::SqlitePool;
use std::env;
use std::error::Error;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![get_historical_data])
        .setup(|app| -> Result<(), Box<dyn Error>> {
            tauri::async_runtime::block_on(async {
                dotenv().ok(); // 函数会尝试从项目根目录的 .env`` 文件中读取键值对，并将其注入到进程的环境变量中
                let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
                println!("Database URL: {}", db_url);
                let pool = SqlitePool::connect(&db_url).await?;
                // sqlx migrate run 执行迁移脚本
                sqlx::migrate!("./migrations").run(&pool).await?;
                app.manage(pool);
                Ok(())
            })
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
