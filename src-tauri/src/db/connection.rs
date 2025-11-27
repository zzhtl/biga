//! 数据库连接管理

use sqlx::{Pool, Sqlite, sqlite::SqlitePoolOptions};
use std::path::PathBuf;
use std::fs;

/// 数据库连接池类型
pub type DbPool = Pool<Sqlite>;

/// 查找数据库路径
pub fn find_database_path() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    
    let possible_paths = [
        current_dir.join("db/stock_data.db"),
        current_dir.join("src-tauri/db/stock_data.db"),
    ];
    
    for path in &possible_paths {
        if path.exists() {
            return Some(path.clone());
        }
    }
    
    None
}

/// 创建数据库连接池
pub async fn create_pool() -> Result<DbPool, sqlx::Error> {
    let current_dir = std::env::current_dir().map_err(sqlx::Error::Io)?;
    
    let possible_paths = [
        current_dir.join("db/stock_data.db"),
        current_dir.join("src-tauri/db/stock_data.db"),
    ];
    
    let mut db_path = None;
    for path in &possible_paths {
        if path.exists() {
            db_path = Some(path.clone());
            break;
        }
    }
    
    let final_db_path = match db_path {
        Some(path) => path,
        None => {
            let preferred_path = if current_dir.join("src-tauri").exists() {
                current_dir.join("db/stock_data.db")
            } else {
                current_dir.join("db/stock_data.db")
            };
            
            if let Some(parent) = preferred_path.parent() {
                fs::create_dir_all(parent).map_err(sqlx::Error::Io)?;
            }
            
            preferred_path
        }
    };
    
    let connection_string = format!("sqlite://{}", final_db_path.display());
    
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .min_connections(2)
        .acquire_timeout(std::time::Duration::from_secs(30))
        .connect(&connection_string)
        .await?;
    
    // 启用 WAL 模式
    sqlx::query("PRAGMA journal_mode=WAL;")
        .execute(&pool)
        .await?;
    
    Ok(pool)
}

/// 创建临时数据库连接
pub async fn create_temp_pool() -> Result<DbPool, String> {
    let db_path = find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    
    let connection_string = format!("sqlite://{}", db_path.display());
    
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("数据库连接失败: {e}"))
}

