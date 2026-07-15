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

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::SqlitePool;

    async fn run_migration(pool: &SqlitePool, sql: &str) {
        for statement in sql.split(';') {
            let statement = statement.trim();
            if statement.is_empty() {
                continue;
            }
            sqlx::query(statement)
                .execute(pool)
                .await
                .expect("股票代码迁移应执行成功");
        }
    }

    async fn run_symbol_migration(pool: &SqlitePool) {
        run_migration(
            pool,
            include_str!("../../migrations/08_canonical_stock_symbols.sql"),
        )
        .await;
    }

    #[tokio::test]
    async fn canonical_symbol_migration_merges_duplicates_and_prevents_recurrence() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("应创建内存 SQLite");

        for sql in [
            include_str!("../../migrations/01_create_tables.sql"),
            include_str!("../../migrations/03_volume_metrics.sql"),
            include_str!("../../migrations/04_stock_fundamentals.sql"),
            include_str!("../../migrations/05_capital_valuation.sql"),
            include_str!("../../migrations/06_stock_category.sql"),
            include_str!("../../migrations/07_watchlist.sql"),
        ] {
            run_migration(&pool, sql).await;
        }

        for statement in [
            "INSERT INTO stock_info VALUES ('002466', '002466', 'sz')",
            "INSERT INTO stock_info VALUES ('002466.SZ', '天齐锂业', 'SZ')",
            "INSERT INTO stock VALUES ('002466.SZ', '天齐锂业', '四川', '小金属', '主板', 'SZ', '20100831', NULL, NULL, '能源金属')",
            "INSERT INTO historical_data VALUES ('002466', '2026-07-15', 47, 47.43, 48, 46, 1, 200, 1, 1, 1, 1, 1)",
            "INSERT INTO historical_data VALUES ('002466.SZ', '2026-07-15', 47, 47.43, 48, 46, 1, 100, 1, 1, 1, 1, 1)",
            "INSERT INTO historical_data VALUES ('002466.SZ', '2026-07-14', 46, 47, 48, 45, 1, 150, 1, 1, 1, 1, 1)",
            "INSERT INTO realtime_data VALUES ('002466', '天齐锂业', '2026-07-14', 47, 1, 100, 1, 1, 1, 1, 1)",
            "INSERT INTO realtime_data VALUES ('002466.SZ', '天齐锂业', '2026-07-15', 48, 1, 200, 1, 1, 1, 1, 1)",
            "INSERT INTO stock_capital VALUES ('002466', 100, 120, 4700, '2026-07-14 08:00:00', 10, 1.5)",
            "INSERT INTO stock_capital VALUES ('002466.SZ', 110, 130, 5200, '2026-07-15 08:00:00', 11, 1.6)",
            "INSERT INTO stock_fundamentals VALUES ('002466.SZ', '2026-03-31', 1, 2, 3, 4, 5, 6, '2026-07-15 08:00:00')",
            "INSERT INTO watchlist VALUES ('002466', '2026-07-15 09:00:00', 2)",
            "INSERT INTO watchlist VALUES ('002466.SZ', '2026-07-14 09:00:00', 1)",
        ] {
            sqlx::query(statement)
                .execute(&pool)
                .await
                .expect("应插入重复代码测试数据");
        }

        run_symbol_migration(&pool).await;
        run_symbol_migration(&pool).await;

        let stock_info: Vec<(String, String, String)> =
            sqlx::query_as("SELECT symbol, name, exchange FROM stock_info")
                .fetch_all(&pool)
                .await
                .expect("应查询清理后的股票信息");
        assert_eq!(
            stock_info,
            vec![(
                "002466".to_string(),
                "天齐锂业".to_string(),
                "sz".to_string()
            )]
        );

        let historical: Vec<(String, String, f64)> =
            sqlx::query_as("SELECT symbol, date, amount FROM historical_data ORDER BY date")
                .fetch_all(&pool)
                .await
                .expect("应查询合并后的历史数据");
        assert_eq!(historical.len(), 2);
        assert!(historical.iter().all(|(symbol, _, _)| symbol == "002466"));
        assert_eq!(
            historical[1],
            ("002466".to_string(), "2026-07-15".to_string(), 200.0)
        );

        let realtime: (String, String, f64) =
            sqlx::query_as("SELECT symbol, date, close FROM realtime_data")
                .fetch_one(&pool)
                .await
                .expect("应查询合并后的实时数据");
        assert_eq!(
            realtime,
            ("002466".to_string(), "2026-07-15".to_string(), 48.0)
        );

        let capital: (String, f64) = sqlx::query_as("SELECT symbol, pe FROM stock_capital")
            .fetch_one(&pool)
            .await
            .expect("应查询合并后的股本数据");
        assert_eq!(capital, ("002466".to_string(), 11.0));

        for table in ["stock", "stock_fundamentals", "watchlist"] {
            let count: i64 = sqlx::query_scalar(&format!(
                "SELECT COUNT(*) FROM {table} WHERE symbol = '002466'",
            ))
            .fetch_one(&pool)
            .await
            .expect("关联表应完成代码归一化");
            assert_eq!(count, 1, "{table} 应只保留规范代码");
        }

        let duplicate = sqlx::query(
            "INSERT INTO stock_info (symbol, name, exchange) VALUES ('002466.SZ', '天齐锂业', 'SZ')",
        )
        .execute(&pool)
        .await;
        assert!(duplicate.is_err(), "数据库唯一索引应拒绝逻辑重复代码");
    }
}
