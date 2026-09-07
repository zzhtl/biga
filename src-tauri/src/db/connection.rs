//! 数据库连接管理

use sqlx::{Pool, Sqlite, sqlite::{SqliteConnectOptions, SqlitePoolOptions}};
use std::path::{Path, PathBuf};
use std::fs;

/// 数据库连接池类型
pub type DbPool = Pool<Sqlite>;

/// 按给定路径打开连接池，库文件不存在时创建它。
///
/// sqlx 的 `connect()` 默认 `create_if_missing = false`——文件不存在直接报
/// `SqliteError { code: 14, "unable to open database file" }`，在 `lib.rs` 的 setup 里
/// 是 `.expect()`，所以全新环境（新 clone / 换机器 / 删过 db 目录）首次启动会直接 panic。
///
/// 另外用 [`SqliteConnectOptions::filename`] 而不是拼 `sqlite://{path}` 连接串：
/// 路径里出现空格、`?`、`#` 时 URL 解析会出错，而用户目录含空格并不罕见。
pub async fn open_pool(db_path: &Path) -> Result<DbPool, sqlx::Error> {
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).map_err(sqlx::Error::Io)?;
    }

    let options = SqliteConnectOptions::new()
        .filename(db_path)
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .min_connections(2)
        .acquire_timeout(std::time::Duration::from_secs(30))
        .connect_with(options)
        .await?;

    // 启用 WAL 模式
    sqlx::query("PRAGMA journal_mode=WAL;")
        .execute(&pool)
        .await?;

    Ok(pool)
}

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
    
    // 两个候选都不存在（全新环境）时落到第一个；目录与文件由 open_pool 负责创建
    let final_db_path = db_path.unwrap_or_else(|| current_dir.join("db/stock_data.db"));

    open_pool(&final_db_path).await
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

    /// 一次性临时目录。用完即删，避免测试之间互相看到对方的库文件。
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("biga_{tag}_{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&path).expect("应创建临时目录");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `create_pool` 依赖进程级的 current_dir，测试里改它会和并行用例打架，
    /// 所以连接逻辑抽到 `open_pool`，这几个用例直接钉它。
    #[tokio::test]
    async fn open_pool_creates_the_database_file_and_its_parent_directory() {
        let dir = TempDir::new("create_if_missing");
        // 连父目录都不存在，模拟全新 clone
        let db_path = dir.0.join("db").join("stock_data.db");
        assert!(!db_path.exists());

        let pool = open_pool(&db_path)
            .await
            .expect("库文件不存在时应自动创建，而不是报 unable to open database file");

        assert!(db_path.exists(), "应在磁盘上真的建出库文件");
        sqlx::query("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
            .execute(&pool)
            .await
            .expect("新建的库应可写");

        let journal: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await
            .expect("应能读取日志模式");
        assert_eq!(journal.0.to_lowercase(), "wal", "WAL 模式应仍然生效");
    }

    #[tokio::test]
    async fn open_pool_reuses_an_existing_database_without_wiping_it() {
        let dir = TempDir::new("reuse");
        let db_path = dir.0.join("stock_data.db");

        let first = open_pool(&db_path).await.expect("首次应创建");
        sqlx::query("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
            .execute(&first)
            .await
            .unwrap();
        sqlx::query("INSERT INTO probe (id) VALUES (42)")
            .execute(&first)
            .await
            .unwrap();
        first.close().await;

        let second = open_pool(&db_path).await.expect("再次打开应复用同一个文件");
        let id: (i64,) = sqlx::query_as("SELECT id FROM probe")
            .fetch_one(&second)
            .await
            .expect("原有数据必须还在——create_if_missing 不能变成每次重建");
        assert_eq!(id.0, 42);
    }

    #[tokio::test]
    async fn open_pool_accepts_paths_with_spaces() {
        // 旧实现拼 `sqlite://{path}` 连接串，路径含空格会被 URL 解析拒掉；
        // 而「我的文档」这类带空格的用户目录很常见。
        let dir = TempDir::new("with space");
        let db_path = dir.0.join("stock data.db");
        let pool = open_pool(&db_path).await.expect("带空格的路径也应能打开");
        assert!(db_path.exists());
        pool.close().await;
    }

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

    /// 按 `lib.rs` 的 `migration_files` 真实顺序跑全部迁移，并跑两遍。
    ///
    /// 迁移执行器用 `sql.split(';')` 朴素拆分、只忽略 "duplicate column name"，
    /// 所以任何新迁移都必须：无分号字面量、无 TRIGGER、每条语句幂等。
    /// 新增迁移文件时把它加进下面的数组——忘了加，这个测试不会失败，
    /// 但 `lib.rs:91` 的数组会漏掉它，功能在真机上直接不存在。
    #[tokio::test]
    async fn app_migration_sequence_is_idempotent() {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("应创建内存 SQLite");

        let migrations = [
            include_str!("../../migrations/01_create_tables.sql"),
            include_str!("../../migrations/02_stock_prediction_model.sql"),
            include_str!("../../migrations/03_volume_metrics.sql"),
            include_str!("../../migrations/04_stock_fundamentals.sql"),
            include_str!("../../migrations/05_capital_valuation.sql"),
            include_str!("../../migrations/06_stock_category.sql"),
            include_str!("../../migrations/07_watchlist.sql"),
            include_str!("../../migrations/08_canonical_stock_symbols.sql"),
            include_str!("../../migrations/09_trading_discipline.sql"),
        ];

        for round in 1..=2 {
            for sql in migrations {
                for statement in sql.split(';') {
                    let statement = statement.trim();
                    if statement.is_empty() {
                        continue;
                    }
                    if let Err(e) = sqlx::query(statement).execute(&pool).await {
                        let message = e.to_string();
                        if message.contains("duplicate column name") {
                            continue;
                        }
                        panic!("第 {round} 遍迁移失败: {e}\n{statement}");
                    }
                }
            }
        }

        let tables: Vec<(String,)> =
            sqlx::query_as("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
                .fetch_all(&pool)
                .await
                .expect("应能列出建好的表");
        let names: Vec<String> = tables.into_iter().map(|(name,)| name).collect();
        for expected in [
            "discipline_account",
            "discipline_events",
            "positions",
            "trades",
            "watchlist",
            "historical_data",
        ] {
            assert!(names.contains(&expected.to_string()), "{expected} 表应存在");
        }

        let seeded: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM discipline_account")
            .fetch_one(&pool)
            .await
            .expect("应能统计账户行");
        assert_eq!(seeded, 1, "账户种子行必须恰好一条，重复迁移不能插第二条");
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
