use std::path::PathBuf;
use sqlx::{sqlite::SqlitePoolOptions, Row};
use chrono::Local;
use crate::stock_prediction::types::HistoricalDataType;

// 数据库路径查找函数
pub fn find_database_path() -> Option<PathBuf> {
    let current_dir = std::env::current_dir().ok()?;
    
    // 尝试多个可能的数据库路径
    let possible_paths = [
        current_dir.join("db/stock_data.db"),
        current_dir.join("src-tauri/db/stock_data.db"),
        current_dir.parent()?.join("src-tauri/db/stock_data.db"), // 如果在 src-tauri 目录内运行
    ];
    
    for path in &possible_paths {
        println!("检查数据库路径: {}", path.display());
        if path.exists() {
            println!("✅ 找到数据库文件: {}", path.display());
            return Some(path.clone());
        }
    }
    
    println!("❌ 未找到数据库文件");
    None
}

// 从数据库获取历史数据
pub async fn get_historical_data_from_db(symbol: &str, start_date: &str, end_date: &str) -> Result<Vec<HistoricalDataType>, String> {
    // 创建一个临时的数据库连接
    let db_path = find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    
    let connection_string = format!("sqlite://{}", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("连接数据库失败: {e}"))?;
    
    let records = sqlx::query_as::<_, HistoricalDataType>(
        r#"SELECT * FROM historical_data 
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date ASC"#
    )
    .bind(symbol)
    .bind(start_date)
    .bind(end_date)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("查询历史数据失败: {e}"))?;
    
    Ok(records)
}

// 从数据库获取最近的市场数据
pub async fn get_recent_market_data(symbol: &str, days: usize) -> Result<(f64, f64, Vec<String>, Vec<f64>, Vec<i64>, Vec<f64>, Vec<f64>), String> {
    // 计算开始日期（大幅增加数据获取范围，确保有足够的历史数据进行技术分析）
    let end_date = Local::now().naive_local().date();
    let buffer_days = 60; // 增加缓冲期到60天，应对节假日
    // 至少获取1年的数据，或者用户指定天数+缓冲期，取更大值
    let total_days = std::cmp::max(365, days + buffer_days); 
    let start_date = end_date - chrono::Duration::days(total_days as i64);
    
    // 使用动态数据库路径查找
    let db_path = find_database_path()
        .ok_or_else(|| "找不到数据库文件".to_string())?;
    
    let connection_string = format!("sqlite://{}", db_path.display());
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect(&connection_string)
        .await
        .map_err(|e| format!("连接数据库失败: {e}"))?;
    
    // 修改查询，获取更多历史数据但保持合理的限制
    let limit = std::cmp::max(300, days * 2); // 至少300条记录，或者请求天数的2倍
    let records = sqlx::query_as::<_, HistoricalDataType>(
        r#"SELECT * FROM historical_data 
           WHERE symbol = ? AND date BETWEEN ? AND ?
           ORDER BY date DESC
           LIMIT ?"#
    )
    .bind(symbol)
    .bind(start_date.format("%Y-%m-%d").to_string())
    .bind(end_date.format("%Y-%m-%d").to_string())
    .bind(limit as i32)
    .fetch_all(&pool)
    .await
    .map_err(|e| format!("查询历史数据失败: {e}"))?;
    
    if records.is_empty() {
        return Err(format!("未找到股票代码 {symbol} 的历史数据"));
    }
    
    // 反向排序以获取时间顺序（从旧到新）
    let mut sorted_records = records;
    sorted_records.reverse();
    
    // 提取数据
    let dates: Vec<String> = sorted_records.iter().map(|r| r.date.clone()).collect();
    let prices: Vec<f64> = sorted_records.iter().map(|r| r.close).collect();
    let volumes: Vec<i64> = sorted_records.iter().map(|r| r.volume).collect();
    let highs: Vec<f64> = sorted_records.iter().map(|r| r.high).collect();
    let lows: Vec<f64> = sorted_records.iter().map(|r| r.low).collect();
    
    // 获取最新价格
    let current_price = prices.last().copied().unwrap_or(0.0);
    
    // 计算最新的涨跌幅
    let current_change_percent = if prices.len() >= 2 {
        let previous_price = prices[prices.len() - 2];
        if previous_price > 0.0 {
            (current_price - previous_price) / previous_price * 100.0
        } else {
            0.0
        }
    } else {
        // 如果没有足够的数据计算涨跌幅，则默认为0
        0.0
    };
    
    println!("📊 获取到{}条历史数据用于预测，时间范围: {} 到 {}", 
             sorted_records.len(),
             sorted_records.first().map(|r| &r.date).unwrap_or(&"未知".to_string()),
             sorted_records.last().map(|r| &r.date).unwrap_or(&"未知".to_string()));
    println!("📈 最新价格: {current_price:.2}, 涨跌幅: {current_change_percent:.2}%");
    
    Ok((current_price, current_change_percent, dates, prices, volumes, highs, lows))
}

// 实现FromRow特征，使其可以从数据库行转换
impl<'r> sqlx::FromRow<'r, sqlx::sqlite::SqliteRow> for HistoricalDataType {
    fn from_row(row: &'r sqlx::sqlite::SqliteRow) -> Result<Self, sqlx::Error> {
        Ok(Self {
            date: row.try_get("date")?,
            open: row.try_get("open")?,
            close: row.try_get("close")?,
            high: row.try_get("high")?,
            low: row.try_get("low")?,
            volume: row.try_get("volume")?,
            change_percent: row.try_get("change_percent")?,
        })
    }
} 