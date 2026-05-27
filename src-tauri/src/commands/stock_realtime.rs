use crate::db::models::RealtimeData;
use crate::error::AppError;
use sqlx::SqlitePool;
use tauri::State;

#[tauri::command]
pub async fn get_realtime_data(
    search: String,
    column: String,
    sort: String,
    pool: State<'_, SqlitePool>,
) -> Result<Vec<RealtimeData>, AppError> {
    let order_by = format!("{column} {sort}");
    // realtime_data.name 历史上存在脏数据（批量入库时把代码当占位名称写入），
    // 真实名称在 stock_info 带交易所后缀的行里。这里用关联子查询按 6 位代码回退到
    // 正确名称（优先取 name<>symbol 的权威行），保证 search/sort 仍作用在解析后的名称上。
    let projection = r#"
        SELECT
            COALESCE(r.symbol, '') as symbol,
            COALESCE(
                (SELECT s.name FROM stock_info s
                   WHERE substr(s.symbol, 1, 6) = substr(r.symbol, 1, 6)
                     AND s.name <> '' AND s.name <> s.symbol
                   LIMIT 1),
                NULLIF(r.name, r.symbol),
                r.name,
                ''
            ) as name,
            r.date as date,
            COALESCE(r.close, 0.00) as close,
            COALESCE(r.volume, 0) as volume,
            COALESCE(r.amount, 0.00) as amount,
            COALESCE(r.amplitude, 0.00) as amplitude,
            COALESCE(r.turnover_rate, 0.00) as turnover_rate,
            COALESCE(r.volume_ratio, 0.00) as volume_ratio,
            COALESCE(r.change_percent, 0.00) as change_percent,
            COALESCE(r.change, 0.00) as change
        FROM realtime_data r
    "#;
    let records = if search.trim().is_empty() {
        sqlx::query_as::<_, RealtimeData>(&format!(
            "SELECT * FROM ({projection}) ORDER BY {order_by}"
        ))
        .fetch_all(&*pool)
        .await?
    } else {
        let search_pattern = format!("%{search}%");
        sqlx::query_as::<_, RealtimeData>(&format!(
            "SELECT * FROM ({projection}) WHERE name LIKE ? OR symbol LIKE ? ORDER BY {order_by}"
        ))
        .bind(search_pattern.clone())
        .bind(search_pattern)
        .fetch_all(&*pool)
        .await?
    };
    Ok(records)
}
