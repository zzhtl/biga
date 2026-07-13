use crate::db::models::RealtimeData;
use crate::error::AppError;
use crate::commands::pagination::{normalize_page, PagedResponse};
use sqlx::SqlitePool;
use tauri::State;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RealtimeSortColumn {
    Symbol,
    Name,
    Volume,
    Amount,
    Change,
    ChangePercent,
}

impl RealtimeSortColumn {
    fn parse(value: &str) -> Result<Self, AppError> {
        match value {
            "symbol" => Ok(Self::Symbol),
            "name" => Ok(Self::Name),
            "volume" => Ok(Self::Volume),
            "amount" => Ok(Self::Amount),
            "change" => Ok(Self::Change),
            "change_percent" => Ok(Self::ChangePercent),
            _ => Err(AppError::InvalidInput("不支持的行情排序字段".to_string())),
        }
    }

    fn as_sql(self) -> &'static str {
        match self {
            Self::Symbol => "symbol",
            Self::Name => "name",
            Self::Volume => "volume",
            Self::Amount => "amount",
            Self::Change => "change",
            Self::ChangePercent => "change_percent",
        }
    }
}

fn sort_direction(value: &str) -> Result<&'static str, AppError> {
    match value {
        "asc" => Ok("ASC"),
        "desc" => Ok("DESC"),
        _ => Err(AppError::InvalidInput("不支持的行情排序方向".to_string())),
    }
}

#[tauri::command]
pub async fn get_realtime_data(
    search: String,
    column: String,
    sort: String,
    page: u32,
    page_size: u32,
    pool: State<'_, SqlitePool>,
) -> Result<PagedResponse<RealtimeData>, AppError> {
    let column = RealtimeSortColumn::parse(&column)?.as_sql();
    let sort = sort_direction(&sort)?;
    let order_by = format!("{column} {sort}");
    let (page, page_size, offset) = normalize_page(page, page_size);
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
    let search = search.trim();
    let (records, total) = if search.is_empty() {
        let total = sqlx::query_scalar::<_, i64>(&format!(
            "SELECT COUNT(*) FROM ({projection})"
        ))
        .fetch_one(&*pool)
        .await?;
        sqlx::query_as::<_, RealtimeData>(&format!(
            "SELECT * FROM ({projection}) ORDER BY {order_by} LIMIT ? OFFSET ?"
        ))
        .bind(i64::from(page_size))
        .bind(offset)
        .fetch_all(&*pool)
        .await
        .map(|records| (records, total))?
    } else {
        let search_pattern = format!("%{search}%");
        let total = sqlx::query_scalar::<_, i64>(&format!(
            "SELECT COUNT(*) FROM ({projection}) WHERE name LIKE ? OR symbol LIKE ?"
        ))
        .bind(search_pattern.clone())
        .bind(search_pattern.clone())
        .fetch_one(&*pool)
        .await?;
        sqlx::query_as::<_, RealtimeData>(&format!(
            "SELECT * FROM ({projection}) WHERE name LIKE ? OR symbol LIKE ? ORDER BY {order_by} LIMIT ? OFFSET ?"
        ))
        .bind(search_pattern.clone())
        .bind(search_pattern)
        .bind(i64::from(page_size))
        .bind(offset)
        .fetch_all(&*pool)
        .await
        .map(|records| (records, total))?
    };
    Ok(PagedResponse {
        data: records,
        total,
        page,
        page_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_known_sort_values() {
        assert!(matches!(
            RealtimeSortColumn::parse("change_percent"),
            Ok(RealtimeSortColumn::ChangePercent)
        ));
        assert!(RealtimeSortColumn::parse("change_percent; DROP TABLE stock").is_err());
        assert!(matches!(sort_direction("asc"), Ok("ASC")));
        assert!(sort_direction("sideways").is_err());
    }
}
