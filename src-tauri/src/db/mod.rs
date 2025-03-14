pub mod models;
use crate::constants;
use crate::error::AppError;
use models::{HistoricalData, StockInfo};
use sqlx::sqlite::SqlitePool;
use sqlx::QueryBuilder;

pub async fn batch_insert_stock_info(
    pool: &SqlitePool,
    data_list: Vec<StockInfo>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }
    let mut tx = pool.begin().await?;
    let mut affected_rows = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
        let mut query_builder =
            QueryBuilder::new("INSERT INTO stock_info (symbol, name, exchange) ");
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(&data.name)
                .push_bind(&data.exchange);
        });
        query_builder.push(" ON CONFLICT(symbol) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;

        affected_rows += result.rows_affected();
    }
    tx.commit().await?;

    Ok(affected_rows)
}

pub async fn batch_insert_historical_data(
    pool: &SqlitePool,
    data_list: Vec<HistoricalData>,
) -> Result<u64, AppError> {
    if data_list.is_empty() {
        return Ok(0);
    }

    let mut tx = pool.begin().await?;

    // 批量生成占位符（自动处理参数展开）
    let mut batch_size = 0;
    for chunk in data_list.chunks(constants::BATCH_SIZE) {
        let mut query_builder = QueryBuilder::new(
            "INSERT INTO historical_data (symbol, date, open, close, high, low, volume) ",
        );
        query_builder.push_values(chunk, |mut b, data| {
            b.push_bind(&data.symbol)
                .push_bind(&data.date)
                .push_bind(&data.open)
                .push_bind(&data.close)
                .push_bind(&data.high)
                .push_bind(&data.low)
                .push_bind(&data.volume);
        });

        query_builder.push(" ON CONFLICT(symbol, date) DO NOTHING");
        let result = query_builder.build().execute(&mut *tx).await?;
        batch_size += result.rows_affected();
    }
    tx.commit().await?;

    Ok(batch_size)
}

// async fn batch_insert(pool: &SqlitePool, users: Vec<User>) -> Result<(), sqlx::Error> {
//     let mut query_builder = QueryBuilder::new("INSERT INTO users (id, name, email) ");

//     query_builder.push_values(users, |mut b, user| {
//         b.push_bind(user.id)
//             .push_bind(user.name)
//             .push_bind(user.email);
//     });
//     query_builder.push(" ON CONFLICT (id) DO NOTHING"); // 关键冲突处理

//     let query = query_builder.build();
//     query.execute(pool).await?;
//     Ok(())
// }

// async fn transactional_batch_insert(
//     pool: &SqlitePool,
//     users: Vec<User>,
//     batch_size: usize,
// ) -> Result<(), sqlx::Error> {
//     let mut tx = pool.begin().await?;

//     for chunk in users.chunks(batch_size) {
//         let mut query_builder = QueryBuilder::new("INSERT INTO users (id, name, email) ");

//         query_builder.push_values(chunk, |mut b, user| {
//             b.push_bind(user.id)
//                 .push_bind(&user.name)
//                 .push_bind(&user.email);
//         });

//         query_builder.push(" ON CONFLICT (id) DO NOTHING"); // 关键冲突处理
//         query_builder.build().execute(&mut *tx).await?;
//     }

//     tx.commit().await?;
//     Ok(())
// }
