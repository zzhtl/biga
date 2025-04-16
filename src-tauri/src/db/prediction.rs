use crate::db::models::{StockPrediction, StockPredictionModel, StockPredictionModelInfo};
use crate::error::AppError;
use chrono::{Local, NaiveDate};
use sqlx::{Pool, Sqlite, Row};

pub async fn save_model(
    pool: &Pool<Sqlite>,
    symbol: &str,
    model_name: &str,
    model_type: &str,
    model_data: Vec<u8>,
    parameters: &str,
    metrics: &str,
) -> Result<i64, AppError> {
    let now = Local::now().naive_local();
    
    let mut tx = pool.begin().await?;
    
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO stock_prediction_models 
        (symbol, model_name, model_type, created_at, updated_at, model_data, parameters, metrics) "
    );
    
    let values = [(symbol, model_name, model_type, &now, &now, &model_data, parameters, metrics)];
    
    query_builder.push_values(
        &values, 
        |mut b, (sym, m_name, m_type, cr_at, up_at, data, params, mtx)| {
            b.push_bind(sym)
             .push_bind(m_name)
             .push_bind(m_type)
             .push_bind(cr_at)
             .push_bind(up_at)
             .push_bind(data)
             .push_bind(params)
             .push_bind(mtx);
        }
    );
    
    query_builder.push(
        " ON CONFLICT(symbol, model_name) DO UPDATE SET
            model_type = excluded.model_type,
            updated_at = excluded.updated_at,
            model_data = excluded.model_data,
            parameters = excluded.parameters,
            metrics = excluded.metrics
        RETURNING id"
    );
    
    let result = query_builder.build()
        .fetch_one(&mut *tx)
        .await?;
    
    tx.commit().await?;
    
    let id: i64 = result.get("id");
    
    Ok(id)
}

pub async fn get_model(
    pool: &Pool<Sqlite>,
    model_id: i64,
) -> Result<StockPredictionModel, AppError> {
    let model = sqlx::query_as::<_, StockPredictionModel>(
        r#"
        SELECT * FROM stock_prediction_models WHERE id = ?
        "#
    )
    .bind(model_id)
    .fetch_one(pool)
    .await?;
    
    Ok(model)
}

pub async fn get_model_by_symbol_and_name(
    pool: &Pool<Sqlite>,
    symbol: &str,
    model_name: &str,
) -> Result<StockPredictionModel, AppError> {
    let model = sqlx::query_as::<_, StockPredictionModel>(
        r#"
        SELECT * FROM stock_prediction_models 
        WHERE symbol = ? AND model_name = ?
        "#
    )
    .bind(symbol)
    .bind(model_name)
    .fetch_one(pool)
    .await?;
    
    Ok(model)
}

pub async fn list_models_for_symbol(
    pool: &Pool<Sqlite>,
    symbol: &str,
) -> Result<Vec<StockPredictionModelInfo>, AppError> {
    let rows = sqlx::query(
        r#"
        SELECT 
            id, symbol, model_name, model_type, 
            created_at, updated_at, parameters, metrics
        FROM stock_prediction_models 
        WHERE symbol = ?
        ORDER BY updated_at DESC
        "#
    )
    .bind(symbol)
    .fetch_all(pool)
    .await?;
    
    let models = rows.into_iter()
        .map(|row| StockPredictionModelInfo {
            id: row.get("id"),
            symbol: row.get("symbol"),
            model_name: row.get("model_name"),
            model_type: row.get("model_type"),
            created_at: row.get("created_at"),
            updated_at: row.get("updated_at"),
            parameters: row.get("parameters"),
            metrics: row.get("metrics"),
        })
        .collect();
    
    Ok(models)
}

pub async fn save_prediction(
    pool: &Pool<Sqlite>,
    symbol: &str,
    model_id: i64,
    target_date: NaiveDate,
    predicted_price: f64,
    predicted_change_percent: f64,
    confidence: f64,
    features_used: &str,
) -> Result<i64, AppError> {
    let prediction_date = Local::now().date_naive();
    let now = Local::now().naive_local();
    
    let mut tx = pool.begin().await?;
    
    let mut query_builder = sqlx::QueryBuilder::new(
        "INSERT INTO stock_predictions
        (symbol, model_id, prediction_date, target_date, predicted_price, 
         predicted_change_percent, confidence, features_used, created_at) "
    );
    
    let values = [(symbol, model_id, &prediction_date, &target_date, predicted_price, 
        predicted_change_percent, confidence, features_used, &now)];
        
    query_builder.push_values(
        &values, 
        |mut b, (sym, m_id, pred_date, targ_date, pred_price, 
                 pred_change, conf, feat, created)| {
            b.push_bind(sym)
             .push_bind(m_id)
             .push_bind(pred_date)
             .push_bind(targ_date)
             .push_bind(pred_price)
             .push_bind(pred_change)
             .push_bind(conf)
             .push_bind(feat)
             .push_bind(created);
        }
    );
    
    query_builder.push(
        " ON CONFLICT(symbol, model_id, target_date) DO UPDATE SET
            prediction_date = excluded.prediction_date,
            predicted_price = excluded.predicted_price,
            predicted_change_percent = excluded.predicted_change_percent,
            confidence = excluded.confidence,
            features_used = excluded.features_used,
            created_at = excluded.created_at
        RETURNING id"
    );
    
    let result = query_builder.build()
        .fetch_one(&mut *tx)
        .await?;
    
    tx.commit().await?;
    
    let id: i64 = result.get("id");
    
    Ok(id)
}

pub async fn get_predictions_for_symbol(
    pool: &Pool<Sqlite>,
    symbol: &str,
    from_date: Option<NaiveDate>,
    to_date: Option<NaiveDate>,
) -> Result<Vec<StockPrediction>, AppError> {
    let from_date = from_date.unwrap_or_else(|| Local::now().date_naive());
    let to_date = to_date.unwrap_or_else(|| from_date.checked_add_days(chrono::Days::new(30)).unwrap());
    
    let predictions = sqlx::query_as::<_, StockPrediction>(
        r#"
        SELECT * FROM stock_predictions 
        WHERE symbol = ? AND target_date BETWEEN ? AND ?
        ORDER BY target_date ASC
        "#
    )
    .bind(symbol)
    .bind(from_date)
    .bind(to_date)
    .fetch_all(pool)
    .await?;
    
    Ok(predictions)
}

pub async fn delete_model(
    pool: &Pool<Sqlite>,
    model_id: i64,
) -> Result<(), AppError> {
    let mut tx = pool.begin().await?;
    
    sqlx::query(
        r#"
        DELETE FROM stock_predictions WHERE model_id = ?
        "#
    )
    .bind(model_id)
    .execute(&mut *tx)
    .await?;
    
    sqlx::query(
        r#"
        DELETE FROM stock_prediction_models WHERE id = ?
        "#
    )
    .bind(model_id)
    .execute(&mut *tx)
    .await?;
    
    tx.commit().await?;
    
    Ok(())
} 