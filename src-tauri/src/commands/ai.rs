use crate::{ai::{process, service::{self, AiState}, types::*}, db::DbPool};
use std::sync::Arc;
use tauri::{AppHandle, State};

#[tauri::command]
pub async fn list_ai_connections(pool: State<'_, DbPool>) -> Result<Vec<AiConnectionConfig>, String> { service::connections(&pool).await }

#[tauri::command]
pub async fn save_ai_connection(config: AiConnectionConfig, pool: State<'_, DbPool>, state: State<'_, Arc<AiState>>) -> Result<AiConnectionConfig, String> {
    let _running = state.running.lock().await;
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ai_prediction_jobs WHERE connection_id=? AND status IN ('running','preparing')").bind(&config.id).fetch_one(&*pool).await.map_err(|e| e.to_string())?;
    if count > 0 { return Err("请先结束此连接的预测任务再修改配置".into()); }
    service::save_connection(&pool, config).await
}

#[tauri::command]
pub async fn delete_ai_connection(id: String, pool: State<'_, DbPool>, state: State<'_, Arc<AiState>>) -> Result<(), String> {
    let _running = state.running.lock().await;
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ai_prediction_jobs WHERE connection_id=? AND status IN ('running','preparing')").bind(&id).fetch_one(&*pool).await.map_err(|e| e.to_string())?;
    if count > 0 { return Err("请先结束此连接的预测任务".into()); }
    sqlx::query("DELETE FROM ai_connections WHERE id=?").bind(id).execute(&*pool).await.map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub async fn check_ai_connection(config: AiConnectionConfig) -> Result<String, String> { process::check_connection(&config).await }

#[tauri::command]
pub async fn start_ai_prediction(request: AiPredictionRequest, pool: State<'_, DbPool>, state: State<'_, Arc<AiState>>, app: AppHandle) -> Result<AiJob, String> { service::start(pool.inner().clone(), state.inner().clone(), Some(app), request).await }

#[tauri::command]
pub async fn cancel_ai_prediction(id: String, state: State<'_, Arc<AiState>>) -> Result<(), String> { service::cancel(&state, &id).await }

#[tauri::command]
pub async fn get_ai_prediction(id: String, pool: State<'_, DbPool>) -> Result<AiJob, String> { service::get_job(&pool, &id).await }

#[tauri::command]
pub async fn list_ai_predictions(symbol: Option<String>, pool: State<'_, DbPool>) -> Result<Vec<AiJob>, String> {
    let symbol = symbol.filter(|s| !s.trim().is_empty()).map(|s| crate::services::market_data::canonical_symbol(&s).map_err(|e| e.to_string())).transpose()?;
    service::list_jobs(&pool, symbol.as_deref()).await
}
