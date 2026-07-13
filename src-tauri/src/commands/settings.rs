use crate::api::stock;
use crate::config::api_token::{
    api_token_status, clear_api_token as clear_token, resolve_api_token,
    save_api_token as save_token, ApiTokenStatus,
};
use crate::error::AppError;

#[tauri::command]
pub async fn get_api_token_status() -> Result<ApiTokenStatus, AppError> {
    api_token_status().await
}

#[tauri::command]
pub async fn save_api_token(token: String) -> Result<ApiTokenStatus, AppError> {
    save_token(token).await
}

#[tauri::command]
pub async fn clear_api_token() -> Result<ApiTokenStatus, AppError> {
    clear_token().await
}

#[tauri::command]
pub async fn test_api_token() -> Result<bool, AppError> {
    let (token, _) = resolve_api_token().await?;
    stock::validate_api_token(&token).await?;
    Ok(true)
}
