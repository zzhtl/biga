use crate::error::AppError;
use keyring::{Entry, Error as KeyringError};
use serde::Serialize;

const KEYRING_SERVICE: &str = "com.biga.app";
const KEYRING_ACCOUNT: &str = "stock_api_token";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiTokenSource {
    Keyring,
    Environment,
    None,
}

#[derive(Debug, Serialize)]
pub struct ApiTokenStatus {
    pub configured: bool,
    pub source: ApiTokenSource,
    pub masked: Option<String>,
}

fn entry() -> Result<Entry, AppError> {
    Entry::new(KEYRING_SERVICE, KEYRING_ACCOUNT)
        .map_err(|error| AppError::SecretStoreError(error.to_string()))
}

fn read_keyring_token() -> Result<Option<String>, AppError> {
    match entry()?.get_password() {
        Ok(token) => Ok(normalize_token(&token)),
        Err(KeyringError::NoEntry) => Ok(None),
        Err(error) => Err(AppError::SecretStoreError(error.to_string())),
    }
}

fn save_and_verify_token(
    token: &str,
    save: impl FnOnce(&str) -> Result<(), AppError>,
    read: impl FnOnce() -> Result<Option<String>, AppError>,
) -> Result<(), AppError> {
    save(token)?;

    match read()? {
        Some(saved_token) if saved_token == token => Ok(()),
        _ => Err(AppError::SecretStoreError(
            "系统钥匙串不可用，API 密钥保存后无法读取".to_string(),
        )),
    }
}

fn write_keyring_token(token: &str) -> Result<(), AppError> {
    save_and_verify_token(
        token,
        |token| {
            entry()?
                .set_password(token)
                .map_err(|error| AppError::SecretStoreError(error.to_string()))
        },
        read_keyring_token,
    )
}

fn normalize_token(token: &str) -> Option<String> {
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

fn environment_token() -> Option<String> {
    std::env::var("STOCK_API_TOKEN")
        .ok()
        .and_then(|token| normalize_token(&token))
}

pub fn mask_token(token: &str) -> String {
    let suffix: String = token
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("****{suffix}")
}

pub async fn resolve_api_token() -> Result<(String, ApiTokenSource), AppError> {
    let keyring_result = tokio::task::spawn_blocking(read_keyring_token)
        .await
        .map_err(|error| AppError::SecretStoreError(error.to_string()))?;

    match keyring_result {
        Ok(Some(token)) => return Ok((token, ApiTokenSource::Keyring)),
        Ok(None) => {}
        Err(error) if environment_token().is_none() => return Err(error),
        Err(_) => {}
    }

    environment_token()
        .map(|token| (token, ApiTokenSource::Environment))
        .ok_or(AppError::MissingApiToken)
}

pub async fn api_token_status() -> Result<ApiTokenStatus, AppError> {
    match resolve_api_token().await {
        Ok((token, source)) => Ok(ApiTokenStatus {
            configured: true,
            source,
            masked: Some(mask_token(&token)),
        }),
        Err(AppError::MissingApiToken) => Ok(ApiTokenStatus {
            configured: false,
            source: ApiTokenSource::None,
            masked: None,
        }),
        Err(error) => Err(error),
    }
}

pub async fn save_api_token(token: String) -> Result<ApiTokenStatus, AppError> {
    let token = normalize_token(&token)
        .ok_or_else(|| AppError::InvalidInput("API 密钥不能为空".to_string()))?;
    let saved_token = token.clone();

    tokio::task::spawn_blocking(move || write_keyring_token(&saved_token))
        .await
        .map_err(|error| AppError::SecretStoreError(error.to_string()))??;

    Ok(ApiTokenStatus {
        configured: true,
        source: ApiTokenSource::Keyring,
        masked: Some(mask_token(&token)),
    })
}

pub async fn clear_api_token() -> Result<ApiTokenStatus, AppError> {
    tokio::task::spawn_blocking(|| match entry()?.delete_credential() {
        Ok(()) | Err(KeyringError::NoEntry) => Ok(()),
        Err(error) => Err(AppError::SecretStoreError(error.to_string())),
    })
    .await
    .map_err(|error| AppError::SecretStoreError(error.to_string()))??;

    api_token_status().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use keyring::credential::CredentialPersistence;
    use std::cell::RefCell;

    #[test]
    fn keyring_backend_persists_credentials() {
        assert!(matches!(
            keyring::default::default_credential_builder().persistence(),
            CredentialPersistence::UntilDelete
        ));
    }

    #[test]
    fn verifies_a_token_can_be_read_after_saving() {
        let stored = RefCell::new(None);

        let result = save_and_verify_token(
            "test-token",
            |token| {
                stored.replace(Some(token.to_string()));
                Ok(())
            },
            || Ok(stored.borrow().clone()),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn rejects_a_token_missing_after_saving() {
        let result = save_and_verify_token("test-token", |_| Ok(()), || Ok(None));

        assert!(matches!(result, Err(AppError::SecretStoreError(_))));
    }

    #[test]
    #[ignore = "requires an unlocked desktop keyring"]
    fn system_keyring_round_trip() {
        let account = format!("stock_api_token_test_{}", uuid::Uuid::new_v4());
        let test_entry = Entry::new(KEYRING_SERVICE, &account)
            .expect("the system keyring entry should be created");

        test_entry
            .set_password("biga-keyring-test")
            .expect("the test credential should be saved");
        let saved = test_entry
            .get_password()
            .expect("the test credential should be readable");
        let deleted = test_entry.delete_credential();

        assert_eq!(saved, "biga-keyring-test");
        deleted.expect("the test credential should be deleted");
    }

    #[test]
    fn masks_only_the_last_four_characters() {
        assert_eq!(mask_token("12345678"), "****5678");
        assert_eq!(mask_token("abc"), "****abc");
    }

    #[test]
    fn trims_and_rejects_empty_tokens() {
        assert_eq!(normalize_token("  token  "), Some("token".to_string()));
        assert_eq!(normalize_token("   "), None);
    }
}
