#[derive(thiserror::Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    SqlxError(#[from] sqlx::Error),

    #[error("API 请求失败，请检查网络连接或代理设置")]
    ApiError(#[from] reqwest::Error),

    #[error("股票数据服务响应超时，API 密钥已保存，请稍后重试")]
    ApiTimeout,

    #[error("无法连接股票数据服务，请检查网络或代理设置")]
    ApiConnection,

    #[error("Invalid date format: {0}")]
    DateError(#[from] chrono::ParseError),

    #[error("Parse float error: {0}")]
    ParseFloatError(std::num::ParseFloatError),

    #[error("Data parsing error: {0}")]
    ParseIntError(std::num::ParseIntError),

    #[error("数据库迁移失败: {0}")]
    MigrationError(#[from] sqlx::migrate::MigrateError),
    
    #[error("无效的输入参数: {0}")]
    InvalidInput(String),

    #[error("模型反序列化错误: {0}")]
    DeserializationError(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("安全配置错误: {0}")]
    SecretStoreError(String),

    #[error("尚未配置股票数据 API 密钥，请先前往系统设置")]
    MissingApiToken,
}

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}

impl From<std::num::ParseFloatError> for AppError {
    fn from(err: std::num::ParseFloatError) -> Self {
        AppError::ParseFloatError(err)
    }
}

impl From<std::num::ParseIntError> for AppError {
    fn from(err: std::num::ParseIntError) -> Self {
        AppError::ParseIntError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn api_errors_do_not_expose_request_urls() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("a local test port should be available");
        let address = listener
            .local_addr()
            .expect("the local test address should be readable");
        drop(listener);

        let secret = "sensitive-test-token";
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("the local test client should build");
        let error = client
            .get(format!("http://{address}/?token={secret}"))
            .send()
            .await
            .expect_err("the closed local port should reject the request");
        let serialized = serde_json::to_string(&AppError::from(error))
            .expect("the application error should serialize");

        assert_eq!(serialized, "\"API 请求失败，请检查网络连接或代理设置\"");
        assert!(!serialized.contains(secret));
    }
}
