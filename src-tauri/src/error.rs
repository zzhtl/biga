#[derive(thiserror::Error, Debug)]
pub enum AppError {
    #[error("Database error: {0}")]
    SqlxError(#[from] sqlx::Error),

    #[error("API request failed: {0}")]
    ApiError(#[from] reqwest::Error),

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

    #[error("模型训练错误: {0}")]
    ModelTrainingError(String),

    #[error("模型预测错误: {0}")]
    PredictionError(String),

    #[error("模型序列化错误: {0}")]
    SerializationError(String),

    #[error("模型反序列化错误: {0}")]
    DeserializationError(String),

    #[error("数据库操作错误: {0}")]
    DbError(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),
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
