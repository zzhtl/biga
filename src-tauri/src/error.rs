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
