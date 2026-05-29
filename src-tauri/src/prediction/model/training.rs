//! 模型训练模块
//!
//! 使用 candle 训练真实的 MLP 模型：特征显式包含量比与换手率，
//! 标签为模型预测周期收益率，按时间切分训练/测试集，返回测试集上的真实方向准确率。

use crate::db::{
    connection::{create_temp_pool, DbPool},
    models::HistoricalData,
    repository::{get_historical_data, get_recent_historical_data},
};
use crate::prediction::model::features::{build_dataset_for_horizon, build_samples, feature_names};
use crate::prediction::model::management::{
    generate_model_id, get_current_timestamp, get_model_file_path, save_model_metadata,
};
use crate::prediction::model::network::train_and_save_with_gap;
use crate::prediction::model::HORIZON_AWARE_MODEL_TYPE;
use crate::prediction::types::{ModelInfo, TrainingRequest, TrainingResult};
use chrono::NaiveDate;

const DEFAULT_TRAINING_BARS: usize = 800;
const LEGACY_CANDLE_MLP_MODEL_TYPE: &str = "candle_mlp";

/// 训练股票预测模型（真实 candle MLP）
pub async fn train_model(request: TrainingRequest) -> Result<TrainingResult, String> {
    println!("🚀 开始训练模型: {}", request.model_name);
    println!("   股票代码: {}", request.stock_code);
    validate_training_model_type(&request.model_type)?;

    // 加载历史数据
    let pool = create_temp_pool().await?;
    let historical = load_training_history(&request, &pool).await?;

    if historical.len() < 80 {
        return Err(format!(
            "历史数据不足（{}），训练至少需要 80 个交易日",
            historical.len()
        ));
    }

    // 构造数据集
    let prediction_days = request.prediction_days.max(1);
    let (features, labels, n) = build_dataset_for_horizon(&historical, prediction_days);
    if n < 40 {
        return Err(format!("有效样本不足（{n}），无法训练"));
    }

    // 训练并保存权重
    let model_id = generate_model_id();
    let model_path = get_model_file_path(&model_id);
    let split = if request.train_test_split > 0.0 {
        request.train_test_split
    } else {
        0.8
    };
    let outcome = train_and_save_with_gap(
        &features,
        &labels,
        n,
        request.epochs.max(50),
        request.learning_rate,
        split,
        prediction_days,
        &model_path,
    )?;
    let (training_start_date, training_end_date) =
        training_sample_date_range(&historical, prediction_days, outcome.train_samples);

    let metadata = ModelInfo {
        id: model_id.clone(),
        name: request.model_name,
        stock_code: request.stock_code,
        created_at: get_current_timestamp(),
        model_type: HORIZON_AWARE_MODEL_TYPE.to_string(),
        features: feature_names(),
        target: request.target,
        prediction_days,
        accuracy: outcome.direction_accuracy,
        training_start_date,
        training_end_date,
        training_samples: Some(outcome.train_samples),
        test_samples: Some(outcome.test_samples),
        mae: Some(outcome.mae),
        rmse: Some(outcome.rmse),
    };
    save_model_metadata(&metadata)?;

    println!(
        "✅ 训练完成：方向准确率 {:.1}%（测试样本 {}，MAE {:.3}）",
        outcome.direction_accuracy * 100.0,
        outcome.test_samples,
        outcome.mae
    );

    Ok(TrainingResult {
        metadata,
        accuracy: outcome.direction_accuracy,
        test_samples: outcome.test_samples,
        mae: outcome.mae,
        rmse: outcome.rmse,
    })
}

/// 重新训练模型：按新的超参数对同一标的重新训练并覆盖权重
pub async fn retrain_model(
    model_id: String,
    epochs: u32,
    _batch_size: u32,
    learning_rate: f64,
) -> Result<(), String> {
    use crate::prediction::model::management::load_model_metadata;

    let metadata = load_model_metadata(&model_id)?;

    let pool = create_temp_pool().await?;
    let historical = get_recent_historical_data(&metadata.stock_code, DEFAULT_TRAINING_BARS, &pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))?;

    let training_horizon = if metadata.model_type == HORIZON_AWARE_MODEL_TYPE {
        metadata.prediction_days.max(1)
    } else {
        1
    };
    let (features, labels, n) = build_dataset_for_horizon(&historical, training_horizon);
    if n < 40 {
        return Err(format!("有效样本不足（{n}），无法重训练"));
    }

    let model_path = get_model_file_path(&model_id);
    let outcome = train_and_save_with_gap(
        &features,
        &labels,
        n,
        (epochs as usize).max(50),
        learning_rate,
        0.8,
        training_horizon,
        &model_path,
    )?;
    let (training_start_date, training_end_date) =
        training_sample_date_range(&historical, training_horizon, outcome.train_samples);

    // 更新元数据中的准确率
    let mut updated = metadata;
    updated.accuracy = outcome.direction_accuracy;
    updated.training_start_date = training_start_date;
    updated.training_end_date = training_end_date;
    updated.training_samples = Some(outcome.train_samples);
    updated.test_samples = Some(outcome.test_samples);
    updated.mae = Some(outcome.mae);
    updated.rmse = Some(outcome.rmse);
    save_model_metadata(&updated)?;

    println!(
        "🔄 重训练完成：方向准确率 {:.1}%",
        outcome.direction_accuracy * 100.0
    );
    Ok(())
}

fn training_sample_date_range(
    historical: &[HistoricalData],
    horizon: usize,
    train_samples: usize,
) -> (Option<String>, Option<String>) {
    let sample_dates: Vec<_> = build_samples(historical, horizon)
        .into_iter()
        .map(|sample| sample.date)
        .collect();
    let start_date = sample_dates
        .first()
        .map(|date| date.format("%Y-%m-%d").to_string());
    let end_date = train_samples
        .checked_sub(1)
        .and_then(|idx| sample_dates.get(idx))
        .map(|date| date.format("%Y-%m-%d").to_string());

    (start_date, end_date)
}

async fn load_training_history(
    request: &TrainingRequest,
    pool: &DbPool,
) -> Result<Vec<HistoricalData>, String> {
    if let Some((start_date, end_date)) = requested_training_date_range(request)? {
        return get_historical_data(&request.stock_code, start_date, end_date, pool)
            .await
            .map_err(|e| format!("获取历史数据失败: {e}"));
    }

    get_recent_historical_data(&request.stock_code, DEFAULT_TRAINING_BARS, pool)
        .await
        .map_err(|e| format!("获取历史数据失败: {e}"))
}

fn requested_training_date_range(request: &TrainingRequest) -> Result<Option<(&str, &str)>, String> {
    let start_date = request.start_date.trim();
    let end_date = request.end_date.trim();

    match (start_date.is_empty(), end_date.is_empty()) {
        (true, true) => Ok(None),
        (true, false) | (false, true) => Err("训练开始日期和结束日期必须同时提供".to_string()),
        (false, false) => {
            let start = NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
                .map_err(|e| format!("训练开始日期格式错误: {e}"))?;
            let end = NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
                .map_err(|e| format!("训练结束日期格式错误: {e}"))?;
            if end < start {
                return Err("训练结束日期不能早于开始日期".to_string());
            }
            Ok(Some((start_date, end_date)))
        }
    }
}

fn validate_training_model_type(model_type: &str) -> Result<(), String> {
    let model_type = model_type.trim();
    if model_type.is_empty()
        || model_type == HORIZON_AWARE_MODEL_TYPE
        || model_type == LEGACY_CANDLE_MLP_MODEL_TYPE
    {
        return Ok(());
    }

    Err(format!(
        "不支持的模型类型 `{model_type}`，当前仅支持 {HORIZON_AWARE_MODEL_TYPE}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_with_dates(start_date: &str, end_date: &str) -> TrainingRequest {
        TrainingRequest {
            stock_code: "600000".to_string(),
            model_name: "test".to_string(),
            start_date: start_date.to_string(),
            end_date: end_date.to_string(),
            features: Vec::new(),
            target: "close".to_string(),
            prediction_days: 5,
            model_type: HORIZON_AWARE_MODEL_TYPE.to_string(),
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            dropout: 0.2,
            train_test_split: 0.8,
        }
    }

    fn make_history(n: usize) -> Vec<HistoricalData> {
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        (0..n)
            .map(|i| HistoricalData {
                symbol: "600000".to_string(),
                date: start + chrono::Duration::days(i as i64),
                open: 10.0 + i as f64,
                close: 10.0 + i as f64,
                high: 10.5 + i as f64,
                low: 9.5 + i as f64,
                volume: 1000 + i as i64,
                amount: 10000.0,
                amplitude: 1.0,
                turnover_rate: 1.0,
                volume_ratio: 1.0,
                change_percent: 1.0,
                change: 0.1,
            })
            .collect()
    }

    #[test]
    fn test_training_sample_date_range_uses_actual_train_samples() {
        let history = make_history(60);

        let (start, end) = training_sample_date_range(&history, 5, 12);

        assert_eq!(start.as_deref(), Some("2025-01-21"));
        assert_eq!(end.as_deref(), Some("2025-02-01"));
    }

    #[test]
    fn test_requested_training_date_range_uses_request_dates() {
        let request = request_with_dates("2025-01-01", "2025-12-31");

        assert_eq!(
            requested_training_date_range(&request).unwrap(),
            Some(("2025-01-01", "2025-12-31"))
        );
    }

    #[test]
    fn test_requested_training_date_range_allows_empty_dates_for_legacy_callers() {
        let request = request_with_dates("", "");

        assert_eq!(requested_training_date_range(&request).unwrap(), None);
    }

    #[test]
    fn test_requested_training_date_range_rejects_partial_or_reversed_dates() {
        let partial = request_with_dates("2025-01-01", "");
        let reversed = request_with_dates("2025-12-31", "2025-01-01");

        assert!(requested_training_date_range(&partial).is_err());
        assert!(requested_training_date_range(&reversed).is_err());
    }

    #[test]
    fn test_validate_training_model_type_accepts_supported_and_legacy_values() {
        assert!(validate_training_model_type("").is_ok());
        assert!(validate_training_model_type(HORIZON_AWARE_MODEL_TYPE).is_ok());
        assert!(validate_training_model_type(LEGACY_CANDLE_MLP_MODEL_TYPE).is_ok());
    }

    #[test]
    fn test_validate_training_model_type_rejects_unknown_values() {
        assert!(validate_training_model_type("candle_lstm").is_err());
    }
}
