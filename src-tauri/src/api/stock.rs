use crate::db::models::HistoricalData;
use crate::error::AppError;
use chrono::NaiveDate;
use serde_json::Value;
use std::env;

// 查看全部股票名称以及代码
const LIST_ALL_API: &str = "https://api.zhituapi.com/hs/list/all?token=ZHITU_TOKEN_LIMIT_TEST";
// 查看股票历史
const HISTORY_API: &str = "https://api.zhituapi.com/hs/history/fsjy";

pub async fn fetch_historical_data(symbol: &str) -> Result<Vec<HistoricalData>, AppError> {
    let token = env::var("DDE0C310-D0DE-4767-9712-A424BAC4326D").unwrap();

    let response = reqwest::get(&format!("{}/{}/d?token={}", HISTORY_API, symbol, token))
        .await?
        .json::<Value>()
        .await?;

    parse_historical_data(response, symbol)
}

fn parse_historical_data(data: Value, symbol: &str) -> Result<Vec<HistoricalData>, AppError> {
    let time_series = data["Time Series (Daily)"]
        .as_object()
        .ok_or(AppError::ParseError("Invalid API response".into()))?;

    let mut result = Vec::new();

    for (date_str, values) in time_series {
        let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")?;

        result.push(HistoricalData {
            symbol: symbol.to_string(),
            date,
            open: values["1. open"].as_str().unwrap().parse()?,
            high: values["2. high"].as_str().unwrap().parse()?,
            low: values["3. low"].as_str().unwrap().parse()?,
            close: values["4. close"].as_str().unwrap().parse()?,
            volume: values["5. volume"].as_str().unwrap().parse()?,
        });
    }

    Ok(result)
}
