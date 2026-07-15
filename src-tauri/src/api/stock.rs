use crate::db::models::{
    HistoricalData, HistoricalDataItem, RealtimeQuoteItem, StockFundamental, StockInfo,
    StockInfoItem,
};
use crate::error::AppError;
use crate::config::api_token::resolve_api_token;
use crate::utils::canonical_stock_symbol;
use chrono::NaiveDate;

// 查看全部股票名称以及代码
const ALL_SYMBOL_API: &str = "https://api.zhituapi.com/hs/list/all";
// 查看股票历史
const HISTORY_API: &str = "https://api.zhituapi.com/hs/history";
// 实时交易（含流通市值 lt、总市值 sz、换手率 hs、量比 lb）
const REALTIME_API: &str = "https://api.zhituapi.com/hs/real/ssjy";
// 财务指标（含 ROE、每股收益、每股净资产、增长率等基本面数据）
const FINANCIAL_API: &str = "https://api.zhituapi.com/hs/gs/cwzb";
const TOKEN_VALIDATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// 将各种格式的股票代码归一化为 zhitu 实时接口所需的纯 6 位数字代码。
/// 例如 "000002.SZ" / "sz000002" → "000002"。
fn normalize_quote_symbol(symbol: &str) -> String {
    canonical_stock_symbol(symbol)
}

pub async fn fetch_stock_infos() -> Result<Vec<StockInfo>, AppError> {
    println!("开始获取股票信息...");
    let (token, _) = resolve_api_token().await?;

    let response = reqwest::Client::new()
        .get(ALL_SYMBOL_API)
        .query(&[("token", token)])
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;
    
    if !response.status().is_success() {
        println!("API请求失败: {}", response.status());
        return Err(AppError::InvalidInput(format!("API请求失败: {}", response.status())));
    }
    
    let stock_infos: Vec<StockInfoItem> = response.json().await?;
    println!("获取到 {} 条股票信息", stock_infos.len());
    
    parse_stock_info(stock_infos)
}

fn parse_stock_info(items: Vec<StockInfoItem>) -> Result<Vec<StockInfo>, AppError> {
    items
        .into_iter()
        .map(|item| {
            Ok(StockInfo {
                symbol: item.symbol,
                name: item.name,
                exchange: item.exchange,
            })
        })
        .collect()
}

pub async fn fetch_historical_data(symbol: &str) -> Result<Vec<HistoricalData>, AppError> {
    println!("开始获取股票 {symbol} 的历史数据...");

    let (token, _) = resolve_api_token().await?;
    let url = format!("{HISTORY_API}/{symbol}/d/n");

    let response = reqwest::Client::new()
        .get(&url)
        .query(&[("token", token)])
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;
    
    if !response.status().is_success() {
        println!("API请求失败: {}", response.status());
        return Err(AppError::InvalidInput(format!("获取历史数据失败: {}", response.status())));
    }
    
    let response_text = response.text().await?;
    println!("API响应长度: {}", response_text.len());
    
    // 尝试解析JSON
    let historical_items: Vec<HistoricalDataItem> = serde_json::from_str(&response_text)
        .map_err(|e| {
            println!("JSON解析失败: {e}");
            AppError::DeserializationError(format!("JSON解析失败: {e}"))
        })?;
    
    println!("解析到 {} 条历史数据", historical_items.len());
    
    parse_historical_data(historical_items, symbol)
}

fn parse_historical_data(
    items: Vec<HistoricalDataItem>,
    symbol: &str,
) -> Result<Vec<HistoricalData>, AppError> {
    items
        .into_iter()
        .map(|item| {
            // 解析日期 - 支持两种格式
            let date = if item.date.contains(" ") {
                // 格式：2002-10-24 00:00:00
                NaiveDate::parse_from_str(item.date.split(' ').next().unwrap_or(&item.date), "%Y-%m-%d")
            } else {
                // 格式：2002-10-24
                NaiveDate::parse_from_str(&item.date, "%Y-%m-%d")
            }
            .map_err(|e| AppError::InvalidInput(format!("日期解析失败: {e}")))?;
            
            // 计算涨跌额
            let change = item.close - item.pre_close;
            
            // 计算涨跌幅（%），保留2位小数
            let change_percent = if item.pre_close != 0.0 {
                ((change / item.pre_close) * 100.0 * 100.0).round() / 100.0
            } else {
                0.0
            };
            
            // 计算振幅（%），保留2位小数
            let amplitude = if item.pre_close != 0.0 {
                (((item.high - item.low) / item.pre_close) * 100.0 * 100.0).round() / 100.0
            } else {
                0.0
            };
            
            // turnover_rate / volume_ratio 由 backfill_volume_metrics 回填
            Ok(HistoricalData {
                symbol: symbol.to_string(),
                date,
                open: item.open,
                high: item.high,
                low: item.low,
                close: item.close,
                volume: item.volume as i64,
                amount: item.amount,
                amplitude,
                turnover_rate: 0.0,
                volume_ratio: 0.0,
                change_percent,
                change,
            })
        })
        .collect()
}

/// 获取股票实时行情中的股本相关数据（流通市值、总市值、换手率、量比）
///
/// 用于推导流通股本以计算历史换手率。网络或解析失败时返回 Err，调用方应优雅降级。
pub async fn fetch_stock_capital(symbol: &str) -> Result<RealtimeQuoteItem, AppError> {
    let (token, _) = resolve_api_token().await?;
    // ssjy 接口只接受纯 6 位数字代码（如 000002），需从 000002.SZ / sz000002 归一化
    let code = normalize_quote_symbol(symbol);
    let url = format!("{REALTIME_API}/{code}");

    let response = reqwest::Client::new()
        .get(&url)
        .query(&[("token", token)])
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(AppError::InvalidInput(format!(
            "获取股本数据失败: {}",
            response.status()
        )));
    }

    let text = response.text().await?;
    let quote: RealtimeQuoteItem = serde_json::from_str(&text)
        .map_err(|e| AppError::DeserializationError(format!("股本数据解析失败: {e}")))?;

    Ok(quote)
}

/// 解析 cwzb 字符串数值："--" / 空 → None；可能含千分位逗号。
fn parse_cw_number(s: &str) -> Option<f64> {
    let t = s.trim().replace(',', "");
    if t.is_empty() || t == "--" {
        return None;
    }
    t.parse::<f64>().ok()
}

/// 拉取财务指标（cwzb），返回各报告期的关键基本面字段（非技术数据）。
///
/// 接口返回按报告期倒序的数组，字段为字符串（缺失为 "--"）。网络或解析失败返回 Err，
/// 调用方应优雅降级。
pub async fn fetch_financial_indicators(symbol: &str) -> Result<Vec<StockFundamental>, AppError> {
    let (token, _) = resolve_api_token().await?;
    let code = normalize_quote_symbol(symbol);
    let url = format!("{FINANCIAL_API}/{code}");

    let response = reqwest::Client::new()
        .get(&url)
        .query(&[("token", token)])
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(AppError::InvalidInput(format!(
            "获取财务指标失败: {}",
            response.status()
        )));
    }

    let text = response.text().await?;
    let rows: Vec<serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| AppError::DeserializationError(format!("财务指标解析失败: {e}")))?;

    let field = |v: &serde_json::Value, key: &str| -> Option<f64> {
        v.get(key).and_then(|x| x.as_str()).and_then(parse_cw_number)
    };

    let out = rows
        .iter()
        .filter_map(|v| {
            let report_date = v.get("date").and_then(|x| x.as_str())?.to_string();
            Some(StockFundamental {
                symbol: code.clone(),
                report_date,
                eps: field(v, "mgsy"),
                bps: field(v, "mgjz"),
                roe: field(v, "jzsy"),
                profit_growth: field(v, "jlzz"),
                revenue_growth: field(v, "zysr"),
                debt_ratio: field(v, "zcfzl"),
            })
        })
        .collect();

    Ok(out)
}

fn map_validation_request_error(error: reqwest::Error) -> AppError {
    if error.is_timeout() {
        AppError::ApiTimeout
    } else if error.is_connect() {
        AppError::ApiConnection
    } else {
        AppError::ApiError(error)
    }
}

async fn send_validation_request(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    timeout: std::time::Duration,
) -> Result<reqwest::Response, AppError> {
    client
        .get(url)
        .query(&[("token", token)])
        .timeout(timeout)
        .send()
        .await
        .map_err(map_validation_request_error)
}

async fn send_validation_request_with_fallback(
    client: &reqwest::Client,
    token: &str,
    endpoints: &[(&str, std::time::Duration)],
) -> Result<reqwest::Response, AppError> {
    let Some((last, primary)) = endpoints.split_last() else {
        return Err(AppError::ApiConnection);
    };

    for (url, timeout) in primary {
        match send_validation_request(client, url, token, *timeout).await {
            Ok(response) => return Ok(response),
            Err(AppError::ApiTimeout) | Err(AppError::ApiConnection) => {}
            Err(error) => return Err(error),
        }
    }

    send_validation_request(client, last.0, token, last.1).await
}

pub async fn validate_api_token(token: &str) -> Result<(), AppError> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;
    let primary_url = format!("{FINANCIAL_API}/000001");
    let fallback_url = format!("{REALTIME_API}/000001");
    let response = send_validation_request_with_fallback(
        &client,
        token,
        &[
            (&primary_url, TOKEN_VALIDATION_TIMEOUT),
            (&fallback_url, TOKEN_VALIDATION_TIMEOUT),
        ],
    )
    .await?;

    if !response.status().is_success() {
        return Err(AppError::InvalidInput(format!(
            "API 密钥验证失败: {}",
            response.status()
        )));
    }

    response
        .json::<serde_json::Value>()
        .await
        .map_err(|_| AppError::InvalidInput("API 密钥无效或服务暂不可用".to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn test_normalize_quote_symbol() {
        // ssjy 接口要求纯 6 位数字
        assert_eq!(normalize_quote_symbol("000002.SZ"), "000002");
        assert_eq!(normalize_quote_symbol("sz000001"), "000001");
        assert_eq!(normalize_quote_symbol("600519.SH"), "600519");
        assert_eq!(normalize_quote_symbol("600519"), "600519");
        assert_eq!(normalize_quote_symbol("000001"), "000001");
    }

    #[tokio::test]
    async fn validation_uses_a_fallback_after_a_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a local test port should be available");
        let address = listener
            .local_addr()
            .expect("the local test address should be readable");
        let server = tokio::spawn(async move {
            let (first, _) = listener
                .accept()
                .await
                .expect("the first request should connect");
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                drop(first);
            });

            let (mut second, _) = listener
                .accept()
                .await
                .expect("the retry should connect");
            let mut request = [0_u8; 1024];
            let bytes_read = second
                .read(&mut request)
                .await
                .expect("the retry request should be readable");
            assert!(bytes_read > 0, "the retry request should not be empty");
            second
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 8\r\nConnection: close\r\n\r\n{\"lt\":1}",
                )
                .await
                .expect("the retry response should be sent");
        });

        let url = format!("http://{address}");
        let response = send_validation_request_with_fallback(
            &reqwest::Client::new(),
            "test-token",
            &[
                (&url, std::time::Duration::from_millis(20)),
                (&url, std::time::Duration::from_millis(20)),
            ],
        )
        .await
        .expect("the retry should succeed");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        server.await.expect("the test server should stop cleanly");
    }

    #[tokio::test]
    async fn validation_returns_a_safe_timeout_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a local test port should be available");
        let address = listener
            .local_addr()
            .expect("the local test address should be readable");
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (connection, _) = listener
                    .accept()
                    .await
                    .expect("each validation attempt should connect");
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    drop(connection);
                });
            }
        });

        let url = format!("http://{address}");
        let error = send_validation_request_with_fallback(
            &reqwest::Client::new(),
            "sensitive-test-token",
            &[
                (&url, std::time::Duration::from_millis(20)),
                (&url, std::time::Duration::from_millis(20)),
            ],
        )
        .await
        .expect_err("both attempts should time out");
        let message = error.to_string();

        assert!(matches!(error, AppError::ApiTimeout));
        assert_eq!(message, "股票数据服务响应超时，API 密钥已保存，请稍后重试");
        assert!(!message.contains("sensitive-test-token"));
        server.await.expect("the test server should stop cleanly");
    }

    #[test]
    fn test_parse_cw_number() {
        assert_eq!(parse_cw_number("--"), None);
        assert_eq!(parse_cw_number(""), None);
        assert_eq!(parse_cw_number("  "), None);
        assert_eq!(parse_cw_number("28.037"), Some(28.037));
        assert_eq!(parse_cw_number("2.67"), Some(2.67));
        // 千分位逗号（jdlr 等接口常见）
        assert_eq!(parse_cw_number("3,527,700.00"), Some(3_527_700.0));
        assert_eq!(parse_cw_number("-2.2817"), Some(-2.2817));
    }

    #[test]
    fn test_parse_historical_data_item() {
        let json_data = r#"[{"t":"2002-10-24 00:00:00","o":12.05,"h":12.46,"l":11.88,"c":12.09,"v":362898.0,"a":440587759.0,"pc":5.4,"sf":0}]"#;
        
        let historical_items: Result<Vec<HistoricalDataItem>, _> = serde_json::from_str(json_data);
        assert!(historical_items.is_ok(), "JSON parsing should succeed");
        
        let items = historical_items.unwrap();
        assert_eq!(items.len(), 1);
        
        let item = &items[0];
        assert_eq!(item.date, "2002-10-24 00:00:00");
        assert_eq!(item.open, 12.05);
        assert_eq!(item.high, 12.46);
        assert_eq!(item.low, 11.88);
        assert_eq!(item.close, 12.09);
        assert_eq!(item.volume, 362898.0);
        assert_eq!(item.amount, 440587759.0);
        assert_eq!(item.pre_close, 5.4);
        
        // 测试计算功能
        let result = parse_historical_data(items, "sz000001");
        assert!(result.is_ok(), "Parse should succeed");
        
        let historical_data = result.unwrap();
        let data = &historical_data[0];
        
        // 验证计算结果
        let expected_change = 12.09 - 5.4; // 6.69
        let expected_change_percent = ((12.09 - 5.4) / 5.4) * 100.0; // 约123.89%
        let expected_amplitude = ((12.46 - 11.88) / 5.4) * 100.0; // 约10.74%
        
        assert!((data.change - expected_change).abs() < 0.01);
        assert!((data.change_percent - expected_change_percent).abs() < 0.01);
        assert!((data.amplitude - expected_amplitude).abs() < 0.01);
    }
}
