use crate::db::models::{HistoricalData, HistoricalDataItem, StockInfo, StockInfoItem};
use crate::error::AppError;
use chrono::NaiveDate;

// 查看全部股票名称以及代码 - 使用更稳定的API端点
const ALL_SYMBOL_API: &str = "https://api.zhituapi.com/hs/list/all?token=ZHITU_TOKEN_LIMIT_TEST";
// 查看股票历史
const HISTORY_API: &str = "https://api.zhituapi.com/hs/history";

pub async fn fetch_stock_infos() -> Result<Vec<StockInfo>, AppError> {
    println!("开始获取股票信息...");
    
    let response = reqwest::Client::new()
        .get(ALL_SYMBOL_API)
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
    
    // 使用更稳定的token，或者从环境变量读取
    let token = std::env::var("STOCK_API_TOKEN")
        .unwrap_or_else(|_| "5E1C3717-F301-4761-B8F9-7ED7FF186381".to_string());
    
    let url = format!("{HISTORY_API}/{symbol}/d/n?token={token}");
    println!("请求URL: {url}");
    
    let response = reqwest::Client::new()
        .get(&url)
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
            println!("响应内容: {}", &response_text[..std::cmp::min(500, response_text.len())]);
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
            
            // 换手率暂时设为0，需要股本数据才能准确计算
            let turnover_rate = 0.0;
                
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
                turnover_rate,
                change_percent,
                change,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
