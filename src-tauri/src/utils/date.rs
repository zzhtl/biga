//! 日期工具函数
//! 
//! 提供A股交易日判断、节假日处理等功能

use chrono::{Datelike, NaiveDate, Weekday};

/// 判断是否为交易日
pub fn is_trading_day(date: NaiveDate) -> bool {
    // 检查是否为工作日
    match date.weekday() {
        Weekday::Sat | Weekday::Sun => return false,
        _ => {}
    }

    let year = date.year();
    let month = date.month();
    let day = date.day();

    // 固定节假日
    match (month, day) {
        (1, 1) => return false,           // 元旦
        (4, 4..=6) => return false,       // 清明节
        (5, 1..=3) => return false,       // 劳动节
        (10, 1..=7) => return false,      // 国庆节
        _ => {}
    }

    // 春节假期（按年份判断）
    match year {
        2024 => {
            if month == 2 && (10..=17).contains(&day) {
                return false;
            }
        }
        2025 => {
            if (month == 1 && day >= 29) || (month == 2 && day <= 4) {
                return false;
            }
        }
        2023 => {
            if month == 1 && (21..=27).contains(&day) {
                return false;
            }
        }
        _ => {}
    }

    true
}

/// 获取下一个交易日
pub fn get_next_trading_day(date: NaiveDate) -> NaiveDate {
    let mut next_date = date + chrono::Duration::days(1);
    let mut count = 0;
    
    while !is_trading_day(next_date) && count < 30 {
        next_date += chrono::Duration::days(1);
        count += 1;
    }
    
    if count >= 30 {
        println!("⚠️ 警告：查找下一个交易日超过30天");
    }
    
    next_date
}

/// 获取N个交易日后的日期
pub fn get_trading_day_after(date: NaiveDate, days: usize) -> NaiveDate {
    let mut result = date;
    let mut count = 0;
    
    while count < days {
        result = get_next_trading_day(result);
        count += 1;
    }
    
    result
}

/// 计算两个日期之间的交易日数量
pub fn count_trading_days(start: NaiveDate, end: NaiveDate) -> usize {
    if start >= end {
        return 0;
    }
    
    let mut count = 0;
    let mut current = start;
    
    while current < end {
        if is_trading_day(current) {
            count += 1;
        }
        current += chrono::Duration::days(1);
    }
    
    count
}

/// 格式化日期为字符串
pub fn format_date(date: NaiveDate) -> String {
    date.format("%Y-%m-%d").to_string()
}

/// 解析日期字符串
pub fn parse_date(date_str: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_trading_day() {
        // 周末
        let saturday = NaiveDate::from_ymd_opt(2024, 3, 16).unwrap();
        assert!(!is_trading_day(saturday));
        
        // 工作日
        let monday = NaiveDate::from_ymd_opt(2024, 3, 18).unwrap();
        assert!(is_trading_day(monday));
    }

    #[test]
    fn test_get_next_trading_day() {
        let friday = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
        let next = get_next_trading_day(friday);
        assert_eq!(next.weekday(), Weekday::Mon);
    }
}

