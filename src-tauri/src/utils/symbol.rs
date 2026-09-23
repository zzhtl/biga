/// 将带交易所前后缀的 A 股代码统一为纯 6 位数字。
///
/// 无法提取出恰好 6 位数字时仅去除首尾空白，避免误改非标准标识。
pub fn canonical_stock_symbol(symbol: &str) -> String {
    let trimmed = symbol.trim();
    let digits: String = trimmed
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();

    if digits.len() == 6 {
        digits
    } else {
        trimmed.to_string()
    }
}

/// A 股涨跌停限幅（百分比，返回 `(跌停, 涨停)`）。
///
/// 与 `professional_engine::get_stock_price_limits` 的区别：那个函数为预测留了裕度
/// （主板返回 ±9.5）；这里是**真实**限幅，用于判断「一字板是否可成交」，以及给区间带
/// 做物理上界截断。放在 utils 是因为 `prediction` 和 `discipline` 都要用，而依赖方向是
/// `discipline → prediction`，反向引用会把依赖关系拧成环。
pub fn price_limit_percent(symbol: &str, name: &str) -> (f64, f64) {
    let code = symbol.trim_start_matches(|c: char| !c.is_ascii_digit());
    // 创业板 / 科创板即使被 ST 也是 ±20%，所以先判板块
    if code.starts_with("688") || code.starts_with("300") || code.starts_with("301") {
        return (-20.0, 20.0);
    }
    if name.to_uppercase().contains("ST") {
        return (-5.0, 5.0);
    }
    // 北交所 ±30%。库内目前无此类标的，保留分支避免误判为主板（known gap）
    if code.starts_with('4') || code.starts_with('8') {
        return (-30.0, 30.0);
    }
    (-10.0, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_supported_stock_symbol_variants() {
        assert_eq!(canonical_stock_symbol("002466"), "002466");
        assert_eq!(canonical_stock_symbol("002466.SZ"), "002466");
        assert_eq!(canonical_stock_symbol("sz002466"), "002466");
        assert_eq!(canonical_stock_symbol(" 600519.SH "), "600519");
    }

    #[test]
    fn star_and_st_boards_get_their_real_limits() {
        assert_eq!(price_limit_percent("688981", "中芯国际"), (-20.0, 20.0));
        assert_eq!(price_limit_percent("300750", "宁德时代"), (-20.0, 20.0));
        assert_eq!(price_limit_percent("600519", "贵州茅台"), (-10.0, 10.0));
        assert_eq!(price_limit_percent("600666", "*ST奥瑞"), (-5.0, 5.0));
        assert_eq!(
            price_limit_percent("300123", "ST太空"),
            (-20.0, 20.0),
            "创业板即使 ST 也是 ±20%"
        );
    }
    #[test]
    fn preserves_non_stock_identifiers() {
        assert_eq!(canonical_stock_symbol(" abc "), "abc");
        assert_eq!(canonical_stock_symbol("12345"), "12345");
        assert_eq!(canonical_stock_symbol("1234567"), "1234567");
    }
}
