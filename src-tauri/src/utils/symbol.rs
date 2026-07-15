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
    fn preserves_non_stock_identifiers() {
        assert_eq!(canonical_stock_symbol(" abc "), "abc");
        assert_eq!(canonical_stock_symbol("12345"), "12345");
        assert_eq!(canonical_stock_symbol("1234567"), "1234567");
    }
}
