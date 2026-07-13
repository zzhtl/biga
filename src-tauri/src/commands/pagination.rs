use serde::Serialize;

pub(crate) const DEFAULT_PAGE_SIZE: u32 = 50;
pub(crate) const MAX_PAGE_SIZE: u32 = 100;

#[derive(Debug, Serialize)]
pub(crate) struct PagedResponse<T> {
    pub data: Vec<T>,
    pub total: i64,
    pub page: u32,
    pub page_size: u32,
}

pub(crate) fn normalize_page(page: u32, page_size: u32) -> (u32, u32, i64) {
    let page = page.max(1);
    let page_size = if page_size == 0 {
        DEFAULT_PAGE_SIZE
    } else {
        page_size.min(MAX_PAGE_SIZE)
    };
    let offset = i64::from((page - 1).saturating_mul(page_size));
    (page, page_size, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_page_bounds() {
        assert_eq!(normalize_page(0, 0), (1, 50, 0));
        assert_eq!(normalize_page(3, 500), (3, 100, 200));
    }
}
