use crate::db::models::Stock;
use std::error::Error;
use std::fs::File;

pub fn read_csv_to_struct(file_path: &str) -> Result<Vec<Stock>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut stocks = Vec::new();

    for result in rdr.deserialize() {
        let stock: Stock = result?;
        stocks.push(stock);
    }

    Ok(stocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_read_csv() {
        let result = read_csv_to_struct("data/stock_basic.csv");
        match result {
            Ok(stocks) => {
                println!("{:?}", stocks.get(0).unwrap_or(&Stock::default()));
            }
            Err(err) => {
                panic!("Failed to read CSV: {}", err);
            }
        }
    }
}
