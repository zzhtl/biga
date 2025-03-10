CREATE TABLE IF NOT EXISTS historical_data (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, date)
);
