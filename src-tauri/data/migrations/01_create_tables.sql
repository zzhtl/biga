CREATE TABLE IF NOT EXISTS stock_info (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stock (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    area TEXT DEFAULT NULL,
    industry TEXT DEFAULT NULL,
    market TEXT DEFAULT NULL,
    exchange TEXT NOT NULL,
    list_date TEXT DEFAULT NULL,
    act_name TEXT DEFAULT NULL,
    act_ent_type TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS historical_data (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    volume INTEGER NOT NULL,
    amount REAL NOT NULL,
    amplitude REAL NOT NULL,
    turnover_rate REAL NOT NULL,
    change_percent REAL NOT NULL,
    change REAL NOT NULL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS realtime_data (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    date DATE NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    amount REAL NOT NULL,
    amplitude REAL NOT NULL,
    turnover_rate REAL NOT NULL,
    change_percent REAL NOT NULL,
    change REAL NOT NULL
);
