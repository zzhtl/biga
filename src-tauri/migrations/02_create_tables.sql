CREATE TABLE IF NOT EXISTS realtime_data (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    date DATE NOT NULL,
    ytd_close REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    change REAL NOT NULL,
    change_percent REAL NOT NULL
);
