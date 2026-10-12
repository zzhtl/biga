CREATE TABLE IF NOT EXISTS market_query_cache (
    resource TEXT NOT NULL,
    symbol TEXT NOT NULL,
    expires_at INTEGER NOT NULL DEFAULT 0,
    retry_at INTEGER NOT NULL DEFAULT 0,
    fetched_at INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    PRIMARY KEY (resource, symbol)
);

CREATE TABLE IF NOT EXISTS ai_connections (
    id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_prediction_jobs (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    connection_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    cache_key TEXT,
    report_json TEXT,
    error TEXT,
    requests INTEGER NOT NULL DEFAULT 0,
    cache_hits INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_symbol ON ai_prediction_jobs(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_cache ON ai_prediction_jobs(cache_key, status);
