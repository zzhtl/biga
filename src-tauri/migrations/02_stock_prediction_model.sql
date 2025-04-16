CREATE TABLE IF NOT EXISTS stock_prediction_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_data BLOB NOT NULL,
    parameters TEXT NOT NULL,
    metrics TEXT NOT NULL,
    UNIQUE(symbol, model_name)
);

CREATE TABLE IF NOT EXISTS stock_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_id INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    predicted_price REAL NOT NULL,
    predicted_change_percent REAL NOT NULL,
    confidence REAL NOT NULL,
    features_used TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES stock_prediction_models(id),
    UNIQUE(symbol, model_id, target_date)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON stock_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON stock_predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_models_symbol ON stock_prediction_models(symbol); 