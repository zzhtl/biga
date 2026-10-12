//! 所有桌面刷新与 AI 查询共用的持久化缓存和请求预算。

use crate::{api::stock, config::api_token::resolve_api_token, db::{repository, models::*, DbPool}, error::AppError, utils::{canonical_stock_symbol, date::{get_next_trading_day, is_trading_day}}};
use chrono::{DateTime, FixedOffset, NaiveDate, TimeZone, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, future::Future, pin::Pin, sync::{Arc, OnceLock, atomic::{AtomicU32, Ordering}}};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Resource { History, Valuation, Fundamentals }

impl Resource {
    pub fn name(self) -> &'static str {
        match self { Self::History => "history", Self::Valuation => "valuation", Self::Fundamentals => "fundamentals" }
    }
}

#[derive(Debug)]
pub struct QueryBudget {
    pub requests: AtomicU32,
    pub hits: AtomicU32,
    limit: u32,
}

impl QueryBudget {
    pub fn new(limit: u32) -> Self { Self { requests: AtomicU32::new(0), hits: AtomicU32::new(0), limit } }
    pub fn reserve(&self) -> Result<(), AppError> {
        self.requests.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < self.limit).then_some(n + 1))
            .map(|_| ()).map_err(|_| AppError::InvalidInput("本次行情请求额度已用完".into()))
    }
    pub fn counts(&self) -> (u32, u32) { (self.requests.load(Ordering::SeqCst), self.hits.load(Ordering::SeqCst)) }
}

pub type FetchResult<'a> = Pin<Box<dyn Future<Output = Result<Value, AppError>> + Send + 'a>>;

pub trait MarketSource: Send + Sync {
    fn fetch<'a>(&'a self, resource: Resource, symbol: &'a str, budget: &'a QueryBudget) -> FetchResult<'a>;
}

pub struct StockApiSource;

impl MarketSource for StockApiSource {
    fn fetch<'a>(&'a self, resource: Resource, symbol: &'a str, budget: &'a QueryBudget) -> FetchResult<'a> {
        Box::pin(async move {
            // 没有密钥时不计作 HTTP 请求。
            resolve_api_token().await?;
            budget.reserve()?;
            let value = match resource {
                Resource::History => serde_json::to_value(stock::fetch_historical_data(symbol).await?),
                Resource::Valuation => {
                    let q = stock::fetch_stock_capital(symbol).await?;
                    Ok(serde_json::json!({"circulating_market_cap": q.circulating_market_cap, "total_market_cap": q.total_market_cap, "pe": q.pe, "pb": q.pb}))
                }
                Resource::Fundamentals => serde_json::to_value(stock::fetch_financial_indicators(symbol).await?),
            };
            value.map_err(|e| AppError::DeserializationError(e.to_string()))
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataReceipt {
    pub resource: Resource,
    pub fetched_at: i64,
    pub stale: bool,
    pub warning: Option<String>,
}

#[derive(sqlx::FromRow)]
struct CacheMeta { expires_at: i64, retry_at: i64, fetched_at: i64, error: Option<String> }

type QueryLocks = Mutex<HashMap<String, Arc<Mutex<()>>>>;
static LOCKS: OnceLock<QueryLocks> = OnceLock::new();

pub fn canonical_symbol(symbol: &str) -> Result<String, AppError> {
    let symbol = canonical_stock_symbol(symbol.trim());
    if symbol.len() != 6 || !symbol.bytes().all(|b| b.is_ascii_digit()) {
        return Err(AppError::InvalidInput("请输入有效的六位 A 股代码".into()));
    }
    Ok(symbol)
}

fn shanghai(timestamp: i64) -> DateTime<FixedOffset> {
    let offset = FixedOffset::east_opt(8 * 3600).expect("上海时区固定为 UTC+8");
    DateTime::from_timestamp(timestamp, 0).unwrap_or_default().with_timezone(&offset)
}

fn close_time(date: NaiveDate) -> i64 {
    let offset = FixedOffset::east_opt(8 * 3600).expect("有效时区");
    offset.from_local_datetime(&date.and_hms_opt(16, 0, 0).expect("有效收盘时间")).single().expect("固定时区无歧义").timestamp()
}

pub fn expiry(resource: Resource, now: i64) -> i64 {
    match resource {
        Resource::Valuation => now + 900,
        Resource::Fundamentals => now + 86400,
        Resource::History => {
            let local = shanghai(now);
            let date = local.date_naive();
            if is_trading_day(date) && local.hour() < 16 { close_time(date) }
            else { close_time(get_next_trading_day(date)) }
        }
    }
}

pub async fn ensure_data(pool: &DbPool, symbol: &str, resource: Resource, budget: &QueryBudget) -> Result<DataReceipt, AppError> {
    ensure_with_source(pool, symbol, resource, budget, &StockApiSource, Utc::now().timestamp()).await
}

pub async fn ensure_with_source(pool: &DbPool, symbol: &str, resource: Resource, budget: &QueryBudget, source: &dyn MarketSource, now: i64) -> Result<DataReceipt, AppError> {
    let symbol = canonical_symbol(symbol)?;
    let key = format!("{}:{symbol}", resource.name());
    let lock = LOCKS.get_or_init(|| Mutex::new(HashMap::new())).lock().await.entry(key).or_default().clone();
    let _guard = lock.lock().await;
    let meta = sqlx::query_as::<_, CacheMeta>("SELECT expires_at, retry_at, fetched_at, error FROM market_query_cache WHERE resource=? AND symbol=?")
        .bind(resource.name()).bind(&symbol).fetch_optional(pool).await?;
    let available = existing_timestamp(pool, &symbol, resource).await?;
    let (fresh_until, fetched_at) = match &meta {
        Some(m) => (m.expires_at, m.fetched_at),
        None => available.map(|t| (expiry(resource, t), t)).unwrap_or((0, 0)),
    };
    if fresh_until > now {
        budget.hits.fetch_add(1, Ordering::SeqCst);
        return Ok(DataReceipt { resource, fetched_at, stale: false, warning: None });
    }
    if let Some(m) = &meta {
        if m.retry_at > now {
            budget.hits.fetch_add(1, Ordering::SeqCst);
            let message = m.error.clone().unwrap_or_else(|| "行情查询暂时处于冷却期".into());
            return Ok(DataReceipt { resource, fetched_at, stale: true, warning: Some(message) });
        }
    }
    // 先确认缓存可写，并保留尝试记录，避免进程中断后立刻重复请求。
    sqlx::query("INSERT INTO market_query_cache(resource,symbol,retry_at) VALUES(?,?,?) ON CONFLICT(resource,symbol) DO UPDATE SET retry_at=excluded.retry_at")
        .bind(resource.name()).bind(&symbol).bind(now.saturating_add(60)).execute(pool).await?;
    match source.fetch(resource, &symbol, budget).await {
        Ok(value) => {
            persist_value(pool, &symbol, resource, value).await?;
            sqlx::query("INSERT INTO market_query_cache(resource,symbol,expires_at,fetched_at,retry_at,error) VALUES(?,?,?,?,0,NULL) ON CONFLICT(resource,symbol) DO UPDATE SET expires_at=excluded.expires_at,fetched_at=excluded.fetched_at,retry_at=0,error=NULL")
                .bind(resource.name()).bind(&symbol).bind(expiry(resource, now)).bind(now).execute(pool).await?;
            Ok(DataReceipt { resource, fetched_at: now, stale: false, warning: None })
        }
        Err(error) => {
            let delay = match &error { AppError::MarketRateLimited(seconds) => (*seconds).max(60).min(i64::MAX as u64) as i64, _ => 60 };
            let message = error.to_string();
            sqlx::query("INSERT INTO market_query_cache(resource,symbol,retry_at,error) VALUES(?,?,?,?) ON CONFLICT(resource,symbol) DO UPDATE SET retry_at=excluded.retry_at,error=excluded.error")
                .bind(resource.name()).bind(&symbol).bind(now.saturating_add(delay)).bind(&message).execute(pool).await?;
            Ok(DataReceipt { resource, fetched_at, stale: true, warning: Some(message) })
        }
    }
}

async fn existing_timestamp(pool: &DbPool, symbol: &str, resource: Resource) -> Result<Option<i64>, AppError> {
    if resource == Resource::History {
        return Ok(repository::get_recent_historical_data(symbol, 1, pool).await?.last().map(|bar| close_time(bar.date)));
    }
    let sql = match resource {
        Resource::Valuation => "SELECT CAST(strftime('%s', MAX(updated_at)) AS INTEGER) FROM stock_capital WHERE symbol=?",
        _ => "SELECT CAST(strftime('%s', MAX(updated_at)) AS INTEGER) FROM stock_fundamentals WHERE symbol=?",
    };
    Ok(sqlx::query_scalar::<_, Option<i64>>(sql).bind(symbol).fetch_one(pool).await?)
}

async fn persist_value(pool: &DbPool, symbol: &str, resource: Resource, value: Value) -> Result<(), AppError> {
    let decode = |e: serde_json::Error| AppError::DeserializationError(e.to_string());
    match resource {
        Resource::History => {
            let bars: Vec<HistoricalData> = serde_json::from_value(value).map_err(decode)?;
            // 手动输入股票代码也可取数，无需额外消耗股票清单接口额度。
            sqlx::query("INSERT OR IGNORE INTO stock_info(symbol,name,exchange) VALUES(?,?,?)")
                .bind(symbol).bind(symbol).bind(if symbol.starts_with('6') { "sh" } else if symbol.starts_with(['4', '8', '9']) { "bj" } else { "sz" }).execute(pool).await?;
            repository::batch_insert_historical_data(symbol, pool, bars).await?;
        }
        Resource::Valuation => {
            let close = repository::get_latest_close_price(symbol, pool).await?.unwrap_or(0.0);
            let number = |field| value.get(field).and_then(Value::as_f64).filter(|n| n.is_finite()).unwrap_or(0.0);
            let circulating = number("circulating_market_cap");
            let capital = StockCapital { symbol: symbol.into(), circulating_shares: if close > 0.0 { circulating / close } else { 0.0 }, total_shares: if close > 0.0 { number("total_market_cap") / close } else { 0.0 }, circulating_market_cap: circulating, pe: number("pe"), pb: number("pb") };
            repository::upsert_stock_capital(pool, &capital).await?;
        }
        Resource::Fundamentals => {
            let funds: Vec<StockFundamental> = serde_json::from_value(value).map_err(decode)?;
            for f in &funds { repository::upsert_stock_fundamental(pool, f).await?; }
        }
    }
    if resource != Resource::Fundamentals { repository::backfill_volume_metrics(symbol, pool).await?; }
    Ok(())
}
