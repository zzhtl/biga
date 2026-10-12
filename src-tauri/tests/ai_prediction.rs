//! AI 验收固定使用模拟行情与模型，不访问付费接口。
use biga_lib::{ai::{process::{AiRunner, CliFuture, CliReply}, service, types::*}, db::{connection::{open_pool, run_migrations}, models::HistoricalData, repository, DbPool}, services::market_data::{ensure_with_source, expiry, FetchResult, MarketSource, QueryBudget, Resource}};
use chrono::{Duration, NaiveDate};
use serde_json::{json, Value};
use std::sync::{atomic::{AtomicU32, Ordering}, Mutex};
use tokio::sync::watch;

async fn database() -> (tempfile::TempDir, DbPool) {
    let dir = tempfile::tempdir().unwrap();
    let pool = open_pool(&dir.path().join("test.db")).await.unwrap();
    run_migrations(&pool).await.unwrap();
    sqlx::query("INSERT INTO stock_info(symbol,name,exchange) VALUES('600519','测试股票','sh')").execute(&pool).await.unwrap();
    (dir, pool)
}

fn history() -> Vec<HistoricalData> {
    (0..300).map(|i| {
        let close = 20.0 + i as f64 * 0.02 + (i as f64 / 7.0).sin();
        HistoricalData { symbol: "600519".into(), date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap() + Duration::days(i), open: close - 0.1, high: close + 0.2, low: close - 0.2, close, volume: 1000000 + i * 100, amount: close * 1000000.0, amplitude: 2.0, turnover_rate: 0.0, volume_ratio: 0.0, change_percent: 0.5, change: 0.1 }
    }).collect()
}

#[derive(Default)]
struct Source { calls: AtomicU32, rate_limit: bool }
impl MarketSource for Source {
    fn fetch<'a>(&'a self, resource: Resource, _symbol: &'a str, budget: &'a QueryBudget) -> FetchResult<'a> {
        Box::pin(async move {
            budget.reserve()?;
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.rate_limit { return Err(biga_lib::error::AppError::MarketRateLimited(120)); }
            tokio::task::yield_now().await;
            Ok(match resource {
                Resource::History => serde_json::to_value(history()).unwrap(),
                Resource::Valuation => json!({"circulating_market_cap":1000000000.0,"total_market_cap":2000000000.0,"pe":18.0,"pb":2.0}),
                Resource::Fundamentals => json!([]),
            })
        })
    }
}

fn config() -> AiConnectionConfig {
    AiConnectionConfig { id:"test".into(), name:"测试 AI".into(), provider:Provider::Codex, transport:Transport::Local, executable:"codex".into(), model:Some("test-model".into()), host:None, user:None, port:None, identity_file:None }
}
fn assessment() -> Assessment {
    Assessment { summary:"基于日线呈现缓慢上升的基准情景".into(), direction:"up".into(), factors:vec![Evidence {text:"价格上升".into(), sources:vec!["history".into()]}], bullish:"放量上涨".into(), base:"缓慢上涨".into(), bearish:"跌破支撑".into(), risks:vec!["波动放大".into()], invalidation:vec!["跌破近期低点".into()], forecast:(1..=5).map(|day| ForecastPoint { day, lower:20.0, expected:22.0 + day as f64, upper:30.0 }).collect() }
}
struct Runner { calls: AtomicU32, queries: bool, contexts: Mutex<Vec<Value>> }
impl AiRunner for Runner {
    fn run<'a>(&'a self, _config: &'a AiConnectionConfig, prompt: String, _cancel: watch::Receiver<bool>) -> CliFuture<'a> {
        Box::pin(async move {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            self.contexts.lock().unwrap().push(serde_json::from_str(&prompt).unwrap());
            let envelope = if self.queries && n == 0 { AiEnvelope { kind:"query".into(), queries:vec![DataQuery {resource:"history".into(),history_days:300}, DataQuery {resource:"fundamentals".into(),history_days:300}], report:None } }
                else { AiEnvelope {kind:"report".into(),queries:vec![],report:Some(assessment())} };
            Ok(CliReply {envelope,usage:json!({"input_tokens":100})})
        })
    }
}
async fn insert_job(pool: &DbPool, id: &str) {
    sqlx::query("INSERT INTO ai_prediction_jobs(id,symbol,connection_id,request_json,status,stage,created_at) VALUES(?,'600519','test','{}','preparing','','2026-01-01')").bind(id).execute(pool).await.unwrap();
}

#[tokio::test]
async fn concurrent_symbol_alias_queries_share_one_request_and_survive_restart() {
    let (dir, pool) = database().await;
    let source = Source::default();
    let a = QueryBudget::new(3); let b = QueryBudget::new(3);
    let now = chrono::Utc::now().timestamp();
    let (left, right) = tokio::join!(ensure_with_source(&pool,"sh600519",Resource::History,&a,&source,now), ensure_with_source(&pool,"600519.SH",Resource::History,&b,&source,now));
    assert!(!left.unwrap().stale); assert!(!right.unwrap().stale);
    assert_eq!(source.calls.load(Ordering::SeqCst),1);
    assert_eq!(a.counts().0 + b.counts().0,1);
    assert_eq!(a.counts().1 + b.counts().1,1);
    pool.close().await;
    let pool = open_pool(&dir.path().join("test.db")).await.unwrap();
    ensure_with_source(&pool,"600519",Resource::History,&a,&source,now + 1).await.unwrap();
    assert_eq!(source.calls.load(Ordering::SeqCst),1);
    assert_eq!(repository::get_recent_historical_data("600519",60,&pool).await.unwrap().len(),60);
    assert_eq!(repository::get_recent_historical_data("600519",250,&pool).await.unwrap().len(),250);
}

#[tokio::test]
async fn negative_cache_rate_limit_and_budget_do_not_retry() {
    let (_dir,pool)=database().await;
    let source=Source::default(); let budget=QueryBudget::new(1); let now=chrono::Utc::now().timestamp();
    ensure_with_source(&pool,"600519",Resource::Fundamentals,&budget,&source,now).await.unwrap();
    ensure_with_source(&pool,"600519",Resource::Fundamentals,&budget,&source,now + 1).await.unwrap();
    let stopped=ensure_with_source(&pool,"600519",Resource::History,&budget,&source,now).await.unwrap();
    assert!(stopped.warning.unwrap().contains("额度"));
    assert_eq!(budget.counts(),(1,1));
    assert_eq!(source.calls.load(Ordering::SeqCst),1);
    let limited=Source {rate_limit:true,..Source::default()}; let budget=QueryBudget::new(3);
    ensure_with_source(&pool,"000001",Resource::History,&budget,&limited,now).await.unwrap();
    ensure_with_source(&pool,"000001",Resource::History,&budget,&limited,now + 61).await.unwrap();
    assert_eq!(limited.calls.load(Ordering::SeqCst),1);
    ensure_with_source(&pool,"000001",Resource::History,&budget,&limited,now + 121).await.unwrap();
    assert_eq!(limited.calls.load(Ordering::SeqCst),2);
}

#[test]
fn history_expiry_uses_shanghai_close_and_skips_weekend() {
    let before=chrono::DateTime::parse_from_rfc3339("2026-07-17T15:00:00+08:00").unwrap().timestamp();
    let after=chrono::DateTime::parse_from_rfc3339("2026-07-17T17:00:00+08:00").unwrap().timestamp();
    assert_eq!(expiry(Resource::History,before),before+3600);
    assert_eq!(expiry(Resource::History,after),chrono::DateTime::parse_from_rfc3339("2026-07-20T16:00:00+08:00").unwrap().timestamp());
}

#[tokio::test]
async fn full_report_queries_share_budget_and_report_cache_avoids_model_call() {
    let (_dir,pool)=database().await;
    let source=Source::default(); let budget=QueryBudget::new(3);
    let runner=Runner {calls:AtomicU32::new(0),queries:true,contexts:Mutex::new(vec![])};
    let request=AiPredictionRequest {symbol:"600519".into(),connection_id:"test".into(),days:5,regenerate:false};
    let (_sender,cancel)=watch::channel(false);
    insert_job(&pool,"one").await;
    let report=service::run_prediction(&pool,"one",&config(),&request,&budget,&runner,&source,cancel.clone(),None).await.unwrap();
    assert_eq!(budget.counts(),(3,2));
    assert_eq!(runner.calls.load(Ordering::SeqCst),2);
    {
        let contexts = runner.contexts.lock().unwrap();
        let technical = &contexts[0]["data"]["technical"];
        assert!(technical.get("indicators").is_some());
        assert!(technical.get("predictions").is_none(), "AI 应独立推导价格，不能复述旧引擎的未来序列");
    }
    assert_eq!(report.snapshot.bars.len(),300);
    assert_eq!(report.dates.len(),5);
    assert!(report.dates.windows(2).all(|d| d[0]<d[1]));
    assert!(report.snapshot.warnings.iter().any(|s| s.contains("基本面")));
    sqlx::query("UPDATE ai_prediction_jobs SET status='completed',report_json=? WHERE id='one'").bind(serde_json::to_string(&report).unwrap()).execute(&pool).await.unwrap();
    insert_job(&pool,"two").await;
    let second=QueryBudget::new(3);
    service::run_prediction(&pool,"two",&config(),&request,&second,&runner,&source,cancel,None).await.unwrap();
    assert_eq!(second.counts(),(0,3));
    assert_eq!(runner.calls.load(Ordering::SeqCst),2);
}

#[tokio::test]
async fn missing_data_and_cancelled_tasks_never_call_ai() {
    let (_dir,pool)=database().await;
    let source=Source {rate_limit:true,..Source::default()}; let budget=QueryBudget::new(3);
    let runner=Runner {calls:AtomicU32::new(0),queries:false,contexts:Mutex::new(vec![])};
    let request=AiPredictionRequest {symbol:"600519".into(),connection_id:"test".into(),days:5,regenerate:false};
    let (sender,cancel)=watch::channel(false);
    insert_job(&pool,"one").await;
    assert!(service::run_prediction(&pool,"one",&config(),&request,&budget,&runner,&source,cancel.clone(),None).await.unwrap_err().contains("60"));
    assert_eq!(runner.calls.load(Ordering::SeqCst),0);
    sender.send(true).unwrap();
    assert!(service::run_prediction(&pool,"one",&config(),&request,&budget,&runner,&source,cancel,None).await.unwrap_err().contains("取消"));
    assert_eq!(budget.counts().0,3);
}

#[test]
fn report_rejects_missing_points_invalid_prices_and_unavailable_evidence() {
    let mut report=assessment();
    assert!(report.validate(5,&["history"]).is_ok());
    assert!(report.validate(3,&["history"]).is_err());
    report.forecast[0].expected=f64::NAN;
    assert!(report.validate(5,&["history"]).is_err());
    let mut report=assessment(); report.forecast[0].lower=50.0;
    assert!(report.validate(5,&["history"]).is_err());
    let mut report=assessment(); report.factors[0].sources=vec!["fundamentals".into()];
    assert!(report.validate(5,&["history"]).is_err());
}

#[tokio::test]
async fn data_revision_and_model_change_invalidate_cached_report() {
    let (_dir,pool)=database().await;
    repository::batch_insert_historical_data("600519",&pool,history()).await.unwrap();
    let request=AiPredictionRequest {symbol:"600519".into(),connection_id:"test".into(),days:5,regenerate:false};
    let initial=service::snapshot(&pool,"600519",5,250,vec![]).await.unwrap();
    let key=service::report_key(&config(),&request,&initial);
    sqlx::query("UPDATE historical_data SET close=close+1 WHERE symbol='600519' AND date='2025-01-01'").execute(&pool).await.unwrap();
    let revised=service::snapshot(&pool,"600519",5,250,vec![]).await.unwrap();
    assert_ne!(key,service::report_key(&config(),&request,&revised));
    let mut other=config(); other.model=Some("another-model".into());
    assert_ne!(key,service::report_key(&other,&request,&initial));
}

#[tokio::test]
async fn migration_and_recovery_preserve_completed_reports() {
    let (_dir,pool)=database().await;
    insert_job(&pool,"one").await;
    insert_job(&pool,"two").await;
    sqlx::query("UPDATE ai_prediction_jobs SET status='completed',report_json='{}' WHERE id='two'").execute(&pool).await.unwrap();
    run_migrations(&pool).await.unwrap(); service::recover_jobs(&pool).await.unwrap();
    assert_eq!(service::get_job(&pool,"one").await.unwrap().status,"interrupted");
    assert_eq!(service::get_job(&pool,"two").await.unwrap().report_json.as_deref(),Some("{}"));
}

#[tokio::test]
async fn refreshed_history_replaces_revised_prices_without_duplicating_bars() {
    let (_dir,pool)=database().await;
    let bars=history();
    repository::batch_insert_historical_data("600519",&pool,bars.clone()).await.unwrap();
    let mut revision=bars.last().unwrap().clone();
    revision.close+=1.0; revision.high=revision.close+0.2;
    repository::batch_insert_historical_data("600519",&pool,vec![revision.clone()]).await.unwrap();
    let actual=repository::get_recent_historical_data("600519",3000,&pool).await.unwrap();
    assert_eq!(actual.len(),300);
    assert_eq!(actual.last().unwrap().close,revision.close);
}

#[tokio::test]
#[cfg(unix)]
async fn concurrent_start_returns_one_job_and_cancel_does_not_store_a_report() {
    use std::{os::unix::fs::PermissionsExt, sync::Arc};
    let (dir,pool)=database().await;
    let source=Source::default(); let budget=QueryBudget::new(3);
    for resource in [Resource::History,Resource::Valuation,Resource::Fundamentals] {
        ensure_with_source(&pool,"600519",resource,&budget,&source,chrono::Utc::now().timestamp()).await.unwrap();
    }
    let counter=dir.path().join("calls");
    let executable=dir.path().join("mock-claude");
    let response=json!({"subtype":"success","structured_output":{"kind":"report","queries":[],"report":assessment()}});
    std::fs::write(&executable,format!("#!/bin/sh\nprintf x >> {}\nsleep 1\ncat >/dev/null\nprintf '%s' {}",biga_lib::ai::process::shell_quote(&counter.to_string_lossy()),biga_lib::ai::process::shell_quote(&response.to_string()))).unwrap();
    std::fs::set_permissions(&executable,std::fs::Permissions::from_mode(0o700)).unwrap();
    let mut connection=config(); connection.provider=Provider::Claude; connection.executable=executable.to_string_lossy().into_owned();
    service::save_connection(&pool,connection).await.unwrap();
    let state=Arc::new(service::AiState::default());
    let request=AiPredictionRequest {symbol:"600519".into(),connection_id:"test".into(),days:5,regenerate:false};
    let (first,second)=tokio::join!(service::start(pool.clone(),state.clone(),None,request.clone()),service::start(pool.clone(),state.clone(),None,request.clone()));
    let first=first.unwrap(); assert_eq!(first.id,second.unwrap().id);
    let completed=tokio::time::timeout(std::time::Duration::from_secs(8),async {
        loop { let job=service::get_job(&pool,&first.id).await.unwrap(); if job.status=="completed" || job.status=="failed" { break job; } tokio::time::sleep(std::time::Duration::from_millis(25)).await; }
    }).await.unwrap();
    assert_eq!(completed.status,"completed", "{:?}",completed.error);
    assert_eq!(completed.requests,0);
    assert_eq!(std::fs::read_to_string(&counter).unwrap(),"x");
    let cancelled=service::start(pool.clone(),state.clone(),None,AiPredictionRequest {regenerate:true,..request}).await.unwrap();
    service::cancel(&state,&cancelled.id).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5),async {
        loop { let job=service::get_job(&pool,&cancelled.id).await.unwrap(); if job.status=="cancelled" { assert!(job.report_json.is_none()); break; } tokio::time::sleep(std::time::Duration::from_millis(25)).await; }
    }).await.unwrap();
}
