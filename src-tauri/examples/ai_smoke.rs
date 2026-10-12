//! 一次真实 AI 联调，只读已有行情，绝不调用行情 API，也不自动重试。
//! cargo run --example ai_smoke -- DB_PATH SYMBOL codex|claude CLI_PATH [SSH_ALIAS] [--refresh]
use biga_lib::{ai::{process::{check_connection, run_cli}, service::snapshot, types::*}, db::connection::{open_pool, run_migrations}, db::repository, utils::date::get_next_trading_day};
use serde_json::json;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::path::Path;
use tokio::sync::watch;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let refresh = args.iter().any(|a| a == "--refresh");
    let args: Vec<_> = args.into_iter().filter(|a| a != "--refresh").collect();
    if args.len() < 4 { return Err("用法：ai_smoke DB_PATH SYMBOL codex|claude CLI_PATH [SSH_ALIAS]".into()); }
    let provider = match args[2].as_str() { "claude" => Provider::Claude, "codex" => Provider::Codex, _ => return Err("仅支持 claude/codex".into()) };
    let config = AiConnectionConfig { id: "smoke".into(), name: "真实连接验收".into(), provider, transport: if args.len() > 4 { Transport::Ssh } else { Transport::Local }, executable: args[3].clone(), model: None, host: args.get(4).cloned(), user: None, port: None, identity_file: None };
    println!("{}", check_connection(&config).await?);
    let source = SqlitePoolOptions::new().max_connections(1).connect_with(SqliteConnectOptions::new().filename(Path::new(&args[0])).read_only(true)).await?;
    let temp = tempfile::tempdir()?;
    let pool = open_pool(&temp.path().join("smoke.db")).await?;
    run_migrations(&pool).await?;
    let info = repository::get_stock_info(&args[1], &source).await?;
    repository::batch_insert_stock_info(&pool, vec![info]).await?;
    let bars = repository::get_recent_historical_data(&args[1], 3000, &source).await?;
    repository::batch_insert_historical_data(&args[1], &pool, bars).await?;
    if let Some(capital) = repository::get_stock_capital(&args[1], &source).await? {
        repository::upsert_stock_capital(&pool, &capital).await?;
        let updated: String = sqlx::query_scalar("SELECT updated_at FROM stock_capital WHERE symbol=?").bind(&args[1]).fetch_one(&source).await?;
        sqlx::query("UPDATE stock_capital SET updated_at=? WHERE symbol=?").bind(updated).bind(&args[1]).execute(&pool).await?;
    }
    for f in repository::get_stock_fundamentals(&args[1], &source).await? {
        repository::upsert_stock_fundamental(&pool, &f).await?;
        let updated: String = sqlx::query_scalar("SELECT updated_at FROM stock_fundamentals WHERE symbol=? AND report_date=?").bind(&args[1]).bind(&f.report_date).fetch_one(&source).await?;
        sqlx::query("UPDATE stock_fundamentals SET updated_at=? WHERE symbol=? AND report_date=?").bind(updated).bind(&args[1]).bind(&f.report_date).execute(&pool).await?;
    }
    let budget = biga_lib::services::market_data::QueryBudget::new(if refresh { 3 } else { 0 });
    let mut receipts = vec![];
    if refresh {
        for resource in [biga_lib::services::market_data::Resource::History, biga_lib::services::market_data::Resource::Valuation, biga_lib::services::market_data::Resource::Fundamentals] {
            let receipt = biga_lib::services::market_data::ensure_data(&pool, &args[1], resource, &budget).await?;
            if let Some(warning) = &receipt.warning { println!("{}：{warning}", resource.name()); }
            receipts.push(receipt);
        }
        println!("行情实际请求 {} 次，缓存命中 {} 次", budget.counts().0, budget.counts().1);
    }
    let data = snapshot(&pool, &args[1], 5, 250, receipts).await?;
    let prompt = json!({"task":"用中文分析给定真实 A 股历史行情，独立推导未来5个交易日的情景预测。只使用给定数据，不调用工具、不虚构新闻或命中率。面向普通投资者，不写称呼、不提JSON字段名，情景范围不得称为概率置信区间。输出 schema 要求的完整报告：kind=report, queries=[]，每个依据标出数据源 history/technical/valuation/fundamentals，只引用存在的数据。forecast 恰好5项，day=1..5，0<lower<=expected<=upper。这是最终轮次。", "data":data}).to_string();
    println!("开始一次 AI 生成。数据截至 {}", data.as_of);
    let (_sender, cancel) = watch::channel(false);
    let reply = run_cli(&config, prompt, cancel).await?;
    if reply.envelope.kind != "report" || !reply.envelope.queries.is_empty() { return Err("未返回最终报告，不自动重试".into()); }
    let assessment = reply.envelope.report.ok_or("缺少报告")?;
    let mut sources = vec!["history", "technical"];
    if data.valuation.is_some() { sources.push("valuation"); }
    if !data.fundamentals.is_empty() { sources.push("fundamentals"); }
    assessment.validate(5, &sources)?;
    let mut date = chrono::NaiveDate::parse_from_str(&data.as_of, "%Y-%m-%d")?;
    let dates = (0..5).map(|_| { date = get_next_trading_day(date); date.to_string() }).collect();
    let report = AiPredictionReport { connection_name:config.name,provider:config.provider,model:"CLI 默认模型（未报告型号）".into(),generated_at:chrono::Utc::now().to_rfc3339(),snapshot:data,assessment,dates,usage:vec![reply.usage] };
    let output = Path::new("target").join(format!("ai-smoke-{}-{}.json", args[1], args[2]));
    std::fs::write(&output, serde_json::to_string_pretty(&report)?)?;
    println!("真实报告校验通过：{}；行情请求 {} 次；AI 报告任务 1 次。", output.display(), budget.counts().0);
    Ok(())
}
