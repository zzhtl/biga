//! CLI 协议适配。用户文本只从 stdin 传入，SSH 仅执行应用生成的固定脚本。

use super::types::{output_schema, AiConnectionConfig, AiEnvelope, Provider, Transport};
use serde_json::{json, Value};
use std::{future::Future, pin::Pin, process::Stdio, time::Duration};
use tokio::{io::{AsyncRead, AsyncReadExt, AsyncWriteExt}, process::Command, sync::watch};

pub struct CliReply { pub envelope: AiEnvelope, pub usage: Value }
pub type CliFuture<'a> = Pin<Box<dyn Future<Output = Result<CliReply, String>> + Send + 'a>>;
pub trait AiRunner: Send + Sync {
    fn run<'a>(&'a self, config: &'a AiConnectionConfig, prompt: String, cancel: watch::Receiver<bool>) -> CliFuture<'a>;
}
pub struct CliRunner;
impl AiRunner for CliRunner {
    fn run<'a>(&'a self, config: &'a AiConnectionConfig, prompt: String, cancel: watch::Receiver<bool>) -> CliFuture<'a> {
        Box::pin(run_cli(config, prompt, cancel))
    }
}

pub fn shell_quote(value: &str) -> String { format!("'{}'", value.replace('\'', "'\\''")) }

fn cli_args(config: &AiConnectionConfig, schema: &str) -> Vec<String> {
    let mut args: Vec<String> = match config.provider {
        Provider::Codex => vec!["exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--output-schema", schema, "-c", "project_doc_max_bytes=0", "-c", "features.shell_tool=false", "-c", "features.unified_exec=false", "-c", "features.apps=false", "-c", "web_search=\"disabled\""],
        Provider::Claude => vec!["-p", "--output-format", "json", "--tools", "", "--permission-mode", "dontAsk", "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}", "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence", "--max-turns", "3"],
    }.into_iter().map(str::to_owned).collect();
    if config.provider == Provider::Claude { args.extend(["--json-schema".into(), output_schema().to_string()]); }
    if let Some(model) = config.model.as_ref().filter(|m| !m.trim().is_empty()) { args.extend(["--model".into(), model.clone()]); }
    if config.provider == Provider::Codex { args.push("-".into()); }
    args
}

fn ssh_command(config: &AiConnectionConfig, script: &str) -> Command {
    let mut command = Command::new("ssh");
    command.args(["-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]);
    if let Some(port) = config.port { command.args(["-p", &port.to_string()]); }
    if let Some(path) = config.identity_file.as_ref().filter(|p| !p.is_empty()) { command.arg("-i").arg(path); }
    if let Some(user) = config.user.as_ref().filter(|u| !u.is_empty()) { command.arg("-l").arg(user); }
    command.arg("--").arg(config.host.as_deref().unwrap_or_default());
    command.arg(format!("sh -c {}", shell_quote(script)));
    command
}

/// 看门狗在 SSH 意外断开时仍限制远端模型进程的运行时间。
fn remote_script(config: &AiConnectionConfig, args: &[String], seconds: u64) -> String {
    let invocation = std::iter::once(&config.executable).chain(args).map(|a| shell_quote(a)).collect::<Vec<_>>().join(" ");
    format!(r#"umask 077
task_dir=$(mktemp -d) || exit 1
ai_pid=''
watchdog=''
cleanup() {{
  [ -z "$ai_pid" ] || kill "$ai_pid" 2>/dev/null || :
  [ -z "$watchdog" ] || kill "$watchdog" 2>/dev/null || :
  rm -rf "$task_dir"
}}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
cd "$task_dir" || exit 1
printf '%s' {} > schema.json
unset STOCK_API_TOKEN
export MAX_STRUCTURED_OUTPUT_RETRIES=0
exec 3<&0
{} <&3 &
ai_pid=$!
( trap 'kill "$sleeper" 2>/dev/null; exit' TERM HUP INT
  sleep {} & sleeper=$!
  wait "$sleeper"
  kill "$ai_pid" 2>/dev/null
  sleep 2
  kill -KILL "$ai_pid" 2>/dev/null
) </dev/null >/dev/null 2>&1 3<&- &
watchdog=$!
wait "$ai_pid"
status=$?
ai_pid=''
exit "$status"
"#, shell_quote(&output_schema().to_string()), invocation, seconds)
}

async fn read_limited(reader: impl AsyncRead + Unpin, limit: usize) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    reader.take(limit as u64 + 1).read_to_end(&mut bytes).await.map_err(|_| "无法读取 CLI 输出".to_string())?;
    if bytes.len() > limit { return Err("CLI 输出超过大小限制".into()); }
    Ok(bytes)
}

async fn stop_process(command: &mut tokio::process::Child) {
    if let Some(pid) = command.id() {
        #[cfg(unix)]
        { let _ = Command::new("kill").args(["-KILL", "--", &format!("-{pid}")]).stdout(Stdio::null()).stderr(Stdio::null()).status().await; }
        #[cfg(windows)]
        { let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null()).status().await; }
    }
    let _ = command.kill().await;
    let _ = command.wait().await;
}

async fn execute(mut command: Command, input: String, mut cancel: watch::Receiver<bool>, seconds: u64) -> Result<String, String> {
    if *cancel.borrow() { return Err("任务已取消".into()); }
    command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true).env_remove("STOCK_API_TOKEN").env("MAX_STRUCTURED_OUTPUT_RETRIES", "0");
    #[cfg(unix)]
    command.process_group(0);
    let mut child = command.spawn().map_err(|e| format!("无法启动 CLI/SSH：{e}"))?;
    let mut stdin = child.stdin.take().ok_or("CLI stdin 不可用")?;
    let stdout = child.stdout.take().ok_or("CLI stdout 不可用")?;
    let stderr = child.stderr.take().ok_or("CLI stderr 不可用")?;
    let work = async {
        let write = async { stdin.write_all(input.as_bytes()).await.map_err(|_| "无法向 CLI 传入分析数据".to_string())?; drop(stdin); Ok::<_, String>(()) };
        let (written, output, errors, status) = tokio::join!(write, read_limited(stdout, 2 * 1024 * 1024), read_limited(stderr, 256 * 1024), async { child.wait().await.map_err(|_| "无法等待 CLI 退出".to_string()) });
        let status = status?;
        let errors = errors?;
        if !status.success() {
            let diagnostics = format!("{}\n{}", String::from_utf8_lossy(&errors), String::from_utf8_lossy(output.as_deref().unwrap_or_default()));
            return Err(classify_error(&diagnostics, status.code()));
        }
        written?;
        String::from_utf8(output?).map_err(|_| "CLI 输出不是有效 UTF-8".into())
    };
    let result = tokio::select! {
        result = tokio::time::timeout(Duration::from_secs(seconds), work) => result.unwrap_or_else(|_| Err("AI 执行超时，未自动重试".into())),
        _ = cancel.changed() => Err("任务已取消".into()),
    };
    if result.is_err() { stop_process(&mut child).await; }
    result
}

fn classify_error(message: &str, code: Option<i32>) -> String {
    let lower = message.to_lowercase();
    let hint = if lower.contains("host key verification failed") { "SSH 主机身份未验证，请先在终端连接并核对指纹" }
        else if lower.contains("permission denied") { "SSH 密钥认证失败或 CLI 没有执行权限" }
        else if lower.contains("login") || lower.contains("unauthorized") || lower.contains("authentication") || lower.contains("401") { "AI 尚未登录或认证失效，请在对应机器上登录 CLI" }
        else if lower.contains("429") || lower.contains("rate limit") || lower.contains("quota") { "AI 额度不足或被限流，未自动重试" }
        else if lower.contains("not found") || code == Some(127) { "找不到 CLI，请配置可执行文件的绝对路径" }
        else if lower.contains("config.toml") { "CLI 配置不兼容，请检查提供方配置或升级 CLI" }
        else if lower.contains("unexpected argument") || lower.contains("unknown option") { "CLI 版本不支持所需参数，请升级后重新检查" }
        else { "CLI 执行失败，请检查登录、模型名称和网络连接" };
    format!("{hint}（退出码 {}）", code.map(|v| v.to_string()).unwrap_or_else(|| "signal".into()))
}

pub async fn check_connection(config: &AiConnectionConfig) -> Result<String, String> {
    config.validate()?;
    let (_sender, cancel) = watch::channel(false);
    let args = if config.provider == Provider::Codex { vec!["exec", "--help"] } else { vec!["--help"] };
    let mut command = if config.transport == Transport::Ssh {
        let script = format!("{} {}", shell_quote(&config.executable), args.iter().map(|a| shell_quote(a)).collect::<Vec<_>>().join(" "));
        ssh_command(config, &script)
    } else { let mut c = Command::new(&config.executable); c.args(args); c };
    command.current_dir(std::env::temp_dir());
    let output = execute(command, String::new(), cancel, 20).await?;
    let required: &[&str] = match config.provider { Provider::Claude => &["--json-schema", "--tools", "--strict-mcp-config", "--setting-sources"], Provider::Codex => &["--output-schema", "--ignore-user-config", "--ephemeral"] };
    if required.iter().any(|flag| !output.contains(flag)) { return Err("CLI 缺少结构化输出或隔离参数，请升级 CLI".into()); }
    Ok("CLI 与连接检查通过；未消耗模型额度，认证在生成报告时验证".into())
}

pub async fn run_cli(config: &AiConnectionConfig, prompt: String, cancel: watch::Receiver<bool>) -> Result<CliReply, String> {
    config.validate()?;
    let temp = tempfile::tempdir().map_err(|_| "无法创建 AI 临时目录")?;
    let schema_path = temp.path().join("schema.json");
    std::fs::write(&schema_path, output_schema().to_string()).map_err(|_| "无法写入输出 schema")?;
    let command = if config.transport == Transport::Ssh {
        let args = cli_args(config, "schema.json");
        ssh_command(config, &remote_script(config, &args, 180))
    } else {
        let args = cli_args(config, &schema_path.to_string_lossy());
        let mut command = Command::new(&config.executable);
        command.args(args).current_dir(temp.path());
        command
    };
    let output = execute(command, prompt, cancel, 190).await?;
    parse_reply(&config.provider, &output)
}

pub fn parse_reply(provider: &Provider, output: &str) -> Result<CliReply, String> {
    let invalid = |_| "AI 输出不符合报告协议，未自动重试".to_string();
    let (value, usage) = match provider {
        Provider::Claude => {
            let value: Value = serde_json::from_str(output).map_err(invalid)?;
            if value.get("is_error").and_then(Value::as_bool) == Some(true) || value.get("subtype").and_then(Value::as_str).is_some_and(|s| s != "success") { return Err("Claude 未成功完成结构化报告，未自动重试".into()); }
            let report = value.get("structured_output").cloned().ok_or("Claude 没有返回 structured_output，请确认版本与模型支持")?;
            (report, json!({"usage":value.get("usage"),"cost_usd":value.get("total_cost_usd")}))
        }
        Provider::Codex => {
            let mut final_text = None;
            let mut usage = Value::Null;
            let mut completed = false;
            for line in output.lines().filter(|line| !line.trim().is_empty()) {
                let event: Value = serde_json::from_str(line).map_err(invalid)?;
                match event.get("type").and_then(Value::as_str) {
                    Some("turn.failed" | "error") => return Err("Codex 执行失败，未自动重试；请检查登录、模型和额度".into()),
                    Some("turn.completed") => { completed = true; usage = event.get("usage").cloned().unwrap_or(Value::Null); }
                    Some("item.completed") if event["item"]["type"] == "agent_message" => { final_text = event["item"]["text"].as_str().map(str::to_owned); }
                    _ => {}
                }
            }
            if !completed { return Err("Codex 输出中断，未收到完成事件".into()); }
            (serde_json::from_str(&final_text.ok_or("Codex 没有返回最终报告")?).map_err(invalid)?, usage)
        }
    };
    let envelope = serde_json::from_value(value).map_err(invalid)?;
    Ok(CliReply { envelope, usage })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quote_never_interprets_shell_content() { assert_eq!(shell_quote("a'$(touch /tmp/no)"), "'a'\\''$(touch /tmp/no)'"); }
    #[test]
    fn truncated_or_failed_output_is_not_a_report() {
        assert!(parse_reply(&Provider::Codex, "{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"{}\"}}").is_err());
        assert!(parse_reply(&Provider::Claude, "{\"is_error\":true,\"structured_output\":{}}").is_err());
    }
    #[test]
    fn both_providers_decode_same_query_protocol() {
        let payload = json!({"kind":"query","queries":[{"resource":"history","history_days":500}],"report":null});
        let claude = json!({"subtype":"success","structured_output":payload}).to_string();
        let codex = format!("{}\n{}", json!({"type":"item.completed","item":{"type":"agent_message","text":payload.to_string()}}), json!({"type":"turn.completed","usage":{"input_tokens":10}}));
        assert_eq!(parse_reply(&Provider::Claude, &claude).unwrap().envelope.queries[0].history_days, 500);
        assert_eq!(parse_reply(&Provider::Codex, &codex).unwrap().envelope.kind, "query");
    }
    #[tokio::test]
    #[cfg(unix)]
    async fn timeout_and_cancel_stop_processes() {
        let (sender, cancel) = watch::channel(false);
        let mut c = Command::new("sh"); c.args(["-c", "sleep 30"]);
        assert!(execute(c, String::new(), cancel.clone(), 1).await.unwrap_err().contains("超时"));
        sender.send(true).unwrap();
        assert!(execute(Command::new("sh"), String::new(), cancel, 1).await.unwrap_err().contains("取消"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn local_and_remote_wrappers_preserve_stdin_and_structured_output_for_both_providers() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let payload = json!({"kind":"query","queries":[{"resource":"history","history_days":250}],"report":null});
        for provider in [Provider::Claude, Provider::Codex] {
            let result = if provider == Provider::Claude { json!({"subtype":"success","structured_output":payload}).to_string() }
                else { format!("{}\n{}",json!({"type":"item.completed","item":{"type":"agent_message","text":payload.to_string()}}),json!({"type":"turn.completed"})) };
            let path = temp.path().join("mock CLI's executable");
            let marker = temp.path().join("should-not-exist");
            let input = format!("中文行情 $(touch {}) ' 引号\n数据", marker.display());
            let script = format!("#!/bin/sh\ninput=$(cat)\n[ \"$input\" = {} ] || exit 9\nprintf '%s\\n' {}\n", shell_quote(&input), shell_quote(&result));
            std::fs::write(&path, script).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            let mut config = AiConnectionConfig { id:"test".into(),name:"test".into(),provider:provider.clone(),transport:Transport::Local,executable:path.to_string_lossy().into_owned(),model:Some("quoted' model".into()),host:Some("example".into()),user:None,port:None,identity_file:None };
            let (_sender,cancel)=watch::channel(false);
            assert_eq!(run_cli(&config,input.clone(),cancel.clone()).await.unwrap().envelope.kind,"query");
            config.transport=Transport::Ssh;
            let script=remote_script(&config,&cli_args(&config,"schema.json"),5);
            let mut command=Command::new("sh"); command.args(["-c",&script]);
            let output=execute(command,input,cancel,8).await.unwrap();
            assert_eq!(parse_reply(&provider,&output).unwrap().envelope.kind,"query");
            assert!(!marker.exists(),"分析文本不能执行为 shell 命令");
        }
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn early_exit_reports_cli_error_instead_of_broken_stdin() {
        let (_sender,cancel)=watch::channel(false);
        let mut command=Command::new("sh"); command.args(["-c","echo 'unknown option' >&2; exit 2"]);
        let result=execute(command,"x".repeat(200000),cancel,5).await.unwrap_err();
        assert!(result.contains("版本不支持"));
    }
}
