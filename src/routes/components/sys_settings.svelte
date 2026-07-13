<script lang="ts">
    import { onMount } from "svelte";
    import { confirm } from "@tauri-apps/plugin-dialog";
    import {
        CheckCircle2,
        Eye,
        EyeOff,
        KeyRound,
        LoaderCircle,
        Save,
        Trash2,
    } from "lucide-svelte";
    import {
        clearApiToken,
        errorMessage,
        getApiTokenStatus,
        saveApiToken,
        testApiToken,
    } from "../services";
    import type { ApiTokenSource, ApiTokenStatus } from "../types";

    let token = $state("");
    let status = $state<ApiTokenStatus>({ configured: false, source: "none" });
    let loading = $state(true);
    let saving = $state(false);
    let testing = $state(false);
    let showToken = $state(false);
    let error = $state("");
    let success = $state("");

    const sourceLabels: Record<ApiTokenSource, string> = {
        keyring: "系统钥匙串",
        environment: "环境变量 STOCK_API_TOKEN",
        none: "未配置",
    };

    async function loadStatus(): Promise<void> {
        loading = true;
        error = "";
        try {
            status = await getApiTokenStatus();
        } catch (reason) {
            error = errorMessage(reason, "无法读取 API 密钥状态");
        } finally {
            loading = false;
        }
    }

    async function saveAndTest(): Promise<void> {
        if (!token.trim()) {
            error = "请输入 API 密钥";
            return;
        }
        saving = true;
        error = "";
        success = "";
        try {
            status = await saveApiToken(token);
            token = "";
            testing = true;
            await testApiToken();
            success = "密钥已保存到系统钥匙串，并通过连接验证";
        } catch (reason) {
            error = errorMessage(reason, "密钥保存或验证失败");
        } finally {
            testing = false;
            saving = false;
        }
    }

    async function testConnection(): Promise<void> {
        testing = true;
        error = "";
        success = "";
        try {
            await testApiToken();
            success = "股票数据服务连接正常";
        } catch (reason) {
            error = errorMessage(reason, "连接验证失败");
        } finally {
            testing = false;
        }
    }

    async function clearToken(): Promise<void> {
        const accepted = await confirm(
            "将删除系统钥匙串中的股票数据 API 密钥。环境变量不受影响。",
            { title: "清除 API 密钥", kind: "warning" },
        );
        if (!accepted) return;
        error = "";
        success = "";
        try {
            status = await clearApiToken();
            success = status.configured
                ? "钥匙串密钥已清除，当前使用环境变量"
                : "API 密钥已清除";
        } catch (reason) {
            error = errorMessage(reason, "清除密钥失败");
        }
    }

    onMount(() => {
        void loadStatus();
    });
</script>

<div class="settings-page">
    <header class="page-header">
        <div>
            <h1>系统设置</h1>
            <p>管理股票数据服务凭据。密钥仅保存在操作系统安全存储中。</p>
        </div>
    </header>

    <section class="settings-section" aria-labelledby="api-heading">
        <div class="section-heading">
            <div class="section-icon"><KeyRound size={18} aria-hidden="true" /></div>
            <div>
                <h2 id="api-heading">股票数据 API</h2>
                <p>用于刷新股票清单、历史行情、估值与基本面数据。</p>
            </div>
        </div>

        <dl class="status-grid" aria-busy={loading}>
            <div>
                <dt>配置状态</dt>
                <dd class:configured={status.configured}>{loading ? "读取中..." : status.configured ? "已配置" : "未配置"}</dd>
            </div>
            <div>
                <dt>当前来源</dt>
                <dd>{sourceLabels[status.source]}</dd>
            </div>
            <div>
                <dt>密钥标识</dt>
                <dd>{status.masked || "-"}</dd>
            </div>
        </dl>

        <div class="token-form">
            <label for="api-token">新 API 密钥</label>
            <div class="token-input">
                <input
                    id="api-token"
                    type={showToken ? "text" : "password"}
                    bind:value={token}
                    autocomplete="off"
                    placeholder="输入后保存到系统钥匙串"
                    onkeydown={(event) => {
                        if (event.key === "Enter") void saveAndTest();
                    }}
                />
                <button class="icon-button" onclick={() => (showToken = !showToken)} aria-label={showToken ? "隐藏密钥" : "显示密钥"} title={showToken ? "隐藏密钥" : "显示密钥"}>
                    {#if showToken}<EyeOff size={17} aria-hidden="true" />{:else}<Eye size={17} aria-hidden="true" />{/if}
                </button>
            </div>
            <div class="form-actions">
                <button class="primary-button" onclick={saveAndTest} disabled={saving || !token.trim()}>
                    {#if saving}<LoaderCircle size={17} class="spin" aria-hidden="true" />保存并验证中{:else}<Save size={17} aria-hidden="true" />保存并验证{/if}
                </button>
                <button class="secondary-button" onclick={testConnection} disabled={testing || !status.configured}>
                    {#if testing}<LoaderCircle size={17} class="spin" aria-hidden="true" />验证中{:else}<CheckCircle2 size={17} aria-hidden="true" />验证连接{/if}
                </button>
                <button class="danger-button" onclick={clearToken} disabled={status.source !== "keyring"}>
                    <Trash2 size={17} aria-hidden="true" />清除钥匙串密钥
                </button>
            </div>
        </div>

        {#if success}<div class="status-panel success" role="status">{success}</div>{/if}
        {#if error}<div class="status-panel error" role="alert">{error}</div>{/if}
    </section>
</div>

<style>
    .settings-page { max-width: 920px; margin: 0 auto; }
    .settings-section { border: 1px solid var(--border); background: var(--surface-1); border-radius: 8px; padding: 1.25rem; }
    .section-heading { display: flex; gap: 0.8rem; align-items: flex-start; }
    .section-heading h2 { margin: 0; font-size: 1rem; }
    .section-heading p { margin: 0.25rem 0 0; color: var(--text-secondary); font-size: 0.84rem; }
    .section-icon { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 6px; background: var(--accent-muted); color: var(--accent); flex: 0 0 auto; }
    .status-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin: 1.25rem 0; border-block: 1px solid var(--border); }
    .status-grid > div { padding: 0.9rem 1rem; border-right: 1px solid var(--border); }
    .status-grid > div:last-child { border-right: 0; }
    dt { color: var(--text-muted); font-size: 0.75rem; }
    dd { margin: 0.25rem 0 0; color: var(--text-primary); font-size: 0.9rem; }
    dd.configured { color: var(--success); }
    .token-form > label { display: block; margin-bottom: 0.45rem; color: var(--text-secondary); font-size: 0.82rem; }
    .token-input { display: flex; gap: 0.5rem; max-width: 640px; }
    .token-input input { flex: 1; min-width: 0; }
    .form-actions { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.85rem; }
    .status-panel { margin-top: 1rem; }
    @media (max-width: 720px) {
        .status-grid { grid-template-columns: 1fr; }
        .status-grid > div { border-right: 0; border-bottom: 1px solid var(--border); }
        .status-grid > div:last-child { border-bottom: 0; }
    }
</style>
