<script lang="ts">
    import { onMount } from 'svelte';
    import { listAiConnections, saveAiConnection, deleteAiConnection, checkAiConnection } from '../services/ai';
    import { errorMessage } from '../services/core';
    import type { AiConnectionConfig } from '../types/ai';

    let connections = $state<AiConnectionConfig[]>([]);
    const empty = (): AiConnectionConfig => ({ id: '', name: '', provider: 'codex', transport: 'local', executable: 'codex', model: null, host: null, user: null, port: null, identity_file: null });
    let form = $state<AiConnectionConfig>(empty());
    let busy = $state(false);
    let message = $state('');
    let error = $state('');

    async function load() {
        try { connections = await listAiConnections(); }
        catch (e) { error = errorMessage(e, '无法读取 AI 连接'); }
    }
    async function perform(action: 'save' | 'check' | 'delete') {
        busy = true; error = ''; message = '';
        try {
            if (action === 'save') {
                form = await saveAiConnection($state.snapshot(form));
                message = '连接已保存，未调用 AI';
                await load();
            } else if (action === 'check') {
                message = await checkAiConnection($state.snapshot(form));
            } else {
                await deleteAiConnection(form.id);
                form = empty(); await load(); message = '连接已删除，历史报告保留';
            }
        } catch (e) { error = errorMessage(e, '操作失败'); }
        finally { busy = false; }
    }
    onMount(() => { void load(); });
</script>

<section class="ai-connections" aria-labelledby="ai-settings-heading">
    <h2 id="ai-settings-heading">Claude / Codex 连接</h2>
    <p>复用对应机器上已登录的 CLI。检查连接不会生成报告，也不会消耗模型额度。</p>
    <div class="connection-list">
        {#each connections as connection}
            <button class:chosen={form.id === connection.id} disabled={busy} onclick={() => { form = { ...connection }; message = ''; error = ''; }}>{connection.name} · {connection.transport === 'ssh' ? 'SSH' : '本地'}</button>
        {/each}
        <button disabled={busy} onclick={() => { form = empty(); error = ''; message = ''; }}>＋ 新建连接</button>
    </div>
    <form onsubmit={(e) => { e.preventDefault(); void perform('save'); }}>
        <fieldset disabled={busy}>
            <div class="fields">
                <label>连接名称<input bind:value={form.name} placeholder="例如：本机 Codex" required maxlength="100" /></label>
                <label>AI 提供方<select bind:value={form.provider} onchange={() => { form.executable = form.provider; }}><option value="codex">Codex</option><option value="claude">Claude Code</option></select></label>
                <label>运行位置<select bind:value={form.transport}><option value="local">本机 CLI</option><option value="ssh">SSH 远程 CLI</option></select></label>
                <label>可执行文件<input bind:value={form.executable} placeholder="codex 或可执行文件绝对路径" required /></label>
                <label>模型（可选）<input bind:value={form.model} placeholder="留空使用 CLI 默认模型" /></label>
                {#if form.transport === 'ssh'}
                    <label>主机 / SSH 别名<input bind:value={form.host} placeholder="my-server 或 192.168.1.10" required /></label>
                    <label>SSH 用户（可选）<input bind:value={form.user} placeholder="留空使用 SSH 配置" /></label>
                    <label>端口（可选）<input type="number" bind:value={form.port} min="1" max="65535" placeholder="使用 SSH 配置，默认 22" /></label>
                    <label>本机私钥路径（可选）<input bind:value={form.identity_file} placeholder="留空使用 SSH agent / 配置" /></label>
                {/if}
            </div>
            <p class="note">{form.transport === 'ssh' ? '支持 Linux/macOS 服务器。请先在终端验证 SSH 主机指纹，并在服务器登录 CLI；建议填写 CLI 的绝对路径。' : '请先安装并登录 CLI。桌面应用找不到命令时，可填写可执行文件的绝对路径。'}分析任务隔离工具配置，只使用本应用提供的数据。</p>
            <div class="actions">
                <button class="primary" type="submit">{busy ? '处理中…' : '保存连接'}</button>
                <button type="button" onclick={() => perform('check')}>检查连接</button>
                {#if form.id}<button class="danger" type="button" onclick={() => perform('delete')}>删除连接</button>{/if}
            </div>
        </fieldset>
    </form>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    {#if message}<p class="success" role="status">{message}</p>{/if}
</section>

<style>
    .ai-connections { padding: 1.4rem; margin-top: 1.4rem; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-1); }
    h2 { margin: 0 0 .5rem; font-size: 1.1rem; }
    p { color: var(--text-secondary); }
    .connection-list, .actions { display: flex; flex-wrap: wrap; gap: .6rem; margin: 1rem 0; }
    .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
    label { display: flex; flex-direction: column; gap: .4rem; font-size: .85rem; }
    fieldset { border: 0; padding: 0; margin: 0; min-width: 0; }
    button { padding: .6rem .8rem; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--surface-2); color: var(--text-primary); cursor: pointer; }
    button.chosen, .primary { border-color: var(--accent); background: var(--accent-muted); }
    .danger, .error { color: var(--danger); }
    .success { color: var(--success); }
    .note { font-size: .8rem; }
    button:disabled { opacity: .55; cursor: wait; }
</style>
