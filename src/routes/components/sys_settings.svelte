<script lang="ts">
    let apiKey = localStorage.getItem("apiKey") || "";

    function saveSettings() {
        localStorage.setItem("apiKey", apiKey);
    }
</script>

<div class="settings-container">
    <h1>系统设置</h1>

    <div class="setting-item">
        <label>API 密钥</label>
        <input
            type="password"
            bind:value={apiKey}
            placeholder="输入您的API密钥"
        />
    </div>

    <button on:click={saveSettings}>保存设置</button>
</div>

<style>
    .settings-container {
        max-width: 600px;
        margin: 0 auto;
    }

    .setting-item {
        margin: 1.5rem 0;
    }

    label {
        display: block;
        margin-bottom: 0.5rem;
        color: #94a3b8;
    }

    input {
        width: 100%;
        padding: 0.875rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        color: inherit;
    }
</style>
