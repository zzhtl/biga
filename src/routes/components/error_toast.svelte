<script lang="ts">
    export let message: string;
</script>

{#if message}
    <div class="toast" on:click={() => (message = "")}>
        {message}
    </div>
{/if}

<style>
    .toast {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #ef4444;
        color: white;
        padding: 1rem 2rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: transform 0.3s ease;
    }

    .toast:hover {
        transform: translateY(-2px);
    }
</style>
