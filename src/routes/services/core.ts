import { invoke } from '@tauri-apps/api/core';

export function errorMessage(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("reading 'invoke'") || message.includes('reading "invoke"')) {
    return '当前功能仅在 BigA 桌面应用中可用';
  }
  return message && message !== '[object Object]' ? message : fallback;
}

export async function invokeCommand<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  return invoke<T>(command, args);
}
