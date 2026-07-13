import { describe, expect, test } from 'bun:test';
import { errorMessage } from './core';

describe('errorMessage', () => {
  test('hides browser-only Tauri implementation errors', () => {
    expect(errorMessage(new TypeError("Cannot read properties of undefined (reading 'invoke')"), '失败'))
      .toBe('当前功能仅在 BigA 桌面应用中可用');
  });

  test('keeps actionable backend errors', () => {
    expect(errorMessage('尚未配置股票数据 API 密钥', '失败'))
      .toBe('尚未配置股票数据 API 密钥');
  });

  test('uses the fallback for opaque values', () => {
    expect(errorMessage({}, '请求失败')).toBe('请求失败');
  });
});
