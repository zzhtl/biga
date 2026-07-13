import type { ApiTokenStatus } from '../types';
import { invokeCommand } from './core';

export function getApiTokenStatus(): Promise<ApiTokenStatus> {
  return invokeCommand('get_api_token_status');
}

export function saveApiToken(token: string): Promise<ApiTokenStatus> {
  return invokeCommand('save_api_token', { token });
}

export function clearApiToken(): Promise<ApiTokenStatus> {
  return invokeCommand('clear_api_token');
}

export function testApiToken(): Promise<boolean> {
  return invokeCommand('test_api_token');
}
