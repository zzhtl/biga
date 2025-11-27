/**
 * 股票服务
 */

import { invoke } from '@tauri-apps/api/core';
import type { StockInfo, HistoricalData, RealtimeData, PagedResponse } from '../types';

/**
 * 获取股票列表
 */
export async function getStockList(
  keyword?: string,
  page: number = 1,
  pageSize: number = 20
): Promise<PagedResponse<StockInfo>> {
  const result = await invoke<{ data: StockInfo[]; total: number }>('get_stock_list', {
    keyword,
    page,
    pageSize,
  });
  
  return {
    data: result.data,
    total: result.total,
    page,
    page_size: pageSize,
  };
}

/**
 * 获取股票信息
 */
export async function getStockInfos(symbols: string[]): Promise<StockInfo[]> {
  return invoke<StockInfo[]>('get_stock_infos', { symbols });
}

/**
 * 刷新股票信息
 */
export async function refreshStockInfos(): Promise<void> {
  await invoke('refresh_stock_infos');
}

/**
 * 获取历史数据
 */
export async function getHistoricalData(
  symbol: string,
  startDate?: string,
  endDate?: string
): Promise<HistoricalData[]> {
  return invoke<HistoricalData[]>('get_historical_data', {
    symbol,
    startDate,
    endDate,
  });
}

/**
 * 刷新历史数据
 */
export async function refreshHistoricalData(symbol: string): Promise<void> {
  await invoke('refresh_historical_data', { symbol });
}

/**
 * 获取实时数据
 */
export async function getRealtimeData(
  keyword?: string,
  page: number = 1,
  pageSize: number = 20
): Promise<PagedResponse<RealtimeData>> {
  const result = await invoke<{ data: RealtimeData[]; total: number }>('get_realtime_data', {
    keyword,
    page,
    pageSize,
  });
  
  return {
    data: result.data,
    total: result.total,
    page,
    page_size: pageSize,
  };
}

