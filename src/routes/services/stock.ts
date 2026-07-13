import type {
  HistoricalData,
  PagedResponse,
  RealtimeData,
  RealtimeSortColumn,
  SortDirection,
  Stock,
  StockInfo,
} from '../types';
import { invokeCommand } from './core';

export function getStockList(
  search = '',
  page = 1,
  pageSize = 50,
): Promise<PagedResponse<Stock>> {
  return invokeCommand('get_stock_list', { search, page, pageSize });
}

export function getStockInfos(): Promise<StockInfo[]> {
  return invokeCommand('get_stock_infos');
}

export async function refreshStockInfos(): Promise<void> {
  await invokeCommand('refresh_stock_infos');
}

export function getHistoricalData(
  symbol: string,
  start: string,
  end: string,
): Promise<HistoricalData[]> {
  return invokeCommand('get_historical_data', { symbol, start, end });
}

export async function refreshHistoricalData(symbol: string): Promise<void> {
  await invokeCommand('refresh_historical_data', { symbol });
}

export function getRealtimeData(
  search = '',
  column: RealtimeSortColumn = 'change_percent',
  sort: SortDirection = 'desc',
  page = 1,
  pageSize = 50,
): Promise<PagedResponse<RealtimeData>> {
  return invokeCommand('get_realtime_data', {
    search,
    column,
    sort,
    page,
    pageSize,
  });
}

export function getWatchlistSymbols(): Promise<string[]> {
  return invokeCommand('get_watchlist_symbols');
}

export async function setWatchlistMembership(
  symbol: string,
  watched: boolean,
): Promise<void> {
  await invokeCommand(watched ? 'remove_from_watchlist' : 'add_to_watchlist', {
    symbol,
  });
}
