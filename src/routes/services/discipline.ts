/**
 * 交易纪律服务层
 *
 * 全部判定都在 Rust 侧。前端不做任何阈值比较——两套实现必然漂移，
 * 而且漂移方向一定是让人心里舒服的那个。
 */

import { invokeCommand } from './core';
import type {
  AccountView,
  DisciplineBoard,
  DisciplineReview,
  DisciplineRules,
  EntryVerdict,
  EventResolution,
  PositionRow,
  RiskLevel,
} from '../types';

export function getDisciplineAccount(): Promise<AccountView> {
  return invokeCommand<AccountView>('get_discipline_account');
}

export function saveDisciplineAccount(cash: number, rules?: DisciplineRules): Promise<AccountView> {
  return invokeCommand<AccountView>('save_discipline_account', { cash, rules });
}

/** 扫描全部持仓：重算止损、回写持仓最高价、落待处理裁决 */
export function getDisciplinePositions(): Promise<DisciplineBoard> {
  return invokeCommand<DisciplineBoard>('get_discipline_positions');
}

export function checkBuyDiscipline(params: {
  symbol: string;
  buyPrice: number;
  stopPrice?: number | null;
  targetPrice?: number | null;
  quantity?: number | null;
  riskLevel?: RiskLevel | null;
}): Promise<EntryVerdict> {
  return invokeCommand<EntryVerdict>('check_buy_discipline', params);
}

export function openDisciplinePosition(params: {
  symbol: string;
  price: number;
  quantity: number;
  stopPrice: number;
  targetPrice: number;
  tradeDate: string;
  fee?: number;
  overrideReason?: string;
}): Promise<PositionRow> {
  return invokeCommand<PositionRow>('open_discipline_position', params);
}

export function recordDisciplineTrade(params: {
  positionId: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  tradeDate: string;
  fee?: number;
  ruleCode?: string | null;
  eventId?: string | null;
}): Promise<PositionRow> {
  return invokeCommand<PositionRow>('record_discipline_trade', params);
}

export async function resolveDisciplineEvent(
  eventId: string,
  resolution: Extract<EventResolution, 'complied' | 'violated'>,
  reason?: string,
): Promise<void> {
  await invokeCommand('resolve_discipline_event', { eventId, resolution, reason });
}

export function getDisciplineReview(): Promise<DisciplineReview> {
  return invokeCommand<DisciplineReview>('get_discipline_review');
}
