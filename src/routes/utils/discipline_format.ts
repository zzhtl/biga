/**
 * 交易纪律的展示层格式化。
 *
 * 这里**只做格式化，不做任何阈值判定**。所有"是否触发"的判断都在 Rust 侧，
 * 前端一旦出现 `if (gain > 0.2)` 这类数字，规则就有了两套实现——
 * 半年后必然不一致，而且不一致的方向一定是让人心里舒服的那个。
 */

import type {
  DisciplineAction,
  DisciplineCategory,
  RiskLevel,
  SizingConstraint,
  StopBasis,
} from '../types';

/** 数据缺失的统一占位符（与收藏池一致） */
export const PLACEHOLDER = '—';

export function fmtPrice(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? PLACEHOLDER
    : value.toFixed(digits);
}

export function fmtMoney(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return PLACEHOLDER;
  return `${value < 0 ? '-' : ''}${Math.abs(value).toLocaleString('zh-CN', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`;
}

/** 带正负号的百分比。盈亏、涨跌一律用它，避免"+0.00%"和"0.00%"混用 */
export function fmtSignedPercent(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? PLACEHOLDER
    : `${value >= 0 ? '+' : ''}${value.toFixed(digits)}%`;
}

export function fmtPercent(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? PLACEHOLDER
    : `${value.toFixed(digits)}%`;
}

export function fmtShares(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? PLACEHOLDER
    : `${Math.trunc(value).toLocaleString('zh-CN')} 股`;
}

/** A 股红涨绿跌：正 → --price-up(红)，负 → --price-down(绿) */
export function pnlClass(value: number | null | undefined): '' | 'up' | 'down' {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return '';
  return value > 0 ? 'up' : 'down';
}

const ACTION_LABELS: Record<DisciplineAction, string> = {
  must_sell: '必须清仓',
  must_reduce: '必须减仓',
  blocked: '拒绝买入',
  warn: '提示',
};

const CATEGORY_LABELS: Record<DisciplineCategory, string> = {
  exit: '卖出',
  entry: '买入',
  sizing: '仓位',
  portfolio: '组合',
  data: '数据',
};

const SEVERITY_LABELS: Record<RiskLevel, string> = { low: '低', medium: '中', high: '高' };

const STOP_BASIS_LABELS: Record<StopBasis, string> = {
  fixed: '固定百分比',
  atr: 'ATR 波动',
  support: '支撑位',
  trailing: '移动止盈',
};

const CONSTRAINT_LABELS: Record<SizingConstraint, string> = {
  risk: '单笔风险预算',
  single_position: '单票仓位上限',
  total_position: '总仓位上限',
  cash: '可用现金',
};

/** 规则码 → 中文短名。未收录的码原样返回，不编造。 */
const RULE_LABELS: Record<string, string> = {
  EXIT_HARD_STOP: '跌破止损',
  EXIT_TRAILING_STOP: '移动止盈回撤',
  EXIT_SUPPORT_BREAKDOWN: '放量破位',
  EXIT_TIME_STOP: '时间止损',
  EXIT_SCALE_OUT_1: '分批止盈·第一档',
  EXIT_SCALE_OUT_2: '分批止盈·第二档',
  EXIT_MA20_BREAK: '盈利仓跌破 MA20',
  EXIT_STOP_SHOULD_RISE: '止损上移',
  EXIT_LIMIT_DOWN_UNFILLABLE: '一字跌停不可成交',
  EXIT_T1_LOCKED: 'T+1 锁定',
  ENTRY_SETTINGS_MISSING: '未设置账户资金',
  ENTRY_STOP_REQUIRED: '未填止损价',
  ENTRY_TARGET_REQUIRED: '未填目标价',
  ENTRY_RR_TOO_LOW: '盈亏比不达标',
  ENTRY_STOP_TOO_WIDE: '止损过宽',
  ENTRY_RISK_LEVEL: '风险等级超限',
  ENTRY_AVERAGING_DOWN: '浮亏加仓',
  ENTRY_COOLDOWN: '止损冷静期',
  ENTRY_CIRCUIT_BREAKER: '连亏熔断',
  ENTRY_MAX_HOLDINGS: '持股只数超限',
  ENTRY_RISK_BUDGET: '超出单笔风险预算',
  ENTRY_POSITION_LIMIT: '超出单票上限',
  ENTRY_TOTAL_EXPOSURE: '超出总仓上限',
  ENTRY_INSUFFICIENT_CASH: '现金不足',
  ENTRY_SIZE_BELOW_LOT: '买不到一手',
  ENTRY_LIMIT_UP_UNFILLABLE: '一字涨停难成交',
  ENTRY_DATA_STALE: '行情陈旧',
  DATA_PRICE_BASIS_CHANGED: '价格基准变化',
  DATA_TARGET_MISSING: '未登记目标价',
  DATA_STALE_QUOTE: '行情陈旧',
  DATA_INSUFFICIENT_WINDOW: 'K 线窗口不足',
};

export function actionLabel(action: DisciplineAction): string {
  return ACTION_LABELS[action] ?? action;
}

export function categoryLabel(category: DisciplineCategory): string {
  return CATEGORY_LABELS[category] ?? category;
}

export function severityLabel(severity: RiskLevel): string {
  return SEVERITY_LABELS[severity] ?? severity;
}

export function stopBasisLabel(basis: StopBasis): string {
  return STOP_BASIS_LABELS[basis] ?? basis;
}

export function constraintLabel(constraint: SizingConstraint): string {
  return CONSTRAINT_LABELS[constraint] ?? constraint;
}

export function ruleLabel(code: string): string {
  return RULE_LABELS[code] ?? code;
}

export function complianceText(rate: number | null | undefined): string {
  return rate === null || rate === undefined || !Number.isFinite(rate)
    ? PLACEHOLDER
    : `${(rate * 100).toFixed(0)}%`;
}

/** 盈亏比：后端算不出时（缺止损或目标价）为 null */
export function fmtRewardRisk(ratio: number | null | undefined): string {
  return ratio === null || ratio === undefined || !Number.isFinite(ratio)
    ? PLACEHOLDER
    : `${ratio.toFixed(2)} : 1`;
}

/** 今天的 YYYY-MM-DD，用作成交日输入框默认值 */
export function todayIso(now = new Date()): string {
  const pad = (value: number) => String(value).padStart(2, '0');
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}
