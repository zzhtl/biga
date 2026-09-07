import { describe, expect, test } from 'bun:test';
import {
  actionLabel,
  complianceText,
  fmtMoney,
  fmtRewardRisk,
  fmtShares,
  fmtSignedPercent,
  PLACEHOLDER,
  pnlClass,
  ruleLabel,
  stopBasisLabel,
  todayIso,
} from './discipline_format';

describe('缺失值', () => {
  test('null / undefined / NaN 一律显示占位符，绝不显示 0', () => {
    expect(fmtMoney(null)).toBe(PLACEHOLDER);
    expect(fmtMoney(undefined)).toBe(PLACEHOLDER);
    expect(fmtMoney(Number.NaN)).toBe(PLACEHOLDER);
    expect(fmtSignedPercent(null)).toBe(PLACEHOLDER);
    expect(fmtShares(null)).toBe(PLACEHOLDER);
    expect(fmtRewardRisk(null)).toBe(PLACEHOLDER);
    expect(complianceText(null)).toBe(PLACEHOLDER);
  });
});

describe('fmtSignedPercent', () => {
  test('正数带 + 号，避免和 0 混淆', () => {
    expect(fmtSignedPercent(3.456)).toBe('+3.46%');
    expect(fmtSignedPercent(-8)).toBe('-8.00%');
    expect(fmtSignedPercent(0)).toBe('+0.00%');
  });
});

describe('pnlClass', () => {
  test('A 股口径：涨用 up(红)、跌用 down(绿)、零不着色', () => {
    expect(pnlClass(1)).toBe('up');
    expect(pnlClass(-1)).toBe('down');
    expect(pnlClass(0)).toBe('');
    expect(pnlClass(null)).toBe('');
  });
});

describe('fmtMoney', () => {
  test('负数的符号在千分位之外', () => {
    expect(fmtMoney(-5270)).toBe('-5,270');
    expect(fmtMoney(120000)).toBe('120,000');
  });
});

describe('标签映射', () => {
  test('已知的规则码与枚举转成中文', () => {
    expect(ruleLabel('EXIT_HARD_STOP')).toBe('跌破止损');
    expect(actionLabel('must_sell')).toBe('必须清仓');
    expect(stopBasisLabel('trailing')).toBe('移动止盈');
  });

  test('未收录的规则码原样返回，不编造中文名', () => {
    expect(ruleLabel('EXIT_SOMETHING_NEW')).toBe('EXIT_SOMETHING_NEW');
  });
});

describe('fmtRewardRisk', () => {
  test('按 x : 1 呈现，和门槛口径一致', () => {
    expect(fmtRewardRisk(3)).toBe('3.00 : 1');
  });
});

describe('todayIso', () => {
  test('补零到 YYYY-MM-DD', () => {
    expect(todayIso(new Date(2026, 2, 5))).toBe('2026-03-05');
    expect(todayIso(new Date(2026, 11, 31))).toBe('2026-12-31');
  });
});
