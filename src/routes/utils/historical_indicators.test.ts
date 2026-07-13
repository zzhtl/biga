import { describe, expect, test } from 'bun:test';
import { calculateEma, calculateKdj, calculateMacd } from './historical_indicators';

describe('historical indicators', () => {
  test('returns no EMA for invalid input', () => {
    expect(calculateEma([], 12)).toEqual([]);
    expect(calculateEma([1, 2], 0)).toEqual([]);
  });

  test('keeps indicator series aligned with prices', () => {
    const closes = Array.from({ length: 30 }, (_, index) => 10 + index * 0.1);
    const macd = calculateMacd(closes);
    expect(macd.macd).toHaveLength(closes.length);
    expect(macd.signal).toHaveLength(closes.length);
    expect(macd.histogram).toHaveLength(closes.length);
  });

  test('handles a flat KDJ range without NaN values', () => {
    const values = [10, 10, 10, 10];
    const kdj = calculateKdj(values, values, values);
    expect(kdj.k).toHaveLength(values.length);
    expect([...kdj.k, ...kdj.d, ...kdj.j].every(Number.isFinite)).toBe(true);
  });
});
