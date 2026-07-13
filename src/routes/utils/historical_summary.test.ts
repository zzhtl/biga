import { describe, expect, test } from 'bun:test';
import type { HistoricalData } from '../types';
import { calculateHistoricalSummary } from './historical_summary';

function history(overrides: Partial<HistoricalData>): HistoricalData {
  return {
    symbol: '600000',
    date: '2026-01-01',
    open: 100,
    close: 100,
    high: 100,
    low: 100,
    volume: 1_000,
    amount: 100_000,
    amplitude: 0,
    turnover_rate: 1,
    volume_ratio: 1,
    change_percent: 0,
    change: 0,
    ...overrides,
  };
}

describe('calculateHistoricalSummary', () => {
  test('returns null when the queried range has no trading data', () => {
    expect(calculateHistoricalSummary([])).toBeNull();
  });

  test('calculates range statistics in date order without mutating the response', () => {
    const data = [
      history({
        date: '2026-01-03',
        open: 108,
        close: 105,
        high: 112,
        low: 104,
        change_percent: -1,
      }),
      history({
        date: '2026-01-01',
        open: 100,
        close: 102,
        high: 103,
        low: 99,
        change: 2,
        change_percent: 2,
      }),
      history({
        date: '2026-01-02',
        open: 102,
        close: 108,
        high: 109,
        low: 101,
        change_percent: 5,
      }),
      history({
        date: '2026-01-04',
        open: 105,
        close: 105,
        high: 106,
        low: 103,
        change_percent: 0,
      }),
    ];
    const originalDates = data.map((item) => item.date);

    const summary = calculateHistoricalSummary(data);

    expect(summary).toEqual({
      startDate: '2026-01-01',
      endDate: '2026-01-04',
      tradingDays: 4,
      rangeChange: 5,
      rangeChangePercent: 5,
      highestPrice: 112,
      lowestPrice: 99,
      averageDailyChangePercent: 1.5,
      upDays: 2,
      downDays: 1,
      flatDays: 1,
    });
    expect(data.map((item) => item.date)).toEqual(originalDates);
  });

  test('includes the first trading day change when its open has a price gap', () => {
    const summary = calculateHistoricalSummary([
      history({
        open: 20,
        close: 21,
        high: 22,
        low: 19,
        change: 0.5,
        change_percent: 2.4390243902439024,
      }),
    ]);

    expect(summary?.rangeChange).toBe(0.5);
    expect(summary?.rangeChangePercent).toBeCloseTo(2.4390243902439024);
  });

  test('does not report a range return when the previous close is not a valid baseline', () => {
    const summary = calculateHistoricalSummary([
      history({ open: 1, close: 1, high: 1, low: 1, change: 1 }),
    ]);

    expect(summary?.rangeChange).toBeNull();
    expect(summary?.rangeChangePercent).toBeNull();
  });
});
