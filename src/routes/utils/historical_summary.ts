import type { HistoricalData } from '../types';

export interface HistoricalSummary {
  startDate: string;
  endDate: string;
  tradingDays: number;
  rangeChange: number | null;
  rangeChangePercent: number | null;
  highestPrice: number;
  lowestPrice: number;
  averageDailyChangePercent: number;
  upDays: number;
  downDays: number;
  flatDays: number;
}

export function calculateHistoricalSummary(
  data: readonly HistoricalData[],
): HistoricalSummary | null {
  const firstItem = data[0];
  if (!firstItem) return null;

  let first = firstItem;
  let last = firstItem;
  let highestPrice = firstItem.high;
  let lowestPrice = firstItem.low;
  let dailyChangePercentTotal = 0;
  let upDays = 0;
  let downDays = 0;
  let flatDays = 0;

  for (const item of data) {
    if (item.date < first.date) first = item;
    if (item.date > last.date) last = item;
    highestPrice = Math.max(highestPrice, item.high);
    lowestPrice = Math.min(lowestPrice, item.low);
    dailyChangePercentTotal += item.change_percent;

    if (item.change_percent > 0) upDays += 1;
    else if (item.change_percent < 0) downDays += 1;
    else flatDays += 1;
  }

  const previousClose = first.close - first.change;
  const rangeChange = previousClose > 0 ? last.close - previousClose : null;

  return {
    startDate: first.date,
    endDate: last.date,
    tradingDays: data.length,
    rangeChange,
    rangeChangePercent:
      rangeChange === null ? null : (rangeChange / previousClose) * 100,
    highestPrice,
    lowestPrice,
    averageDailyChangePercent: dailyChangePercentTotal / data.length,
    upDays,
    downDays,
    flatDays,
  };
}
