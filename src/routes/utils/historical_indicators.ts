export interface MacdSeries {
  macd: number[];
  signal: number[];
  histogram: number[];
}

export interface KdjSeries {
  k: number[];
  d: number[];
  j: number[];
}

export function calculateEma(data: number[], period: number): number[] {
  if (data.length === 0 || period <= 0) return [];
  const multiplier = 2 / (period + 1);
  const ema = [data[0]];
  for (let index = 1; index < data.length; index += 1) {
    ema[index] = (data[index] - ema[index - 1]) * multiplier + ema[index - 1];
  }
  return ema;
}

export function calculateMacd(closes: number[]): MacdSeries {
  const ema12 = calculateEma(closes, 12);
  const ema26 = calculateEma(closes, 26);
  const macd = ema12.map((value, index) => value - ema26[index]);
  const signal = calculateEma(macd, 9);
  return {
    macd,
    signal,
    histogram: macd.map((value, index) => value - signal[index]),
  };
}

export function calculateKdj(
  highs: number[],
  lows: number[],
  closes: number[],
  period = 9,
): KdjSeries {
  const k: number[] = [];
  const d: number[] = [];
  const j: number[] = [];
  let previousK = 50;
  let previousD = 50;

  for (let index = 0; index < closes.length; index += 1) {
    const start = Math.max(0, index - period + 1);
    const highest = Math.max(...highs.slice(start, index + 1));
    const lowest = Math.min(...lows.slice(start, index + 1));
    const rsv = highest === lowest
      ? 0
      : ((closes[index] - lowest) / (highest - lowest)) * 100;
    const currentK = (2 * previousK + rsv) / 3;
    const currentD = (2 * previousD + currentK) / 3;
    k.push(currentK);
    d.push(currentD);
    j.push(3 * currentK - 2 * currentD);
    previousK = currentK;
    previousD = currentD;
  }

  return { k, d, j };
}
