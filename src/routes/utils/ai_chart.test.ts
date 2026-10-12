import { describe, expect, test } from 'bun:test';
import type { AiJob, AiReport } from '../types/ai';
import { acceptAiJob, canonicalAiSymbol, isActiveJob, reportText } from '../types/ai';
import { aiChartOptions, volumeChartOptions } from './ai_chart';

describe('AI report presentation', () => {
  test('narrative uses chart labels without changing prices or parts of other words', () => {
    expect(reportText('expected为41.50元，lower为40.20元，day按交易日排列')).toBe('基准价格为41.50元，情景下界为40.20元，交易日序号按交易日排列');
    expect(reportText('holiday forecast')).toBe('holiday 逐日预测');
  });
  test('stock switches and out-of-order tasks cannot overwrite the selected report', () => {
    const incoming = { id: 'old', symbol: '600519', status: 'completed' } as AiJob;
    expect(acceptAiJob('new', '600519', incoming)).toBe(false);
    expect(acceptAiJob('old', '000001', incoming)).toBe(false);
    expect(acceptAiJob('old', 'sh600519', incoming)).toBe(true);
    expect(canonicalAiSymbol('600519.SH')).toBe('600519');
    expect(canonicalAiSymbol('600519; echo')).toBe('');
    expect(isActiveJob(incoming)).toBe(false);
    expect(isActiveJob({ ...incoming, status: 'preparing' })).toBe(true);
  });
  test('chart uses report prices unchanged and never draws forecasts as historical prices', () => {
    const report = {
      snapshot: { as_of: '2026-07-17', bars: [{ date: '2026-07-16', close: 10, open: 9, volume: 100 }, { date: '2026-07-17', close: 11, open: 12, volume: 200 }] },
      dates: ['2026-07-20', '2026-07-21'],
      assessment: { forecast: [{ day: 1, lower: 9, expected: 12, upper: 15 }, { day: 2, lower: 8, expected: 13, upper: 16 }] },
    } as AiReport;
    const options = aiChartOptions(report);
    const series = options.series as Array<{ data: unknown[] }>;
    expect(series[0].data).toEqual([10, 11, null, null]);
    expect(series[1].data).toEqual([null, 11, 12, 13]);
    expect(series[2].data).toEqual([null, 11, 9, 8]);
    expect(series[3].data).toEqual([null, 11, 15, 16]);
    expect(series[5].data).toEqual([null, 0, 6, 8]);
    expect(JSON.stringify(options)).not.toContain('80%');
    expect((volumeChartOptions(report).series as Array<{ data: Array<{ value: number }> }>)[0].data.map(p => p.value)).toEqual([100, 200]);
  });
});
