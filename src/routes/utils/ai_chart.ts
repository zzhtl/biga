import type { EChartsOption } from 'echarts';
import type { AiReport } from '../types/ai';

export function aiChartOptions(report: AiReport): EChartsOption {
  const bars = report.snapshot.bars.slice(-90);
  const forecast = report.assessment.forecast;
  const labels = [...bars.map(b => b.date), ...report.dates];
  const prefix = Array(Math.max(0, bars.length - 1)).fill(null);
  const anchor = bars.at(-1)?.close ?? null;
  const lower = [...prefix, anchor, ...forecast.map(p => p.lower)];
  const expected = [...prefix, anchor, ...forecast.map(p => p.expected)];
  const upper = [...prefix, anchor, ...forecast.map(p => p.upper)];
  return {
    backgroundColor: 'transparent',
    color: ['#94a3b8', '#31b5c8', '#e8ae4a', '#e8ae4a'],
    legend: { textStyle: { color: '#aeb7c2' }, data: ['历史收盘', 'AI 基准情景', '情景下界', '情景上界'] },
    tooltip: { trigger: 'axis', renderMode: 'richText', valueFormatter: value => typeof value === 'number' ? value.toFixed(2) : '—' },
    grid: { left: 60, right: 25, top: 50, bottom: 75 },
    xAxis: { type: 'category', data: labels, boundaryGap: false, axisLabel: { color: '#aeb7c2' } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#aeb7c2' }, splitLine: { lineStyle: { color: '#343a42' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 20, bottom: 10 }],
    series: [
      { name: '历史收盘', type: 'line', symbol: 'none', data: [...bars.map(b => b.close), ...forecast.map(() => null)] },
      { name: 'AI 基准情景', type: 'line', symbol: 'circle', symbolSize: 4, data: expected, lineStyle: { width: 2.5 },
        markLine: { symbol: 'none', label: { formatter: '预测起点', color: '#aeb7c2' }, data: [{ xAxis: report.snapshot.as_of }] } },
      { name: '情景下界', type: 'line', symbol: 'none', data: lower, lineStyle: { type: 'dashed', width: 1 } },
      { name: '情景上界', type: 'line', symbol: 'none', data: upper, lineStyle: { type: 'dashed', width: 1 },
        markArea: { silent: true, itemStyle: { color: 'rgba(49,181,200,.04)' }, data: [[{ xAxis: report.snapshot.as_of }, { xAxis: report.dates.at(-1) }]] } },
      { name: '区间基线', type: 'line', stack: 'scenario', data: lower, symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false } },
      { name: '情景范围', type: 'line', stack: 'scenario', data: [...prefix, 0, ...forecast.map(p => p.upper - p.lower)], symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(232,174,74,.13)' }, tooltip: { show: false } },
    ],
  };
}

export function volumeChartOptions(report: AiReport): EChartsOption {
  const bars = report.snapshot.bars.slice(-60);
  return {
    tooltip: { trigger: 'axis', renderMode: 'richText' },
    grid: { left: 65, right: 25, top: 15, bottom: 30 },
    xAxis: { type: 'category', data: bars.map(b => b.date), axisLabel: { color: '#aeb7c2' } },
    yAxis: { type: 'value', axisLabel: { color: '#aeb7c2' }, splitLine: { lineStyle: { color: '#343a42' } } },
    series: [{ name: '成交量', type: 'bar', data: bars.map(b => ({ value: b.volume, itemStyle: { color: b.close >= b.open ? '#f15b64' : '#35c889' } })) }],
  };
}
