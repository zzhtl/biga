import type { HistoricalData } from './index';

export interface AiConnectionConfig {
  id: string;
  name: string;
  provider: 'claude' | 'codex';
  transport: 'local' | 'ssh';
  executable: string;
  model: string | null;
  host: string | null;
  user: string | null;
  port: number | null;
  identity_file: string | null;
}

export interface AiReport {
  connection_name: string;
  provider: 'claude' | 'codex';
  model: string;
  generated_at: string;
  snapshot: {
    symbol: string;
    as_of: string;
    bars: HistoricalData[];
    valuation: Record<string, number | null> | null;
    fundamentals: Array<Record<string, unknown>>;
    technical: Record<string, unknown>;
    sources: Array<{ resource: string; fetched_at: number; stale: boolean; warning: string | null }>;
    warnings: string[];
  };
  assessment: {
    summary: string;
    direction: 'up' | 'down' | 'sideways';
    factors: Array<{ text: string; sources: string[] }>;
    bullish: string;
    base: string;
    bearish: string;
    risks: string[];
    invalidation: string[];
    forecast: Array<{ day: number; lower: number; expected: number; upper: number }>;
  };
  dates: string[];
  usage: unknown[];
}

export interface AiJob {
  id: string;
  symbol: string;
  connection_id: string;
  status: 'preparing' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  stage: string;
  created_at: string;
  error: string | null;
  requests: number;
  cache_hits: number;
  report_json: string | null;
}

export const isActiveJob = (job: AiJob | null): boolean => job?.status === 'preparing' || job?.status === 'running';
export const sourceLabels: Record<string, string> = { history: '历史行情', valuation: '股本估值', fundamentals: '财务指标', technical: '技术分析' };

/** 将模型偶尔解释的协议字段转为读者看到的图表名称。 */
export function reportText(text: string): string {
  const labels: Record<string, string> = { forecast: '逐日预测', day: '交易日序号', expected: '基准价格', lower: '情景下界', upper: '情景上界' };
  return text.replace(/\b(forecast|day|expected|lower|upper)\b/g, word => labels[word]);
}

export function canonicalAiSymbol(value: string): string {
  const match = value.trim().match(/^(?:sh|sz|bj)?(\d{6})(?:\.(?:sh|sz|bj))?$/i);
  return match?.[1] ?? '';
}

export function acceptAiJob(currentId: string | null, symbol: string, incoming: AiJob): boolean {
  return currentId === incoming.id && canonicalAiSymbol(symbol) === incoming.symbol;
}
