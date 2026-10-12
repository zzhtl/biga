import { expect, test, type Page } from '@playwright/test';

async function mockDesktop(page: Page, options: { empty?: boolean; running?: boolean; failure?: boolean } = {}) {
  await page.addInitScript((options) => {
    const calls: Array<{ command: string; args: any }> = [];
    const callbacks = new Map<number, (value: any) => void>();
    let listener: number | undefined;
    let connections = options.empty ? [] : [{ id: 'local', name: '本地 Codex', provider: 'codex', transport: 'local', executable: 'codex', model: null, host: null, user: null, port: null, identity_file: null }];
    let jobs: any[] = [];
    const report = (symbol: string) => ({
      connection_name: '本地 Codex', provider: 'codex', model: '测试模型', generated_at: '2026-07-17T10:00:00Z',
      snapshot: {
        symbol, as_of: '2026-07-17',
        bars: Array.from({ length: 60 }, (_, i) => ({ date: new Date(Date.UTC(2026, 4, 19 + i)).toISOString().slice(0, 10), close: 10 + i * .1, open: 10 + i * .1 - .05, high: 11 + i * .1, low: 9 + i * .1, volume: 1000 + i * 10 })),
        valuation: null, fundamentals: [], technical: {},
        sources: [{ resource: 'history', fetched_at: 1784289600, stale: false, warning: null }], warnings: ['缺少基本面数据'],
      },
      assessment: { summary: `${symbol}：价格温和上升，成交量提供支持。`, direction: 'up', factors: [{ text: '历史收盘价走高', sources: ['history'] }], bullish: '放量突破近期高点', base: '维持当前趋势', bearish: '跌破支撑后回落', risks: ['量能不足'], invalidation: ['跌破近期低点'], forecast: Array.from({ length: 5 }, (_, i) => ({ day: i + 1, lower: 14 - i * .1, expected: 16 + i * .1, upper: 18 + i * .1 })) },
      dates: ['2026-07-20', '2026-07-21', '2026-07-22', '2026-07-23', '2026-07-24'], usage: [{ input_tokens: 123 }],
    });
    const complete = (job: any) => {
      job.status = options.failure ? 'failed' : 'completed';
      job.stage = options.failure ? '预测失败，未自动重试' : '报告已完成';
      job.error = options.failure ? 'AI 额度不足或被限流，未自动重试' : null;
      job.report_json = options.failure ? null : JSON.stringify(report(job.symbol));
      if (listener !== undefined) callbacks.get(listener)?.({ payload: { ...job } });
    };
    (window as any).__aiTest = { calls, complete: () => jobs.forEach(complete) };
    (window as any).__TAURI_INTERNALS__ = {
      transformCallback: (cb: (value: unknown) => void) => { const id = callbacks.size + 1; callbacks.set(id, cb); return id; },
      unregisterCallback: (id: number) => callbacks.delete(id),
      invoke: async (command: string, args: any = {}) => {
        calls.push({ command, args });
        switch (command) {
          case 'get_discipline_positions': return { positions: [], pending_events: [] };
          case 'get_stock_infos': return [{ symbol: '600519', name: '测试股票', exchange: 'sh' }, { symbol: '000001', name: '另一股票', exchange: 'sz' }];
          case 'get_api_token_status': return { configured: false, source: 'none', masked: null };
          case 'list_ai_connections': return structuredClone(connections);
          case 'save_ai_connection': {
            const config = { ...args.config, id: args.config.id || 'added' };
            connections = [...connections.filter(c => c.id !== config.id), config];
            return config;
          }
          case 'delete_ai_connection': connections = connections.filter(c => c.id !== args.id); return;
          case 'check_ai_connection': return 'CLI 与连接检查通过；未消耗模型额度，认证在生成报告时验证';
          case 'list_ai_predictions': return jobs.filter(j => !args.symbol || args.symbol === j.symbol).map(j => ({ ...j, report_json: null }));
          case 'start_ai_prediction': {
            const job = { id: `job-${jobs.length}`, symbol: args.request.symbol, connection_id: 'local', status: 'running', stage: 'AI 分析中', created_at: '2026-07-17T10:00:00Z', error: null, requests: 0, cache_hits: 3, report_json: null };
            jobs.push(job);
            if (!options.running) complete(job);
            return { ...job };
          }
          case 'get_ai_prediction': return { ...jobs.find(j => j.id === args.id) };
          case 'cancel_ai_prediction': {
            const job = jobs.find(j => j.id === args.id); job.status = 'cancelled'; job.stage = '任务已取消';
            if (listener !== undefined) callbacks.get(listener)?.({ payload: { ...job } }); return;
          }
          case 'refresh_historical_data': return { requests: 0, cache_hits: 3, warnings: [] };
          case 'plugin:event|listen': listener = args.handler; return 1;
          case 'plugin:event|unlisten': listener = undefined; return;
          default: throw new Error(`Unexpected command: ${command}`);
        }
      },
    };
  }, options);
}

test('connection onboarding and checks never generate predictions', async ({ page }) => {
  await mockDesktop(page, { empty: true });
  await page.goto('/');
  await expect(page.getByText('连接你的 AI，开始第一份分析')).toBeVisible();
  await page.getByRole('button', { name: '添加 AI 连接' }).click();
  await page.getByLabel('连接名称').fill('远程 Claude');
  await page.getByLabel('AI 提供方').selectOption('claude');
  await page.getByLabel('运行位置').selectOption('ssh');
  await page.getByLabel('主机 / SSH 别名').fill('research-server');
  await page.getByRole('button', { name: '保存连接' }).click();
  await expect(page.getByText('连接已保存，未调用 AI')).toBeVisible();
  await page.getByRole('button', { name: '检查连接' }).click();
  await expect(page.getByText('CLI 与连接检查通过', { exact: false })).toBeVisible();
  expect(await page.evaluate(() => (window as any).__aiTest.calls.filter((c: any) => c.command === 'start_ai_prediction' || c.command === 'refresh_historical_data').length)).toBe(0);
});

test('report, charts and history agree without duplicate generation', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await mockDesktop(page);
  await page.goto('/');
  await page.getByLabel('股票代码', { exact: true }).fill('600519');
  await page.getByRole('button', { name: '生成 AI 预测', exact: true }).click();
  await expect(page.getByText('600519：价格温和上升，成交量提供支持。')).toBeVisible();
  await expect(page.getByRole('img', { name: '最近 60 日实际成交量' })).toBeVisible();
  await page.getByText('逐日预测与数据来源', { exact: true }).click();
  await expect(page.getByRole('cell', { name: '2026-07-24', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: '16.40', exact: true })).toBeVisible();
  await page.locator('.history-list button').first().click();
  expect(await page.evaluate(() => (window as any).__aiTest.calls.filter((c: any) => c.command === 'start_ai_prediction').length)).toBe(1);
  await page.setViewportSize({ width: 900, height: 900 });
  await expect(page.locator('.price-chart canvas')).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.setViewportSize({ width: 600, height: 900 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('cancel and late results cannot replace another stock', async ({ page }) => {
  await mockDesktop(page, { running: true });
  await page.goto('/');
  await page.getByLabel('股票代码', { exact: true }).fill('600519');
  await page.getByRole('button', { name: '生成 AI 预测', exact: true }).click();
  await expect(page.getByRole('button', { name: '取消任务' })).toBeVisible();
  await expect(page.getByRole('button', { name: '生成 AI 预测', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '取消任务' }).click();
  await expect(page.getByText('任务已取消', { exact: true })).toBeVisible();
  await page.getByLabel('股票代码', { exact: true }).fill('000001');
  await page.evaluate(() => (window as any).__aiTest.complete());
  await expect(page.getByText('600519：价格温和上升，成交量提供支持。')).not.toBeVisible();
});

test('provider failure displays actionable error and never renders fabricated charts', async ({ page }) => {
  await mockDesktop(page, { failure: true });
  await page.goto('/');
  await page.getByLabel('股票代码', { exact: true }).fill('600519');
  await page.getByRole('button', { name: '生成 AI 预测', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('AI 额度不足');
  await expect(page.locator('.price-chart')).toHaveCount(0);
  expect(await page.evaluate(() => (window as any).__aiTest.calls.filter((c: any) => c.command === 'start_ai_prediction').length)).toBe(1);
});
