import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: '**/*.e2e.ts',
  timeout: 30000,
  use: {
    baseURL: 'http://127.0.0.1:5173',
    viewport: { width: 1440, height: 1080 },
    launchOptions: process.env.BIGA_CHROME_PATH ? { executablePath: process.env.BIGA_CHROME_PATH } : {},
    screenshot: 'only-on-failure',
  },
  webServer: { command: 'bun run dev --host 127.0.0.1', url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI },
});
