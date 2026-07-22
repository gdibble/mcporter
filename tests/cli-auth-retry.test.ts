import { describe, expect, it, vi } from 'vitest';

process.env.MCPORTER_DISABLE_AUTORUN = '1';
const cliModulePromise = import('../src/cli.js');

const baseDefinition = {
  name: 'adhoc-server',
  command: { kind: 'http' as const, url: new URL('https://example.com/mcp') },
  source: { kind: 'local' as const, path: '<adhoc>' },
};

describe('handleAuth retry logic', () => {
  it('retries once when the first attempt is unauthorized', async () => {
    const { handleAuth } = await cliModulePromise;
    const listTools = vi
      .fn()
      .mockRejectedValueOnce(new Error('SSE error: Non-200 status code (401)'))
      .mockResolvedValueOnce([{ name: 'ok' }]);
    const runtime = {
      registerDefinition: vi.fn(),
      getDefinition: vi.fn().mockReturnValue(baseDefinition),
      listTools,
    } as unknown as Awaited<ReturnType<(typeof import('../src/runtime.js'))['createRuntime']>>;

    await expect(handleAuth(runtime, ['adhoc-server'])).resolves.toBeUndefined();
    expect(listTools).toHaveBeenCalledTimes(2);
  });

  it('throws after the second unauthorized attempt', async () => {
    const { handleAuth } = await cliModulePromise;
    const listTools = vi.fn().mockRejectedValue(new Error('SSE error: Non-200 status code (401)'));
    const runtime = {
      registerDefinition: vi.fn(),
      getDefinition: vi.fn().mockReturnValue(baseDefinition),
      listTools,
    } as unknown as Awaited<ReturnType<(typeof import('../src/runtime.js'))['createRuntime']>>;

    await expect(handleAuth(runtime, ['adhoc-server'])).rejects.toThrow(/Failed to authorize/);
    expect(listTools).toHaveBeenCalledTimes(2);
  });
});
