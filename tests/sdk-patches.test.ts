import { EventEmitter } from 'node:events';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { evaluateStdioLogPolicy, type StdioLogMode } from '../src/sdk-patches.js';

function evaluate(mode: StdioLogMode, hasStderr: boolean, exitCode: number | null) {
  return evaluateStdioLogPolicy(mode, hasStderr, exitCode);
}

describe('sdk-patches STDIO log policy', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('prints logs in auto mode only when stderr exists and exit code is non-zero', () => {
    expect(evaluate('auto', true, 1)).toBe(true);
    expect(evaluate('auto', true, 0)).toBe(false);
    expect(evaluate('auto', false, 1)).toBe(false);
  });

  it('always prints when mode is forced to always', () => {
    expect(evaluate('always', true, 0)).toBe(true);
    expect(evaluate('always', true, null)).toBe(true);
  });

  it('never prints when mode is silent', () => {
    expect(evaluate('silent', true, 2)).toBe(false);
    expect(evaluate('silent', true, null)).toBe(false);
  });

  it('escalates from SIGTERM to SIGKILL when child close waits time out', async () => {
    vi.useFakeTimers();
    const child = new EventEmitter() as EventEmitter & {
      exitCode: number | null;
      kill: ReturnType<typeof vi.fn>;
      stderr: EventEmitter;
      stdin: EventEmitter;
      stdout: EventEmitter;
      stdio: EventEmitter[];
      unref: ReturnType<typeof vi.fn>;
    };
    child.exitCode = null;
    child.kill = vi.fn();
    child.unref = vi.fn();
    child.stdin = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.stdio = [child.stdin, child.stdout, child.stderr];
    for (const stream of child.stdio) {
      Object.assign(stream, {
        destroy: vi.fn(),
        end: vi.fn(),
        unref: vi.fn(),
      });
    }
    const transport = Object.create(StdioClientTransport.prototype) as {
      _abortController: AbortController | null;
      _process: typeof child | null;
      _readBuffer: { clear: ReturnType<typeof vi.fn> } | null;
      _stderrStream: null;
      onclose: ReturnType<typeof vi.fn>;
    };
    transport._process = child;
    transport._stderrStream = null;
    transport._abortController = { abort: vi.fn() } as unknown as AbortController;
    transport._readBuffer = { clear: vi.fn() };
    transport.onclose = vi.fn();

    const close = StdioClientTransport.prototype.close.call(transport as unknown as StdioClientTransport);
    await vi.advanceTimersByTimeAsync(700);
    expect(child.kill).toHaveBeenCalledWith('SIGTERM');

    await vi.advanceTimersByTimeAsync(700);
    expect(child.kill).toHaveBeenCalledWith('SIGKILL');

    await vi.advanceTimersByTimeAsync(500);
    await close;
    expect(transport.onclose).toHaveBeenCalled();
  });
});
