import { describe, expect, it } from 'vitest';
import { daemonIdleWatcherInterval, shouldShutdownDaemonForIdle } from '../src/daemon/request-utils.js';

describe('daemon idle helpers', () => {
  it('detects global daemon idle timeout', () => {
    expect(shouldShutdownDaemonForIdle(1_000, 4_000, 3_000)).toBe(true);
    expect(shouldShutdownDaemonForIdle(1_000, 3_999, 3_000)).toBe(false);
    expect(shouldShutdownDaemonForIdle(1_000, 10_000, undefined)).toBe(false);
    expect(shouldShutdownDaemonForIdle(1_000, 10_000, 3_000, 1)).toBe(false);
  });

  it('uses short watcher intervals for short configured timeouts', () => {
    expect(daemonIdleWatcherInterval(undefined)).toBe(30_000);
    expect(daemonIdleWatcherInterval(150)).toBe(100);
    expect(daemonIdleWatcherInterval(2_000)).toBe(1_000);
    expect(daemonIdleWatcherInterval(120_000)).toBe(30_000);
  });
});
