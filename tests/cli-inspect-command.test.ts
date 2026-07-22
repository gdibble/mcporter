import { describe, expect, it } from 'vitest';
import { __test as inspectInternals } from '../src/cli/inspect-cli-command.js';

describe('inspect-cli flag parsing', () => {
  it('supports --json shorthand', () => {
    const args = ['--json', '/tmp/artifact'];
    const parsed = inspectInternals.parseInspectFlags(args);
    expect(parsed.format).toBe('json');
    expect(parsed.artifactPath).toBe('/tmp/artifact');
  });

  it('validates explicit format values', () => {
    expect(() => inspectInternals.parseInspectFlags(['--format', 'xml', 'artifact'])).toThrow(/format/);
  });

  it('rejects extra positional arguments', () => {
    expect(() => inspectInternals.parseInspectFlags(['artifact', 'shadow'])).toThrow(
      /Unexpected inspect-cli argument 'shadow'/
    );
  });
});
