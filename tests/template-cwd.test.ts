import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { renderTemplate, templateTestHelpers } from '../src/cli/generate/template.js';
import type { CliArtifactMetadata } from '../src/cli-metadata.js';
import type { ServerDefinition } from '../src/config.js';

const { computeRelativeStdioCwd } = templateTestHelpers;

function stdioDef(overrides: { cwd?: string; args?: string[] } = {}): ServerDefinition {
  return {
    name: 'demo',
    command: {
      kind: 'stdio',
      command: 'node',
      args: overrides.args ?? ['dist/index.js'],
      ...(overrides.cwd !== undefined ? { cwd: overrides.cwd } : {}),
    },
  } as ServerDefinition;
}

describe('computeRelativeStdioCwd', () => {
  it('returns null when outputPath is missing', () => {
    expect(computeRelativeStdioCwd(stdioDef(), undefined)).toBeNull();
  });

  it('returns null for HTTP-backed servers', () => {
    const httpDef: ServerDefinition = {
      name: 'demo',
      command: { kind: 'http', url: new URL('https://example.com/mcp') },
    } as ServerDefinition;
    expect(computeRelativeStdioCwd(httpDef, '/pkg/dist/cli.cjs')).toBeNull();
  });

  it('preserves an explicit absolute cwd from the embedded definition', () => {
    expect(computeRelativeStdioCwd(stdioDef({ cwd: '/pkg' }), '/pkg/dist/cli.cjs')).toBeNull();
  });

  it('relativizes ad-hoc stdio cwd against the final artifact directory', () => {
    const rel = computeRelativeStdioCwd(stdioDef(), path.join(process.cwd(), 'dist', 'cli.cjs'));
    expect(rel).toBe('..');
  });

  it('resolves to "." when relative cwd equals the output directory', () => {
    expect(computeRelativeStdioCwd(stdioDef({ cwd: 'dist' }), path.join(process.cwd(), 'dist', 'cli.cjs'))).toBe('.');
  });

  it('resolves relative cwd inputs against process.cwd()', () => {
    const outputPath = '/pkg/dist/cli.cjs';
    const expected = path.relative(path.dirname(outputPath), path.resolve(process.cwd(), 'relative-dir'));
    expect(computeRelativeStdioCwd(stdioDef({ cwd: 'relative-dir' }), outputPath)).toBe(expected);
  });
});

describe('renderTemplate', () => {
  it('rejects sanitized command name collisions before emitting a broken CLI', () => {
    expect(() =>
      renderTemplate({
        runtimeKind: 'node',
        timeoutMs: 30_000,
        definition: stdioDef(),
        serverName: 'demo',
        generator: { name: 'mcporter', version: 'test' },
        metadata: metadataFor('demo'),
        tools: [
          {
            tool: { name: 'foo/bar', inputSchema: undefined, outputSchema: undefined },
            methodName: 'fooSlash',
            options: [],
          },
          {
            tool: { name: 'foo_bar', inputSchema: undefined, outputSchema: undefined },
            methodName: 'fooUnderscore',
            options: [],
          },
        ],
      })
    ).toThrow(/Generated command name collision 'foo-bar'/);
  });

  it('emits strict numeric parsing and empty required-string validation', () => {
    const source = renderTemplate({
      runtimeKind: 'node',
      timeoutMs: 30_000,
      definition: stdioDef(),
      serverName: 'demo',
      generator: { name: 'mcporter', version: 'test' },
      metadata: metadataFor('demo'),
      tools: [
        {
          tool: {
            name: 'sum',
            inputSchema: {
              type: 'object',
              properties: {
                count: { type: 'number' },
                coords: { type: 'array', items: { type: 'number' } },
                name: { type: 'string' },
              },
              required: ['name'],
            },
            outputSchema: undefined,
          },
          methodName: 'sum',
          options: [
            {
              property: 'count',
              cliName: 'count',
              required: false,
              type: 'number',
              placeholder: '<count:number>',
            },
            {
              property: 'coords',
              cliName: 'coords',
              required: false,
              type: 'array',
              arrayItemType: 'number',
              placeholder: '<coords:value1,value2>',
            },
            {
              property: 'name',
              cliName: 'name',
              required: true,
              type: 'string',
              placeholder: '<name>',
            },
          ],
        },
      ],
    });
    expect(source).toContain('parseFiniteNumber');
    expect(source).not.toContain('parseFloat');
    expect(source).toContain("typeof entry.value === 'string' && entry.value.trim() === ''");
  });
});

function metadataFor(serverName: string): CliArtifactMetadata {
  return {
    schemaVersion: 1,
    generatedAt: '1970-01-01T00:00:00.000Z',
    generator: { name: 'mcporter', version: 'test' },
    server: {
      name: serverName,
      definition: {
        name: serverName,
        command: { kind: 'stdio' as const, command: 'node', args: [], cwd: process.cwd() },
      },
    },
    artifact: { path: '', kind: 'template' as const },
    invocation: { runtime: 'node' as const, timeoutMs: 30_000, minify: false },
  };
}
