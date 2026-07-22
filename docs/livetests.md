---
summary: 'Opt-in live MCP integration tests that hit real hosted servers (off by default in CI).'
read_when:
  - 'Running end-to-end validation against hosted MCP servers'
---

# Live MCP Tests

These tests hit real hosted MCP servers and require outbound HTTP. They are **off by default** to keep CI and local runs deterministic.

## When to run

- Before releases when you want end-to-end validation against hosted servers.
- When debugging regressions that only repro against real servers (e.g., DeepWiki).

## How to run

```bash
MCP_LIVE_TESTS=1 pnpm test:live
```

This runs the Vitest suite under `tests/live`, in-band, with longer timeouts.

## Current coverage

- **DeepWiki**:
  - Streamable HTTP success path: `https://mcp.deepwiki.com/mcp`
  - Deprecated SSE endpoint classification: `https://mcp.deepwiki.com/sse`
  - Tests:
    - call `read_wiki_structure repoName:facebook/react` and assert a non-empty result over Streamable HTTP
    - assert the legacy SSE endpoint currently returns a structured HTTP `410` issue envelope

## Notes

- Tests are skipped entirely unless `MCP_LIVE_TESTS=1` is set.
- Ensure network egress is allowed. No secrets are required for the current DeepWiki checks.
- As of 2026-03-29, DeepWiki's hosted `/sse` endpoint responds with HTTP `410`, so the live suite treats that as a compatibility/error-classification smoke rather than a success-path transport check.
- Keep assertions minimal to reduce flake; these are availability smokes, not full contract tests.
