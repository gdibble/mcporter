# MCPorter Vision

MCPorter should make any commonly supported MCP server usable from TypeScript, scripts, generated CLIs, and agent workflows with minimal setup.

## Product Goal

If an MCP server works in common MCP clients, it should be practical to use through MCPorter too. That includes local stdio servers, hosted HTTP/SSE servers, OAuth-protected providers, imported client configs, and ad-hoc endpoints.

MCPorter should stay small enough to understand, reliable enough for automation, and clear enough that failures tell the user what to fix next.

## What Good Work Looks Like

- Compatibility fixes for commonly used MCP servers, transports, schemas, auth flows, and client config formats.
- Bug fixes with a clear reproduction, root cause, and verification path.
- Performance work that improves startup, listing, calling, generated CLIs, daemon behavior, or repeated tool use without adding much complexity.
- Small UI/UX improvements to CLI output, errors, help text, docs, and generated artifacts.
- Refactors that make the correct fix cleaner, easier to test, or easier to maintain.
- Tests and live/manual verification for behavior that can realistically be exercised.

## Non-Goals

- Localization.
- Major new product areas that are not about making MCP servers easier to discover, call, generate, type, host, or debug through MCPorter.
- Broad features that make the product harder to reason about without a strong compatibility or reliability payoff.
- Complex provider-specific flows when a small generic MCP/auth/transport improvement would solve the same class of problem.
- Cosmetic churn, large rewrites, or dependency/tooling swaps without a concrete user-facing benefit.

## Triage Rule

Autonomous work is appropriate when it improves compatibility, correctness, performance, small CLI UX, docs, tests, or maintainability within this vision and can be verified end to end.

Ask first when the work changes product direction, adds a major feature, increases complexity substantially, needs unavailable live credentials, or cannot be verified with confidence.
