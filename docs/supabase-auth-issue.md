---
summary: 'Notes on OAuth scope expectations for hosted MCP servers (e.g., Supabase) and why mcporter requests currently fail.'
read_when:
  - 'Investigating OAuth errors or scope negotiation problems'
---

# OAuth Notes & Hosted MCP Compatibility

## MCP Spec Expectations

- The June 18 2025 MCP “Authorization” spec (and the linked OAuth flow document) requires OAuth 2.1 dynamic client registration plus RFC 7235 `WWW-Authenticate` challenges for scope negotiation.
- Servers SHOULD expose discovery metadata via `/.well-known/oauth-authorization-server` and `/.well-known/oauth-protected-resource`. Those documents list supported scopes and additional OAuth endpoints.
- Clients are expected to start with a baseline scope (the spec uses `mcp:tools`) and then re-authenticate with any server-supplied scope hints.

We currently hard-code `mcp:tools` because it is the only scope guaranteed to exist in the MCP reference implementation. We need richer negotiation to support providers that enforce product-specific scopes.

## Hosted Supabase MCP (~Oct 2025)

Supabase’s hosted MCP server (`https://mcp.supabase.com/mcp`) validates the requested scopes against their management API permissions (`organizations:read`, `projects:read`, `database:write`, `storage:read`, etc.). When mcporter asks for `mcp:tools`, their authorization server rejects the request with HTTP 400 and a body similar to:

```
{"message":"scope.0: Invalid enum value. Expected 'organizations:read' | 'projects:read' | 'projects:write' | 'database:write' | 'database:read' | 'analytics:read' | 'secrets:read' | 'edge_functions:read' | 'edge_functions:write' | 'environment:read' | 'environment:write' | 'storage:read', received 'mcp:tools'"}
```

Key takeaways from Supabase’s docs:

1. Hosted installs rely on dynamic client registration and are currently integrated with Cursor/Windsurf (which embed the Supabase scope list).
2. Manual authentication is offered for self-hosted/CI workflows (PAT headers or custom OAuth app), but their hosted server still expects Supabase-specific scopes.
3. There is no public metadata endpoint describing those scopes, so third-party MCP clients cannot opt in automatically.

## mcporter’s Current Behavior

- We auto-promote ad-hoc HTTP servers to OAuth and retry once when we see 401/403 errors.
- We surface any server-supplied OAuth error payload so it’s obvious whether the problem is scope-related, a missing token, etc.
- After the second failure we stop retrying to avoid infinite loops.

This works for providers that use standard MCP scopes (e.g., the MCP example server, Vercel’s MCP), but it fails for Supabase because they reject `mcp:tools` outright.

## Roadmap / Proposed Improvements

1. **Scope discovery & negotiation**
   - Fetch `/.well-known/oauth-protected-resource` and `/.well-known/oauth-authorization-server` when a server advertises OAuth. If `scopes_supported` is present, intersect it with user overrides and use that list instead of `mcp:tools`.
   - Parse `WWW-Authenticate` challenges on 401/403 responses and restart the authorization flow with the scopes the server demands.
2. **User overrides**
   - Add CLI/env controls (`--oauth-scope`, `MCPORTER_OAUTH_SCOPES`) so users can manually supply provider-specific scope strings (comma-separated) when the server doesn’t expose metadata.
   - Persist these overrides in the server definition (especially for `--persist` ad-hoc runs) so future calls reuse the negotiated scope list.
3. **Better errors & docs**
   - When the server rejects `mcp:tools`, suggest either setting custom scopes or following the provider’s manual-auth instructions (e.g., Supabase’s PAT flow).
   - Link CLI errors to this document so users understand whether the limitation is on our side or the provider’s.
4. **Provider outreach**
   - File upstream issues (e.g., Supabase) requesting support for standard MCP scopes or publication of `scopes_supported` metadata, so we can integrate without custom code.

## Workarounds Today

- Use a supported GUI client (Cursor, Claude Desktop, Windsurf) for Supabase’s hosted MCP—they already ship the necessary scopes.
- Self-host Supabase MCP and configure PAT headers or your own OAuth client; you can then relax scope validation to include `mcp:tools`.
- For other providers, consult their docs for discovery metadata. If they list scopes, set them via a future `--oauth-scope` flag once we implement it (tracked in #TODO).

Until Supabase (and other providers with custom scopes) expose metadata or accept the standard scope, mcporter cannot complete their hosted OAuth flow. This is a platform limitation, not a CLI bug.
