import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport, StreamableHTTPError } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { ServerDefinition } from '../config.js';
import { resolveEnvValue, withEnvOverrides } from '../env.js';
import { analyzeConnectionError } from '../error-classifier.js';
import type { Logger } from '../logging.js';
import { createOAuthSession, type OAuthSession } from '../oauth.js';
import { readCachedAccessToken } from '../oauth-persistence.js';
import { materializeHeaders } from '../runtime-header-utils.js';
import { isUnauthorizedError, maybeEnableOAuth } from '../runtime-oauth-support.js';
import { closeTransportAndWait } from '../runtime-process-utils.js';
import {
  connectWithAuth,
  isOAuthFlowError,
  isPostAuthConnectError,
  type OAuthCapableTransport,
  OAuthTimeoutError,
} from './oauth.js';
import { resolveCommandArgument, resolveCommandArguments } from './utils.js';

const STDIO_TRACE_ENABLED = process.env.MCPORTER_STDIO_TRACE === '1';

function extractTransportStatusCode(error: unknown): number | undefined {
  if (!error || typeof error !== 'object') {
    return undefined;
  }
  const record = error as Record<string, unknown>;
  for (const candidate of [record.code, record.status, record.statusCode]) {
    if (typeof candidate === 'number') {
      return candidate;
    }
    if (typeof candidate === 'string') {
      const parsed = Number.parseInt(candidate, 10);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
}

function isLegacySseTransportMismatch(error: unknown): boolean {
  if (error instanceof StreamableHTTPError) {
    return error.code === 404 || error.code === 405;
  }
  const directStatusCode = extractTransportStatusCode(error);
  if (directStatusCode === 404 || directStatusCode === 405) {
    return true;
  }
  const issue = analyzeConnectionError(error);
  return issue.kind === 'http' && (issue.statusCode === 404 || issue.statusCode === 405);
}

interface ResolvedHttpTransportOptions {
  requestInit?: RequestInit;
  authProvider?: OAuthSession['provider'];
}

function attachStdioTraceLogging(_transport: StdioClientTransport, _label?: string): void {
  // STDIO instrumentation is handled via sdk-patches side effects. This helper remains
  // so runtime callers can opt-in without sprinkling conditional checks everywhere.
}

export interface ClientContext {
  readonly client: Client;
  readonly transport: Transport & { close(): Promise<void> };
  readonly definition: ServerDefinition;
  readonly oauthSession?: OAuthSession;
}

export interface CreateClientContextOptions {
  readonly maxOAuthAttempts?: number;
  readonly oauthTimeoutMs?: number;
  readonly onDefinitionPromoted?: (definition: ServerDefinition) => void;
  readonly allowCachedAuth?: boolean;
}

function removeAuthorizationHeader(headers: Record<string, string> | undefined): Record<string, string> | undefined {
  if (!headers) {
    return undefined;
  }
  for (const key of Object.keys(headers)) {
    if (key.toLowerCase() === 'authorization') {
      delete headers[key];
    }
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function createHttpTransportOptions(
  definition: ServerDefinition,
  oauthSession: OAuthSession | undefined,
  shouldEstablishOAuth: boolean
): ResolvedHttpTransportOptions {
  const command = definition.command;
  if (command.kind !== 'http') {
    throw new Error(`Server '${definition.name}' is not configured for HTTP transport.`);
  }
  const resolvedHeaders = materializeHeaders(command.headers, definition.name);
  const effectiveHeaders = shouldEstablishOAuth ? removeAuthorizationHeader(resolvedHeaders) : resolvedHeaders;
  return {
    requestInit: effectiveHeaders ? { headers: effectiveHeaders as HeadersInit } : undefined,
    authProvider: oauthSession?.provider,
  };
}

async function closeOAuthSession(oauthSession?: OAuthSession): Promise<void> {
  await oauthSession?.close().catch(() => {});
}

function shouldAbortSseFallback(error: unknown): boolean {
  if (isPostAuthConnectError(error)) {
    return !isLegacySseTransportMismatch(error);
  }
  return isOAuthFlowError(error) || error instanceof OAuthTimeoutError;
}

function maybePromoteHttpDefinition(
  definition: ServerDefinition,
  logger: Logger,
  options: CreateClientContextOptions
): ServerDefinition | undefined {
  if (options.maxOAuthAttempts === 0) {
    return undefined;
  }
  return maybeEnableOAuth(definition, logger);
}

async function connectHttpTransport<TTransport extends OAuthCapableTransport>(
  client: Client,
  transport: TTransport,
  oauthSession: OAuthSession | undefined,
  logger: Logger,
  connectOptions: Parameters<typeof connectWithAuth>[4]
): Promise<TTransport> {
  try {
    return (await connectWithAuth(client, transport, oauthSession, logger, connectOptions)) as TTransport;
  } catch (error) {
    await closeTransportAndWait(logger, transport).catch(() => {});
    throw error;
  }
}

export async function createClientContext(
  definition: ServerDefinition,
  logger: Logger,
  clientInfo: { name: string; version: string },
  options: CreateClientContextOptions = {}
): Promise<ClientContext> {
  const client = new Client(clientInfo);
  let activeDefinition = definition;

  if (options.allowCachedAuth && activeDefinition.auth === 'oauth' && activeDefinition.command.kind === 'http') {
    try {
      const cached = await readCachedAccessToken(activeDefinition, logger);
      if (cached) {
        const existingHeaders = activeDefinition.command.headers ?? {};
        if (!('Authorization' in existingHeaders)) {
          activeDefinition = {
            ...activeDefinition,
            command: {
              ...activeDefinition.command,
              headers: {
                ...existingHeaders,
                Authorization: `Bearer ${cached}`,
              },
            },
          };
          logger.debug?.(`Using cached OAuth access token for '${activeDefinition.name}' (non-interactive).`);
        }
      }
    } catch (error) {
      logger.debug?.(
        `Failed to read cached OAuth token for '${activeDefinition.name}': ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  return withEnvOverrides(activeDefinition.env, async () => {
    if (activeDefinition.command.kind === 'stdio') {
      const resolvedEnvOverrides =
        activeDefinition.env && Object.keys(activeDefinition.env).length > 0
          ? Object.fromEntries(
              Object.entries(activeDefinition.env)
                .map(([key, raw]) => [key, resolveEnvValue(raw)])
                .filter(([, value]) => value !== '')
            )
          : undefined;
      const mergedEnv =
        resolvedEnvOverrides && Object.keys(resolvedEnvOverrides).length > 0
          ? { ...process.env, ...resolvedEnvOverrides }
          : { ...process.env };
      const transport = new StdioClientTransport({
        command: resolveCommandArgument(activeDefinition.command.command),
        args: resolveCommandArguments(activeDefinition.command.args),
        cwd: activeDefinition.command.cwd,
        env: mergedEnv,
      });
      if (STDIO_TRACE_ENABLED) {
        attachStdioTraceLogging(transport, activeDefinition.name ?? activeDefinition.command.command);
      }
      try {
        await client.connect(transport);
      } catch (error) {
        await closeTransportAndWait(logger, transport).catch(() => {});
        throw error;
      }
      return { client, transport, definition: activeDefinition, oauthSession: undefined };
    }

    while (true) {
      const command = activeDefinition.command;
      if (command.kind !== 'http') {
        throw new Error(`Server '${activeDefinition.name}' is not configured for HTTP transport.`);
      }
      let oauthSession: OAuthSession | undefined;
      const shouldEstablishOAuth = activeDefinition.auth === 'oauth' && options.maxOAuthAttempts !== 0;
      if (shouldEstablishOAuth) {
        oauthSession = await createOAuthSession(activeDefinition, logger);
      }
      const transportOptions = createHttpTransportOptions(activeDefinition, oauthSession, shouldEstablishOAuth);

      try {
        const createStreamableTransport = () => new StreamableHTTPClientTransport(command.url, transportOptions);
        const streamableTransport = await connectHttpTransport(
          client,
          createStreamableTransport(),
          oauthSession,
          logger,
          {
            serverName: activeDefinition.name,
            maxAttempts: options.maxOAuthAttempts,
            oauthTimeoutMs: options.oauthTimeoutMs,
            recreateTransport: async () => createStreamableTransport(),
          }
        );
        return {
          client,
          transport: streamableTransport,
          definition: activeDefinition,
          oauthSession,
        };
      } catch (primaryError) {
        if (shouldAbortSseFallback(primaryError)) {
          await closeOAuthSession(oauthSession);
          throw primaryError;
        }
        if (isUnauthorizedError(primaryError)) {
          await closeOAuthSession(oauthSession);
          oauthSession = undefined;
          const promoted = maybePromoteHttpDefinition(activeDefinition, logger, options);
          if (promoted) {
            activeDefinition = promoted;
            options.onDefinitionPromoted?.(promoted);
            continue;
          }
        }
        if (primaryError instanceof Error) {
          logger.info(`Falling back to SSE transport for '${activeDefinition.name}': ${primaryError.message}`);
        }
        try {
          const connectedTransport = await connectHttpTransport(
            client,
            new SSEClientTransport(command.url, transportOptions),
            oauthSession,
            logger,
            {
              serverName: activeDefinition.name,
              maxAttempts: options.maxOAuthAttempts,
              oauthTimeoutMs: options.oauthTimeoutMs,
            }
          );
          return { client, transport: connectedTransport, definition: activeDefinition, oauthSession };
        } catch (sseError) {
          await closeOAuthSession(oauthSession);
          if (sseError instanceof OAuthTimeoutError) {
            throw sseError;
          }
          if (isUnauthorizedError(sseError)) {
            const promoted = maybePromoteHttpDefinition(activeDefinition, logger, options);
            if (promoted) {
              activeDefinition = promoted;
              options.onDefinitionPromoted?.(promoted);
              continue;
            }
          }
          throw sseError;
        }
      }
    }
  });
}
