import type { Client } from '@modelcontextprotocol/sdk/client/index.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type { Logger } from '../logging.js';
import type { OAuthSession } from '../oauth.js';
import { isUnauthorizedError } from '../runtime-oauth-support.js';

export const DEFAULT_OAUTH_CODE_TIMEOUT_MS = 60_000;
const OAUTH_FLOW_ERROR = Symbol('oauth-flow-error');
const POST_AUTH_CONNECT_ERROR = Symbol('post-auth-connect-error');

export interface OAuthCapableTransport extends Transport {
  close(): Promise<void>;
  finishAuth?: (authorizationCode: string) => Promise<void>;
}

export interface ConnectWithAuthOptions {
  serverName?: string;
  maxAttempts?: number;
  oauthTimeoutMs?: number;
  recreateTransport?: (transport: OAuthCapableTransport) => Promise<OAuthCapableTransport>;
}

export class OAuthTimeoutError extends Error {
  public readonly timeoutMs: number;
  public readonly serverName: string;

  constructor(serverName: string, timeoutMs: number) {
    const seconds = Math.round(timeoutMs / 1000);
    super(`OAuth authorization for '${serverName}' timed out after ${seconds}s; aborting.`);
    this.name = 'OAuthTimeoutError';
    this.timeoutMs = timeoutMs;
    this.serverName = serverName;
  }
}

export function markOAuthFlowError(error: unknown): unknown {
  return markError(error, OAUTH_FLOW_ERROR);
}

export function isOAuthFlowError(error: unknown): boolean {
  return hasErrorMarker(error, OAUTH_FLOW_ERROR);
}

export function markPostAuthConnectError(error: unknown): unknown {
  return markError(error, POST_AUTH_CONNECT_ERROR);
}

export function isPostAuthConnectError(error: unknown): boolean {
  return hasErrorMarker(error, POST_AUTH_CONNECT_ERROR);
}

function markError(error: unknown, marker: symbol): unknown {
  if (!error || (typeof error !== 'object' && typeof error !== 'function')) {
    return error;
  }
  Object.defineProperty(error, marker, {
    value: true,
    enumerable: false,
    configurable: true,
  });
  return error;
}

function hasErrorMarker(error: unknown, marker: symbol): boolean {
  return (
    !!error &&
    (typeof error === 'object' || typeof error === 'function') &&
    marker in error &&
    Boolean((error as Record<PropertyKey, unknown>)[marker])
  );
}

export async function connectWithAuth(
  client: Client,
  transport: OAuthCapableTransport,
  session: OAuthSession | undefined,
  logger: Logger,
  options: ConnectWithAuthOptions = {}
): Promise<OAuthCapableTransport> {
  const { serverName, maxAttempts = 3, oauthTimeoutMs = DEFAULT_OAUTH_CODE_TIMEOUT_MS, recreateTransport } = options;
  let activeTransport = transport;
  let attempt = 0;
  let hasCompletedAuthFlow = false;

  while (true) {
    try {
      await client.connect(activeTransport);
      return activeTransport;
    } catch (error) {
      const unauthorized = isUnauthorizedError(error);
      if (hasCompletedAuthFlow && !unauthorized) {
        await closeReplacementTransport(transport, activeTransport);
        throw markPostAuthConnectError(error);
      }
      if (!unauthorized || !session) {
        await closeReplacementTransport(transport, activeTransport);
        throw error;
      }
      attempt += 1;
      if (attempt > maxAttempts) {
        await closeReplacementTransport(transport, activeTransport);
        throw hasCompletedAuthFlow ? markPostAuthConnectError(error) : error;
      }
      logger.warn(`OAuth authorization required for '${serverName ?? 'unknown'}'. Waiting for browser approval...`);
      try {
        activeTransport = await completeAuthorizationChallenge(activeTransport, session, logger, error, {
          serverName,
          oauthTimeoutMs,
          recreateTransport,
        });
        hasCompletedAuthFlow = true;
        logger.info('Authorization code accepted. Retrying connection...');
      } catch (authError) {
        logger.error('OAuth authorization failed while waiting for callback.', authError);
        await closeReplacementTransport(transport, activeTransport);
        throw markOAuthFlowError(authError);
      }
    }
  }
}

async function closeReplacementTransport(
  originalTransport: OAuthCapableTransport,
  activeTransport: OAuthCapableTransport
): Promise<void> {
  if (activeTransport === originalTransport) {
    return;
  }
  await activeTransport.close().catch(() => {});
}

async function completeAuthorizationChallenge(
  transport: OAuthCapableTransport,
  session: OAuthSession,
  logger: Logger,
  connectError: unknown,
  options: Pick<ConnectWithAuthOptions, 'serverName' | 'oauthTimeoutMs' | 'recreateTransport'>
): Promise<OAuthCapableTransport> {
  const code = await waitForAuthorizationCodeWithTimeout(
    session,
    logger,
    options.serverName,
    options.oauthTimeoutMs ?? DEFAULT_OAUTH_CODE_TIMEOUT_MS
  );
  if (typeof transport.finishAuth !== 'function') {
    logger.warn('Transport does not support finishAuth; cannot complete OAuth flow automatically.');
    throw connectError;
  }
  await transport.finishAuth(code);
  if (!options.recreateTransport) {
    return transport;
  }
  const nextTransport = await options.recreateTransport(transport);
  await transport.close().catch(() => {});
  return nextTransport;
}

// Race the pending OAuth browser handshake so the runtime can't sit on an unresolved promise forever.
export function waitForAuthorizationCodeWithTimeout(
  session: OAuthSession,
  logger: Logger,
  serverName?: string,
  timeoutMs = DEFAULT_OAUTH_CODE_TIMEOUT_MS
): Promise<string> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return session.waitForAuthorizationCode();
  }
  const displayName = serverName ?? 'unknown';
  return new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => {
      const error = new OAuthTimeoutError(displayName, timeoutMs);
      logger.warn(error.message);
      reject(error);
    }, timeoutMs);
    session.waitForAuthorizationCode().then(
      (code) => {
        clearTimeout(timer);
        resolve(code);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

export function parseOAuthTimeout(raw: string | undefined): number {
  if (!raw) {
    return DEFAULT_OAUTH_CODE_TIMEOUT_MS;
  }
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_OAUTH_CODE_TIMEOUT_MS;
  }
  return parsed;
}

export function resolveOAuthTimeoutFromEnv(): number {
  return parseOAuthTimeout(process.env.MCPORTER_OAUTH_TIMEOUT_MS ?? process.env.MCPORTER_OAUTH_TIMEOUT);
}
