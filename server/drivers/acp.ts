// System 2: one ACP agent per worker exchange (the 'acp' channel). JSON-RPC messages go both
// ways unchanged, one per io.send / io.recv, so the recording holds the whole conversation and a
// replay needs no agent. The agent is either a local process the server starts (FV_ACP_COMMAND,
// newline-delimited JSON-RPC on stdin/stdout) or a remote endpoint (FV_ACP_URL, one JSON-RPC
// message per WebSocket text frame), both set in the web UI's Settings.

import { spawn } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { WebSocket } from 'ws';
import type { LiveDriver } from '../../src/core/world';
import type { Json } from '../../src/core/types';

export interface AcpConfig {
  /** A remote ACP endpoint (ws:// or wss://); when set, no process is started. */
  url?: string;
  /** Sent as a bearer token to the remote endpoint. */
  token?: string;
  /** Started once per worker, through a shell. */
  command?: string;
  /** Where a started agent runs. */
  cwd: string;
  /** Added to a started agent's environment (its API key and base URL). */
  env: Record<string, string>;
}

/** Started agents, by process group, so the server takes them down with it. */
const agents = new Set<number>();
let exitHook = false;
const hookExit = () => {
  if (exitHook) return;
  exitHook = true;
  process.on('exit', () => {
    for (const pid of agents) {
      try {
        process.kill(-pid, 'SIGTERM');
      } catch {
        // already gone
      }
    }
  });
};

const parse = (line: string): Json => {
  try {
    return JSON.parse(line) as Json;
  } catch {
    // Not JSON-RPC: something the agent printed to stdout.
    return { $stdout: line };
  }
};

/**
 * The first message is ours: {$open} with where the agent runs (the folder it was started in, or
 * the endpoint); a started agent's stderr lines come as {$stderr}, for errors.
 */
export const acpDriver =
  (config: () => AcpConfig): LiveDriver =>
  (_req, sink) => {
    const c = config();
    let done = false;
    const close = (data: Json) => {
      if (done) return;
      done = true;
      sink.close(data);
    };
    if (c.url) return remote(c, sink, close);
    if (!c.command) {
      setTimeout(() => close({ error: 'no System 2 agent configured: set FV_ACP_COMMAND or FV_ACP_URL in Settings' }), 0);
      return { send() {}, close() {} };
    }
    // The node starts talking at once: hold its messages until the agent is up.
    const early: string[] = [];
    let write = (line: string) => void early.push(line);
    let stopped = false;
    let stop = () => void (stopped = true);
    (async () => {
      await mkdir(c.cwd, { recursive: true });
      if (stopped) return;
      hookExit();
      // Its own process group: the command runs through a shell (npx, pnpm exec …), and stopping must reach the agent under it.
      const child = spawn(c.command!, { cwd: c.cwd, env: { ...process.env, ...c.env }, shell: true, detached: true, stdio: ['pipe', 'pipe', 'pipe'] });
      const kill = (sig: NodeJS.Signals) => {
        try {
          process.kill(-child.pid!, sig);
        } catch {
          // already gone
        }
      };
      if (child.pid) agents.add(child.pid);
      sink.recv({ $open: { cwd: c.cwd, command: c.command!, pid: child.pid ?? null } });
      createInterface({ input: child.stdout }).on('line', (line) => !done && line.trim() && sink.recv(parse(line)));
      createInterface({ input: child.stderr }).on('line', (line) => !done && sink.recv({ $stderr: line }));
      child.stdin.on('error', () => {});
      write = (line) => void (child.stdin.writable && child.stdin.write(`${line}\n`));
      for (const line of early.splice(0)) write(line);
      child.on('error', (e) => close({ error: `could not start the agent: ${String(e)}` }));
      // 'close', not 'exit': it comes after stdout and stderr are drained, so the last lines go out first.
      child.on('close', (code, signal) => {
        agents.delete(child.pid!);
        close(code === 0 ? null : { code: code ?? null, reason: `agent exited (${signal ?? code})`, error: `agent exited (${signal ?? code})` });
      });
      stop = () => {
        if (child.exitCode === null && child.pid) {
          kill('SIGTERM');
          setTimeout(() => child.exitCode === null && kill('SIGKILL'), 3000).unref();
        }
      };
      if (stopped) stop();
    })().catch((e) => close({ error: String(e) }));
    return {
      send: (data) => write(JSON.stringify(data)),
      close() {
        done = true;
        stopped = true;
        stop();
      },
    };
  };

/** A remote ACP endpoint over a WebSocket: one JSON-RPC message per text frame. */
function remote(c: AcpConfig, sink: Parameters<LiveDriver>[1], close: (data: Json) => void) {
  const ws = new WebSocket(c.url!, { headers: c.token ? { authorization: `Bearer ${c.token}` } : {} });
  const pending: string[] = [];
  let closedByNode = false;
  ws.on('open', () => {
    if (closedByNode) return ws.close(1000);
    sink.recv({ $open: { url: c.url! } });
    for (const p of pending.splice(0)) ws.send(p);
  });
  ws.on('message', (data: Buffer) => !closedByNode && sink.recv(parse(data.toString('utf8'))));
  ws.on('unexpected-response', (_req, res) => {
    if (!closedByNode) close({ code: 1011, error: `System 2 endpoint: HTTP ${res.statusCode}` });
    ws.terminate();
  });
  ws.on('error', (e) => !closedByNode && close({ code: 1006, error: String(e).slice(0, 200) }));
  ws.on('close', (code, reason) => !closedByNode && close(code === 1000 ? null : { code, reason: reason.toString('utf8'), error: `websocket closed (${code})` }));
  return {
    send(data: Json) {
      const line = JSON.stringify(data);
      if (ws.readyState === WebSocket.OPEN) ws.send(line);
      else if (ws.readyState === WebSocket.CONNECTING) pending.push(line);
    },
    close() {
      closedByNode = true;
      // Still connecting: its open handler closes it.
      if (ws.readyState === WebSocket.OPEN) ws.close(1000);
    },
  };
}
