// A stand-in ACP agent on stdin/stdout (newline-delimited JSON-RPC), for the server's acp driver.
import { createInterface } from 'node:readline';

const send = (m) => process.stdout.write(`${JSON.stringify(m)}\n`);
process.stderr.write(`fake agent in ${process.cwd()} key=${process.env.ANTHROPIC_API_KEY ?? ''}\n`);
createInterface({ input: process.stdin }).on('line', (line) => {
  const m = JSON.parse(line);
  if (m.method === 'initialize') send({ jsonrpc: '2.0', id: m.id, result: { protocolVersion: 1 } });
  else if (m.method === 'session/new') send({ jsonrpc: '2.0', id: m.id, result: { sessionId: 's1' } });
  else if (m.method === 'session/prompt') {
    send({ jsonrpc: '2.0', method: 'session/update', params: { sessionId: 's1', update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: `echo: ${m.params.prompt[0].text}` } } } });
    send({ jsonrpc: '2.0', id: m.id, result: { stopReason: 'end_turn' } });
  }
});
