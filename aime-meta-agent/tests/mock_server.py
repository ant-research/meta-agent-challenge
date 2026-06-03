#!/usr/bin/env python3
"""
Mock OpenAI-compatible LLM server for AIME proxy testing.
Returns deterministic responses with configurable token counts in usage.

Usage:
  python3 aime-meta-agent/tests/mock_server.py          # default port 9090
  python3 aime-meta-agent/tests/mock_server.py 9090      # custom port
"""
import http.server
import json
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9090


class LLMHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/v1/models'):
            self._json(200, {
                'object': 'list',
                'data': [{'id': 'mock-model', 'object': 'model', 'owned_by': 'mock'}],
            })
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(length) if length else b'{}'
        body = json.loads(raw)
        if '/chat/completions' in self.path:
            msgs = body.get('messages', [])
            user_msg = next(
                (m['content'] for m in reversed(msgs) if m.get('role') == 'user'), 'hi'
            )
            reply = f"Mock answer to: {user_msg[:60]}"
            prompt_tokens = max(10, len(user_msg) // 4)
            completion_tokens = max(5, len(reply) // 4)
            self._json(200, {
                'id': 'mock-chatcmpl-0001',
                'object': 'chat.completion',
                'model': body.get('model', 'mock-model'),
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': reply},
                    'finish_reason': 'stop',
                }],
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                },
            })
        else:
            self._json(404, {'error': 'not found'})

    def _json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress logs


if __name__ == '__main__':
    srv = http.server.HTTPServer(('0.0.0.0', PORT), LLMHandler)
    print(f'[mock] LLM server http://0.0.0.0:{PORT}', flush=True)
    print('[mock] Ctrl+C to stop', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\n[mock] Stopped.')
