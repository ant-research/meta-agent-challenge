#!/usr/bin/env python3
"""
宿主机 mock server，供测试用：
  - port 9090: OpenAI-compatible LLM API
  - port 9091: Search API（返回固定结果）

用法：
  python3 science-meta-agent/tests/mock_server.py
  python3 science-meta-agent/tests/mock_server.py 9090 9091   # 自定义端口
"""
import http.server
import json
import sys
import threading
import time

LLM_PORT    = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
SEARCH_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9091


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
                    'prompt_tokens': max(10, len(user_msg) // 4),
                    'completion_tokens': max(5, len(reply) // 4),
                    'total_tokens': max(15, (len(user_msg) + len(reply)) // 4),
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
        pass


class SearchHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            'results': [
                {
                    'title': 'Mock Search Result',
                    'url': 'https://mock.example.com/result',
                    'snippet': 'This is a mock search result for testing purposes.',
                }
            ],
            'total_results': 1,
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def serve(handler_cls, port):
    srv = http.server.HTTPServer(('0.0.0.0', port), handler_cls)
    srv.serve_forever()


if __name__ == '__main__':
    threading.Thread(target=serve, args=(LLMHandler, LLM_PORT), daemon=True).start()
    threading.Thread(target=serve, args=(SearchHandler, SEARCH_PORT), daemon=True).start()
    print(f'[mock] LLM server    http://0.0.0.0:{LLM_PORT}', flush=True)
    print(f'[mock] Search server http://0.0.0.0:{SEARCH_PORT}', flush=True)
    print('[mock] Ctrl+C to stop', flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print('\n[mock] Stopped.')
