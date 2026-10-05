#!/usr/bin/env python3
"""Read-only EdiVal result browser, reusing the existing four-image review layout."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Review:
    def __init__(self, run, review, translations=None):
        self.run, self.review = run.resolve(), review.resolve()
        self.translations = translations
        self.hashes, self.lock = {}, threading.Lock()

    def file_hash(self, path):
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        old = self.hashes.get(path)
        if old and old[0] == key:
            return old[1]
        value = sha256(path)
        self.hashes[path] = (key, value)
        return value

    def manifest(self):
        with self.lock:
            run = read_json(self.run / 'run.json')
            if run.get('benchmark') != 'edival' or run['settings']['settings']['resolution'] != 512:
                raise ValueError('Expected 512-pixel EdiVal run')
            if len(run['samples']) != 572:
                raise ValueError('Expected complete EdiVal release selection')
            translated = {}
            if self.translations and self.translations.exists():
                bundle = read_json(self.translations)
                if bundle['run_fingerprint'] != run['fingerprint']:
                    raise ValueError('Translations belong to a different run')
                translated = bundle['sessions']
            sessions, completed_turns, attention_count = [], 0, 0
            for sid, instructions in run['samples']:
                if not re.fullmatch(r'edival/(0|[1-9][0-9]*)', sid) or len(instructions) != 3:
                    raise ValueError('Invalid session identity')
                directory = self.run / sid
                spec_path = directory / 'session.json'
                spec = read_json(spec_path) if spec_path.exists() else None
                if spec and (spec['session_id'] != sid or spec['instructions'] != instructions):
                    raise ValueError('Saved session identity differs')
                translation = translated.get(sid)
                if translation and (translation['en'] != instructions or len(translation['zh']) != 3):
                    raise ValueError('Translated instruction sequence differs')
                images = []
                turns = 0
                for turn in range(4):
                    name = 'turn_0_input.png' if turn == 0 else f'turn_{turn}.png'
                    path = directory / name
                    available = bool(spec and path.is_file())
                    if turn:
                        meta_path = directory / f'turn_{turn}.json'
                        attention_path = directory / f'turn_{turn}.attention.npz'
                        available = available and meta_path.is_file() and attention_path.is_file()
                        if available:
                            meta = read_json(meta_path)
                            if (meta['session_fingerprint'] != spec['fingerprint'] or meta['turn'] != turn
                                    or meta['instruction'] != instructions[turn-1]):
                                raise ValueError('Turn metadata identity differs')
                            turns += 1
                            attention_count += 1
                    images.append({'path': f'{sid}/{name}', 'url': f'/image/{sid}/{turn}',
                                   'available': available, 'sha256': self.file_hash(path) if available else None})
                completed_turns += turns
                sessions.append({'id': sid, 'split': 'edival', 'fingerprint': spec['fingerprint'] if spec else None,
                    'instructions_en': instructions, 'instructions_zh': translation['zh'] if translation else ['', '', ''],
                    'translated': bool(translation), 'completed_turns': turns,
                    'complete': turns == 3, 'images': images})
            status_path = self.run.parent / 'status.json'
            status = read_json(status_path) if status_path.exists() else {'state': 'unknown'}
            validation_path = self.run.parent / 'validation.json'
            validation = read_json(validation_path) if validation_path.exists() else None
            return {'version': 1, 'benchmark': 'edival', 'run_fingerprint': run['fingerprint'],
                    'resolution': 512, 'total_sessions': 572, 'total_turns': 1716,
                    'completed_sessions': sum(s['complete'] for s in sessions),
                    'completed_turns': completed_turns, 'attention_files': attention_count,
                    'state': status['state'], 'validated': bool(validation and validation['status'] == 'passed'),
                    'translation_note': '中文仅用于辅助浏览，保留英文原始指令；缺少译文的会话直接显示英文。',
                    'sessions': sessions}

    def image_path(self, route):
        match = re.fullmatch(r'/image/(edival/(?:0|[1-9][0-9]*))/([0-3])', route)
        if not match:
            return None
        sid, turn = match.group(1), int(match.group(2))
        name = 'turn_0_input.png' if turn == 0 else f'turn_{turn}.png'
        path = (self.run / sid / name).resolve()
        if not path.is_relative_to(self.run) or not path.is_file():
            return None
        if turn and not (path.parent / f'turn_{turn}.json').is_file():
            return None
        return path

    def finalize(self):
        manifest = self.manifest()
        if not manifest['validated'] or manifest['completed_turns'] != 1716:
            raise ValueError('Complete provenance validation is required before publishing final manifest')
        self.review.mkdir(parents=True, exist_ok=True)
        path = self.review / 'manifest.json'
        if path.exists() and read_json(path) != manifest:
            raise ValueError('Existing review manifest differs')
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        return manifest


def serve(args):
    review = Review(args.run, args.review, args.translations)
    page = (PROJECT / 'web/edival-review.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(True)

        def do_GET(self):
            self.respond(False)

        def respond(self, head):
            route = urlsplit(self.path).path
            path, body = None, None
            if route in ('/', '/index.html'):
                mime, body = 'text/html; charset=utf-8', page
            elif route == '/api/manifest':
                try:
                    body = json.dumps(review.manifest(), ensure_ascii=False).encode()
                except (FileNotFoundError, ValueError, KeyError) as exc:
                    self.send_error(503, str(exc))
                    return
                mime = 'application/json; charset=utf-8'
            else:
                path = review.image_path(route)
                if not path:
                    self.send_error(404)
                    return
                mime = 'image/png'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(path.stat().st_size if path else len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            if not head:
                try:
                    if path:
                        with path.open('rb') as stream:
                            shutil.copyfileobj(stream, self.wfile)
                    else:
                        self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'EdiVal results: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('serve', 'finalize'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--translations', type=Path)
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    if args.command == 'serve':
        serve(args)
    else:
        manifest = Review(args.run, args.review, args.translations).finalize()
        print(json.dumps({k: manifest[k] for k in ('completed_sessions', 'completed_turns', 'attention_files', 'validated')}))


if __name__ == '__main__':
    main()
