#!/usr/bin/env python3
"""Read-only, loopback-only MICE review server. Python standard library only."""
import argparse
import hashlib
import json
import random
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT / 'outputs/mice/full_20261004_attention/run'
DEFAULT_REVIEW = PROJECT / 'outputs/review/mice_20_20261004'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select(run, seed, per_split):
    rng = random.Random(seed)
    chosen, population = [], {}
    for split in ('cm', 'cu'):
        paths = sorted(p for p in (run / split).iterdir() if p.is_dir())
        population[split] = len(paths)
        chosen.extend(sorted(rng.sample(paths, per_split)))
    return chosen, population


def prepare(args):
    run = args.run.resolve()
    translations = read_json(args.translations)
    chosen, population = select(run, args.seed, args.per_split)
    sessions = []
    for directory in chosen:
        spec = read_json(directory / 'session.json')
        sid = directory.relative_to(run).as_posix()
        if spec['session_id'] != sid:
            raise ValueError(f'Session identity mismatch: {sid}')
        translation = translations['sessions'][sid]
        if translation['fingerprint'] != spec['fingerprint']:
            raise ValueError(f'Translation belongs to another session version: {sid}')
        zh = translation['zh']
        if len(zh) != 3 or len(spec['instructions']) != 3 or not all(zh):
            raise ValueError(f'Expected three translated turns: {sid}')
        images = []
        for turn in range(4):
            name = 'turn_0_input.png' if turn == 0 else f'turn_{turn}.png'
            path = directory / name
            if not path.resolve().is_relative_to(run) or not path.is_file():
                raise ValueError(f'Missing or out-of-root image: {path}')
            if turn:
                meta = read_json(directory / f'turn_{turn}.json')
                if (meta['instruction'] != spec['instructions'][turn - 1]
                        or meta['session_fingerprint'] != spec['fingerprint']
                        or meta['turn'] != turn):
                    raise ValueError(f'Turn metadata mismatch: {sid}/{turn}')
            images.append({'path': f'{sid}/{name}', 'url': f'/image/{sid}/{turn}',
                           'sha256': sha256(path), 'bytes': path.stat().st_size})
        sessions.append({'id': sid, 'split': directory.parent.name,
                         'fingerprint': spec['fingerprint'],
                         'instructions_en': spec['instructions'],
                         'instructions_zh': zh, 'images': images})
    manifest = {'version': 1, 'run': str(run), 'seed': args.seed,
                'per_split': args.per_split, 'population': population,
                'sampling': 'Python Random(seed); sorted directories; CM then CU; sample without replacement',
                'translation_note': 'Assistant translation for display only; original instructions unchanged; pronouns preserved.',
                'translations_sha256': sha256(args.translations), 'sessions': sessions}
    destination = args.review / 'manifest.json'
    if destination.exists():
        if read_json(destination) != manifest:
            raise FileExistsError('Existing manifest differs; choose another --review directory')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'manifest': str(destination), 'sessions': len(sessions),
                      'images': 4 * len(sessions), 'population': population}, ensure_ascii=False))


def serve(args):
    manifest = read_json(args.review / 'manifest.json')
    run = Path(manifest['run']).resolve()
    images = {}
    for session in manifest['sessions']:
        for item in session['images']:
            path = (run / item['path']).resolve()
            if not path.is_relative_to(run) or sha256(path) != item['sha256']:
                raise ValueError(f'Image changed after selection: {path}')
            images[item['url']] = path
    public = {k: v for k, v in manifest.items() if k != 'run'}
    data = json.dumps(public, ensure_ascii=False).encode()
    page = (PROJECT / 'web/mice-review.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond()

        def respond(self, head=False):
            route = urlsplit(self.path).path
            if route in ('/', '/index.html'):
                body, mime, path = page, 'text/html; charset=utf-8', None
            elif route == '/api/manifest':
                body, mime, path = data, 'application/json; charset=utf-8', None
            elif route in images:
                body, mime, path = None, 'image/png', images[route]
            else:
                self.send_error(404)
                return
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
    print(f'MICE review: http://127.0.0.1:{args.port} ({len(images)} images)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'serve'))
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    parser.add_argument('--translations', type=Path, default=PROJECT / 'configs/mice_review_20_zh.json')
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--per-split', type=int, default=10)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else serve)(args)


if __name__ == '__main__':
    main()
