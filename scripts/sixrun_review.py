#!/usr/bin/env python3
"""Read-only bare/chat browser for the completed sixrun archive; no GPU imports."""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit
from PIL import Image, PngImagePlugin

PROJECT = Path(__file__).resolve().parents[1]
COUNTS = {'edival': (572, 1716), 'mice': (720, 2160), 'imgedit': (30, 88)}
PngImagePlugin.MAX_TEXT_CHUNK = 4 * 1024 * 1024


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixel_digest(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        value = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
        value.update(rgb.tobytes())
        return value.hexdigest()


class Archive:
    def __init__(self, base):
        self.base = base.resolve()
        self.images, self.sessions, self.catalog = {}, {}, {}
        plan = read(self.base / 'plan.json')
        paths = []
        for name in plan['runs']:
            root = self.base / 'inference' / name / 'run'
            for sid, instructions in read(root / 'run.json')['samples']:
                directory = (root / sid).resolve()
                assert directory.is_relative_to(root.resolve()), sid
                paths.extend(directory / ('turn_0_input.png' if t == 0 else f'turn_{t}.png')
                             for t in range(len(instructions) + 1))
        def inspect(path):
            return path, (digest(path), pixel_digest(path))
        print(f'Verifying {len(paths)} PNG references with 4 CPU threads', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            verified = dict(pool.map(inspect, paths))
        for bench, (count, turns) in COUNTS.items():
            runs = {}
            for label in ('bare', 'chat'):
                name = f'{bench}_{label}'
                root = self.base / 'inference' / name
                assert read(root / 'completion.json')['state'] == 'completed', name
                validation = read(root / 'validation.json')
                assert validation['status'] == 'passed', name
                run = read(root / 'run/run.json')
                assert run['fingerprint'] == plan['runs'][name]['fingerprint'], name
                assert len(run['samples']) == count and sum(len(x[1]) for x in run['samples']) == turns
                assert (validation['sessions'], validation['turns']) == (count, turns)
                runs[label] = (root / 'run', run)
            assert runs['bare'][1]['samples'] == runs['chat'][1]['samples'], bench
            entries = []
            for index, (sid, instructions) in enumerate(runs['bare'][1]['samples']):
                item = {'id': sid, 'index': index, 'split': sid.split('/')[0],
                        'instructions_en': instructions, 'variants': {}}
                source_hash = None
                for label, (run_root, run) in runs.items():
                    directory = (run_root / sid).resolve()
                    assert directory.is_relative_to(run_root.resolve()), sid
                    spec = read(directory / 'session.json')
                    assert spec['session_id'] == sid and spec['instructions'] == instructions
                    if source_hash is None:
                        source_hash = spec['source_hash']
                    assert spec['source_hash'] == source_hash, sid
                    hashes, images, metas = [source_hash], [], []
                    for turn in range(len(instructions) + 1):
                        image = directory / ('turn_0_input.png' if turn == 0 else f'turn_{turn}.png')
                        assert image.is_file(), image
                        sha, pixels = verified[image]
                        if turn:
                            meta = read(directory / f'turn_{turn}.json')
                            assert meta['session_fingerprint'] == spec['fingerprint']
                            assert meta['turn'] == turn and meta['instruction'] == instructions[turn - 1]
                            assert meta['output_hash'] == pixels and meta['input_hashes'] == hashes
                            assert meta['protocol']['version'] == f'lance-history-{label}-' + ('v2' if label == 'bare' else 'v1')
                            hashes.append(pixels)
                            metas.append(meta)
                        else:
                            assert pixels == source_hash
                        route = f'/image/{bench}/{index}/{label}/{turn}'
                        self.images[route] = (image, sha)
                        images.append({'url': route, 'sha256': sha})
                    item['variants'][label] = {'fingerprint': spec['fingerprint'], 'images': images,
                                               'turn_metadata': metas}
                assert item['variants']['bare']['images'][0]['sha256'] == item['variants']['chat']['images'][0]['sha256']
                self.sessions[f'/api/session/{bench}/{index}'] = item
                entries.append({'id': sid, 'index': index, 'split': item['split'], 'instructions_en': instructions})
            self.catalog[bench] = {'sessions': entries, 'total_sessions': count, 'turns_per_variant': turns,
                                  'run_fingerprints': {k: v[1]['fingerprint'] for k, v in runs.items()}}
            print(f'{bench}: both variants verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    archive = Archive(args.base)
    print(json.dumps({'status': 'passed', 'base': str(archive.base),
                      'benchmarks': {b: {'sessions': x['total_sessions'], 'turns_per_variant': x['turns_per_variant']}
                                     for b, x in archive.catalog.items()},
                      'image_references': len(archive.images), 'checks': ['run identity', 'paired selection',
                      'completion and validation', 'session identity', 'PNG SHA256', 'full history hash chain',
                      'protocol version', 'same source image']}, ensure_ascii=False), flush=True)
    if args.check:
        return
    page = (PROJECT / 'web/sixrun-review.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(True)

        def do_GET(self):
            self.respond(False)

        def respond(self, head):
            route = urlsplit(self.path).path
            path, body = None, None
            if route == '/':
                body, mime = page, 'text/html; charset=utf-8'
            elif route == '/api/catalog':
                body, mime = json.dumps(archive.catalog, ensure_ascii=False).encode(), 'application/json; charset=utf-8'
            elif route in archive.sessions:
                body, mime = json.dumps(archive.sessions[route], ensure_ascii=False).encode(), 'application/json; charset=utf-8'
            elif route in archive.images:
                path, expected = archive.images[route]
                if digest(path) != expected:
                    self.send_error(409, 'Image changed after startup verification')
                    return
                mime = 'image/png'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(path.stat().st_size if path else len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'")
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
    print(f'Sixrun comparison: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
