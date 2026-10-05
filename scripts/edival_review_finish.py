#!/usr/bin/env python3
"""Publish a frozen EdiVal review manifest after inference and HTTP image validation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

from edival_review import Review, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--translations', type=Path)
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    args.review.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('edival_review.py'),
               Path(__file__).resolve().parents[1] / 'web/edival-review.html']
    if args.translations:
        sources.append(args.translations.resolve())
    pins = {str(p): sha256(p) for p in sources}
    status_path = args.run.parent / 'status.json'
    try:
        while True:
            status = read_json(status_path)
            if status['state'] == 'failed':
                raise RuntimeError('Inference failed: ' + status.get('error', 'see run.log'))
            if status['state'] == 'completed':
                break
            time.sleep(20)
        review = Review(args.run, args.review, args.translations)
        manifest = review.manifest()
        if not manifest['validated'] or manifest['completed_turns'] != 1716:
            raise ValueError('Full inference provenance audit has not passed')
        base = f'http://127.0.0.1:{args.port}'
        with urlopen(base + '/api/manifest', timeout=30) as response:
            served = json.load(response)
        if served != manifest:
            raise ValueError('Served manifest differs from final results')
        images = 0
        for session in manifest['sessions']:
            for image in session['images']:
                if not image['available']:
                    raise ValueError('Missing final review image')
                with urlopen(base + image['url'], timeout=30) as response:
                    digest = hashlib.sha256(response.read()).hexdigest()
                if digest != image['sha256']:
                    raise ValueError('HTTP image hash differs: ' + image['path'])
                images += 1
        if images != 2288 or pins != {str(p): sha256(p) for p in sources}:
            raise ValueError('Review coverage or source changed')
        review.finalize()
        result = {'status': 'passed', 'sessions': 572, 'turns': 1716,
                  'attention_files': 1716, 'http_images_verified': images,
                  'run_fingerprint': manifest['run_fingerprint'], 'source_hashes': pins,
                  'finished_at': datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                  'finished_at': datetime.now(timezone.utc).isoformat()}
    (args.review / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
