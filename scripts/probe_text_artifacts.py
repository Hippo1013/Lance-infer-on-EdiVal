"""Controlled first-turn text-artifact diagnostic; separate outputs only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

SAMPLES = ('cu/024257c555c6d20e', 'cu/00db092ac5a3c7bc',
           'cm/009e412e4d25ec57', 'cm/05447e326032ca20')
MODES = ('full', 'label_only', 'framing_only', 'bare')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/mice/full_20261004_attention/run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from PIL import Image
    from lance_mice.backend import OmniBackend, find_history_metadata, history_payload
    from lance_mice.images import image_hash
    from lance_mice.protocol import derive_seed
    from lance_mice.runner import write_json
    from lance_mice.settings import Settings

    settings = Settings(seed=args.seed, cache_mode='none')
    cases = []
    for sid in SAMPLES:
        directory = args.run / sid
        spec = json.loads((directory / 'session.json').read_text())
        with Image.open(directory / 'turn_0_input.png') as im:
            source = im.convert('RGB')
        assert image_hash(source) == spec['source_hash']
        cases.append((sid, spec, source))
    plan = {'sessions': list(SAMPLES), 'modes': MODES, 'settings': settings.identity(),
            'scope': 'Four purposively selected first-turn cases, fixed image/instruction/VAE RNG/noise across modes; not a population estimate.',
            'source_run': str(args.run.resolve()),
            'production_code_unchanged': True,
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              [Path(__file__), Path('src/lance_mice/text_artifact_probe.py'),
                               Path('src/lance_mice/omni_pipeline.py'), Path('src/lance_mice/protocol.py')]}}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    records = []
    backend = OmniBackend(Path('/home/chs/model/Lance'), settings,
                          pipeline_class='lance_mice.text_artifact_probe.TextArtifactProbePipeline')
    try:
        for sid, spec, source in cases:
            destination = args.output / sid
            destination.mkdir(parents=True)
            source.save(destination / 'input.png')
            for mode in MODES:
                payload = history_payload(sid, [source], [spec['instructions'][0]], settings,
                                          audit=True, end_session=True)
                payload['extra_args']['lance_history']['text_probe_mode'] = mode
                params = copy.deepcopy(backend.engine.default_sampling_params_list)
                params[0].num_inference_steps = settings.steps
                params[0].seed = derive_seed(settings.seed, sid, 'noise', 1)
                outputs = list(backend.engine.generate(payload, sampling_params_list=params, use_tqdm=False))
                if len(outputs) != 1 or len(outputs[0].images or []) != 1:
                    raise RuntimeError('Expected one image')
                result = outputs[0].images[0].convert('RGB')
                result.save(destination / f'{mode}.png')
                meta = find_history_metadata(outputs[0])
                record = {'session_id': sid, 'mode': mode, 'instruction': spec['instructions'][0],
                          'source_hash': image_hash(source), 'output_hash': image_hash(result),
                          'backend': meta}
                if mode == 'full' and args.seed == 42:
                    with Image.open(args.run / sid / 'turn_1.png') as im:
                        record['matches_full_run_pixels'] = image_hash(im) == image_hash(result)
                    if not record['matches_full_run_pixels']:
                        raise RuntimeError(f'Full-mode baseline does not reproduce saved output: {sid}')
                write_json(destination / f'{mode}.json', record)
                records.append(record)
                write_json(args.output / 'results.json', {'completed':len(records), 'records': records})
                print(f'DONE {sid} {mode}', flush=True)
    finally:
        backend.close()
    write_json(args.output / 'summary.json', {'status':'completed', 'images':len(records),
               'baseline_matches': [r.get('matches_full_run_pixels') for r in records if r['mode']=='full'],
               'visual_review': 'pending'})


if __name__ == '__main__':
    main()
