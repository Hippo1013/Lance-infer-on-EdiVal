"""Small, reproducible three-turn prompt-layout study; never changes formal runs."""
import argparse
import copy
import hashlib
import html
import json
import os
import random
from pathlib import Path

DEV = ('cu/024257c555c6d20e', 'cu/00db092ac5a3c7bc',
       'cm/009e412e4d25ec57', 'cm/05447e326032ca20')
LAYOUTS = {
    'full': 'Original chronological explanation and HISTORY/CURRENT EDIT labels',
    'bare': 'All original images/instructions interleaved; no added words or labels',
    'edit_labels': 'All original history; each instruction prefixed with Edit: ',
    'last_image': 'Bare first turn; current instruction prefixed with Edit the last image. on later turns',
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def prepare(args):
    from lance_mice.settings import Settings
    rng = random.Random(20261005)
    holdout = []
    for split in ('cm', 'cu'):
        population = sorted(p.relative_to(args.run).as_posix()
                            for p in (args.run / split).iterdir() if p.is_dir())
        holdout.extend(sorted(rng.sample([s for s in population if s not in DEV], 6)))
    plan = {
        'version': 1, 'source_run': str(args.run.resolve()), 'layouts': LAYOUTS,
        'development': list(DEV), 'holdout': holdout,
        'sampling': 'Random(20261005), sorted CM then CU, six per split excluding development',
        'settings': Settings(cache_mode='none').identity(),
        'stages': {
            'development': {'seed': 42, 'modes': list(LAYOUTS), 'sessions': list(DEV)},
            'robustness': {'seed': 43, 'modes': ['full', 'bare'], 'sessions': list(DEV)},
            'holdout': {'seed': 42, 'modes': ['bare'], 'sessions': holdout},
        },
        'selection': 'Bare is predeclared primary candidate; other layouts are diagnostic comparators. Holdout not used for tuning.',
        'review': 'Assistant visual inspection: unrequested new lettering separately from edit success and constraint/reference retention. Not formal benchmark scoring.',
        'controls': 'Original instruction bytes, source pixels, settings and per-session RNG fixed; each arm uses its own saved outputs for every later turn. No failed-history replacement.',
        'code_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__).relative_to(Path.cwd()), Path('src/lance_mice/text_artifact_probe.py'),
                         Path('src/lance_mice/protocol.py'), Path('src/lance_mice/omni_pipeline.py')]},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


def validate_trace(meta, images, instructions):
    from lance_mice.images import image_hash
    assert meta['image_hashes'] == [image_hash(im) for im in images]
    pos, neg = meta['positive_trace'], meta['negative_trace']
    for trace in (pos, neg):
        assert [s['image_index'] for s in trace if s['kind'] == 'vae'] == list(range(len(images)))
        assert [s['image_index'] for s in trace if s['kind'] == 'vit'] == list(range(len(images)))
    assert [s['text'] for s in pos if s.get('role') in ('current', 'history')] == instructions
    assert [s['text'] for s in neg if s.get('role') in ('current', 'history')] == instructions[:-1]
    assert [s for s in pos if s.get('role') != 'current'] == neg


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from PIL import Image
    from lance_mice.backend import OmniBackend, find_history_metadata, history_payload
    from lance_mice.images import image_hash
    from lance_mice.protocol import derive_seed
    from lance_mice.settings import Settings
    plan = read(args.output / 'plan.json')
    for path, expected in plan['code_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    spec = plan['stages'][args.stage]
    settings = Settings(seed=spec['seed'], cache_mode='none')
    destination = args.output / args.stage
    destination.mkdir(exist_ok=False)
    records = []
    write(destination / 'status.json', {'status': 'starting', 'completed': 0})
    backend = None
    try:
        backend = OmniBackend(Path('/home/chs/model/Lance'), settings,
                              pipeline_class='lance_mice.text_artifact_probe.TextArtifactProbePipeline')
        for sid in spec['sessions']:
            original = Path(plan['source_run']) / sid
            session = read(original / 'session.json')
            with Image.open(original / 'turn_0_input.png') as im:
                source = im.convert('RGB')
            assert image_hash(source) == session['source_hash']
            assert len(session['instructions']) == 3
            for mode in spec['modes']:
                arm = destination / sid / mode
                arm.mkdir(parents=True)
                images = [source]
                for turn in range(1, 4):
                    instructions = session['instructions'][:turn]
                    payload = history_payload(sid, images, instructions, settings, audit=True, end_session=turn == 3)
                    payload['extra_args']['lance_history']['text_probe_mode'] = mode
                    params = copy.deepcopy(backend.engine.default_sampling_params_list)
                    params[0].num_inference_steps = settings.steps
                    params[0].seed = derive_seed(settings.seed, sid, 'noise', turn)
                    result = list(backend.engine.generate(payload, sampling_params_list=params, use_tqdm=False))
                    assert len(result) == 1 and len(result[0].images or []) == 1
                    output = result[0].images[0].convert('RGB')
                    meta = find_history_metadata(result[0])
                    validate_trace(meta, images, instructions)
                    output.save(arm / f'turn_{turn}.png')
                    with Image.open(arm / f'turn_{turn}.png') as im:
                        saved = im.convert('RGB')
                    assert image_hash(saved) == image_hash(output)
                    record = {'session_id': sid, 'mode': mode, 'seed': spec['seed'], 'turn': turn,
                              'instructions': instructions, 'output_hash': image_hash(saved), 'backend': meta,
                              'history_chain_and_cfg_verified': True}
                    if mode == 'full' and spec['seed'] == 42:
                        with Image.open(original / f'turn_{turn}.png') as im:
                            record['matches_formal_pixels'] = image_hash(im) == image_hash(saved)
                        assert record['matches_formal_pixels'], (sid, turn, 'baseline mismatch')
                    write(arm / f'turn_{turn}.json', record)
                    records.append(record)
                    images.append(saved)
                    write(destination / 'results.json', records)
                    write(destination / 'status.json', {'status': 'running', 'completed': len(records),
                                                       'total': len(spec['sessions']) * len(spec['modes']) * 3})
                    print(f'DONE {args.stage} {sid} {mode} T{turn}', flush=True)
        write(destination / 'status.json', {'status': 'completed', 'completed': len(records),
                                           'all_history_and_cfg_checks': True})
    except Exception as exc:
        write(destination / 'status.json', {'status': 'failed', 'completed': len(records), 'error': repr(exc)})
        raise
    finally:
        if backend is not None:
            backend.close()


def render(args):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    plan = read(args.output / 'plan.json')
    gallery = args.output / 'gallery'
    gallery.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=18)
    sections = []
    for stage, spec in plan['stages'].items():
        for sid in spec['sessions']:
            original = Path(plan['source_run']) / sid
            instructions = read(original / 'session.json')['instructions']
            rows = [('Source / formal seed 42', [original / 'turn_0_input.png'] +
                     [original / f'turn_{t}.png' for t in range(1, 4)])]
            for mode in spec['modes']:
                paths = [original / 'turn_0_input.png'] + [args.output / stage / sid / mode / f'turn_{t}.png' for t in range(1, 4)]
                if all(p.exists() for p in paths):
                    rows.append((f'{mode} / seed {spec["seed"]}', paths))
            if len(rows) == 1:
                continue
            # Keep full-size individual images on the server; review sheets preserve aspect ratio.
            cellw, cellh, titleh = 400, 440, 30
            sheet = Image.new('RGB', (cellw * 4, (cellh + titleh) * len(rows)), '#eeeeee')
            draw = ImageDraw.Draw(sheet)
            for r, (label, paths) in enumerate(rows):
                y = r * (cellh + titleh)
                draw.text((8, y + 4), label, font=font, fill='black')
                for t, path in enumerate(paths):
                    with Image.open(path) as im:
                        thumb = ImageOps.contain(im.convert('RGB'), (cellw - 8, cellh - 8))
                    sheet.paste(thumb, (t * cellw + (cellw - thumb.width)//2, y + titleh + (cellh-thumb.height)//2))
            name = f'{stage}_{sid.replace("/", "_")}.jpg'
            sheet.save(gallery / name, quality=93)
            prompt_list = ''.join(f'<li>{html.escape(t)}</li>' for t in instructions)
            sections.append(f'<section id="{stage}_{sid.replace("/", "_")}"><h2>{stage}: {sid}</h2><ol>{prompt_list}</ol><a href="gallery/{name}"><img loading="lazy" src="gallery/{name}"></a></section>')
    page = '''<!doctype html><meta charset="utf-8"><title>Lance prompt layout study</title>
<style>body{font:17px system-ui;margin:30px auto;max-width:1640px;background:#eee;color:#222}section{background:white;padding:20px;margin:30px 0}img{width:100%}li{margin:8px}h1{font-size:28px}</style>
<h1>Lance 提示词编排对照</h1><p>每行从左到右：原图、第一轮、第二轮、第三轮。full 为现有封装，bare 为原始图文交错，edit_labels 为简短标签，last_image 为后续轮次的简短说明。英文为原始指令。点击图片查看大图。</p>
<p>开发集与第二种子为定向问题样本；留出集提前随机抽取。检查乱码与编辑效果时请分开判断。这是探索实验，不是正式能力评分。</p>'''
    (args.output / 'index.html').write_text(page + '\n'.join(sections))
    print(json.dumps({'sheets': len(sections), 'gallery': str(gallery)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'render'))
    parser.add_argument('--run', type=Path, default=Path('outputs/mice/full_20261004_attention/run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('development', 'robustness', 'holdout'))
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.command == 'run' and not args.stage:
        parser.error('--stage required for run')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
