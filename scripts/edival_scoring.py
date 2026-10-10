#!/usr/bin/env python3
"""Pinned official EdiVal multipass scoring of existing Lance images.

No generation, selective retries, alternate prompts, or score imputation.
Separate process environments run IF, CC/RAHF, and official HPS backfill.
"""
import argparse
import ast
import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

REVISION = '96d34b00d7ea2dc3de90f2bb01f292f6dec6294a'
PROTOCOL = 'edival-official-multipass-v1'
STAGES = ('if', 'metrics', 'hps')
ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite score')
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v)


def pixel_sha(image):
    image = image.convert('RGB')
    return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode() + image.tobytes()).hexdigest()


def official_namespace(source, names, extra=None):
    """Execute unchanged selected top-level definitions without unused heavy imports."""
    namespace = dict(ast=ast, os=os, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple,
                     dataclass=dataclass)
    namespace.update(extra or {})
    tree = ast.parse((source / 'eval.py').read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source / 'eval.py'), 'exec'), namespace)
    if not set(names).issubset(namespace):
        raise ValueError('Missing official definitions')
    return namespace


def verify_official(source):
    manifest = read(source / 'source.json')
    if manifest['revision'] != REVISION:
        raise ValueError('Unexpected official source revision')
    for name, expected in manifest['files'].items():
        if sha(source / name) != expected:
            raise ValueError(f'Official source changed: {name}')
    return manifest


def check_record(record):
    for item in record['images'] + record['turn_records'] + [record['session_record']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Input changed: {item["path"]}')


def stage_path(output, stage, index):
    return output / 'stages' / stage / f'{index}.json'


def checked_stage(output, stage, record):
    data = read(stage_path(output, stage, record['index']))
    if (data['status'] != 'passed' or data['stage'] != stage
            or data['input_identity'] != identity(record)):
        raise ValueError(f'Stage identity/status mismatch: {stage}/{record["index"]}')
    finite(data)
    return data


def prepare(args):
    """CPU-only release, RGB chain, annotation and frozen-source preflight."""
    from PIL import Image
    import pandas as pd

    output = args.output.resolve()
    if output.exists():
        raise ValueError('New scoring output required; existing outputs are never overwritten')
    official = verify_official(args.official_source)
    setup = read(args.setup / 'summary.json')
    if setup['status'] != 'ready' or setup['resource_count'] != 8:
        raise ValueError('Accepted scoring environment required')
    inference = args.inference.resolve()
    validation = read(inference / 'validation.json')
    completion = read(inference / 'completion.json')
    if (validation['status'] != 'passed' or validation['sessions'] != 572
            or validation['turns'] != 1716 or completion['state'] != 'completed'
            or completion['exit_code'] != 0):
        raise ValueError('Complete validated 572-session inference required')
    dataset_pins = read(ROOT / 'configs/edival_dataset.json')
    for name, expected in dataset_pins['sha256'].items():
        if sha(args.dataset / name) != expected:
            raise ValueError(f'Dataset identity differs: {name}')
    csv_path = args.dataset / 'oai_instruction_generation_output.csv'
    df = pd.read_csv(csv_path)
    if len(df) != 1716 or df.duplicated(['image_index', 'turns']).any():
        raise ValueError('Duplicate/missing official rows')
    parser = official_namespace(args.official_source, ['parse_multipass_instruction_data'], dict(pd=pd))
    ids = sorted(int(x) for x in df.image_index.unique())
    if len(ids) != 572:
        raise ValueError('Expected 572 sessions')
    selected = ids if not args.indices else [int(x) for x in args.indices.split(',')]
    if len(set(selected)) != len(selected) or not set(selected).issubset(ids):
        raise ValueError('Invalid session selection')
    run = read(inference / 'run/run.json')
    sessions = []
    with ZipFile(args.dataset / 'input_images_resize_512.zip') as archive:
        for index in selected:
            folder = inference / 'run/edival' / str(index)
            spec = read(folder / 'session.json')
            source = Image.open(io.BytesIO(archive.read(f'{index}_input_raw.jpg'))).convert('RGB')
            if source.size != (512, 512) or pixel_sha(source) != spec['source_hash']:
                raise ValueError('Canonical source pixels differ')
            images = []
            for turn in range(4):
                path = folder / ('turn_0_input.png' if turn == 0 else f'turn_{turn}.png')
                with Image.open(path) as image:
                    if image.size != (512, 512):
                        raise ValueError('Input image resolution changed')
                    pixel = pixel_sha(image)
                expected = spec['source_hash'] if turn == 0 else read(folder / f'turn_{turn}.json')['output_hash']
                if pixel != expected:
                    raise ValueError('Saved image pixels differ from inference record')
                images.append(dict(path=str(path), sha256=sha(path), pixel_sha256=pixel))
            turns, records = [], []
            for turn in range(1, 4):
                row = df[(df.image_index == index) & (df.turns == turn)]
                if len(row) != 1:
                    raise ValueError('Missing official prefix row')
                parsed = parser['parse_multipass_instruction_data'](row.iloc[0], index, turn)
                if parsed is None or not isinstance(row.iloc[0]['bg_consistency'], (bool, type(df.bg_consistency.iloc[0]))):
                    raise ValueError('Invalid official annotation')
                path = folder / f'turn_{turn}.json'
                record = read(path)
                if (record['instruction'] != parsed['instruction']
                        or record['session_fingerprint'] != spec['fingerprint']
                        or record['input_hashes'] != [x['pixel_sha256'] for x in images[:turn]]):
                    raise ValueError('Actual instruction/history differs from scored row')
                turns.append(parsed)
                records.append(dict(path=str(path), sha256=sha(path)))
            if spec['instructions'] != [x['instruction'] for x in turns]:
                raise ValueError('Session instruction list differs')
            sessions.append(dict(index=index, turns=turns, images=images, turn_records=records,
                                 session_record=dict(path=str(folder / 'session.json'), sha256=sha(folder / 'session.json'))))
    manifest = dict(protocol=PROTOCOL, mode='multipass', created_at=now(), inference=str(inference),
                    run_fingerprint=run['fingerprint'], official_source=str(args.official_source),
                    official_revision=REVISION, official_files=official['files'],
                    model_root=str(args.model_root), setup=str(args.setup), setup_sha256=sha(args.setup / 'summary.json'),
                    dataset=str(args.dataset), dataset_sha256=dataset_pins['sha256'],
                    inference_validation_sha256=sha(inference / 'validation.json'),
                    inference_completion_sha256=sha(inference / 'completion.json'),
                    sessions=sessions, session_count=len(sessions), turn_count=3*len(sessions),
                    shards=[[r['index'] for r in sessions[i::args.num_shards]] for i in range(args.num_shards)],
                    runtime_files={str(p.relative_to(ROOT)): sha(p) for p in
                        [ROOT / 'scripts/edival_scoring.py', ROOT / 'scripts/full_edival_scoring_job.py',
                         ROOT / 'configs/edival_dataset.json', ROOT / 'configs/edival_scoring_models.json']},
                    hps_protocol='official update_hps_scores.py backfill: empty prompt, first raw dimension, no rounding',
                    sampling=dict(temperature=0.0, max_tokens=1024, min_tokens=1, object_name_max_tokens=20),
                    runtime=dict(tensor_parallel_size=1, max_model_len=2048, image_limit=2,
                                 gpu_memory_utilization=0.5, enforce_eager=True, seed=0,
                                 VLLM_USE_V2_MODEL_RUNNER='0'))
    output.mkdir(parents=True)
    write(output / 'manifest.json', manifest)
    write(output / 'preflight.json', dict(status='passed', sessions=len(sessions), turns=len(sessions)*3,
                                          images=len(sessions)*4, manifest_sha256=sha(output / 'manifest.json')))
    if args.seed_from:
        seed_manifest = read(args.seed_from / 'manifest.json')
        if (read(args.seed_from / 'validation.json')['status'] != 'passed'
                or seed_manifest['runtime_files'] != manifest['runtime_files']
                or seed_manifest['official_files'] != manifest['official_files']
                or seed_manifest['setup_sha256'] != manifest['setup_sha256']):
            raise ValueError('Accepted seed differs from frozen configuration')
        seeded = []
        for record in sessions:
            if record['index'] not in {r['index'] for r in seed_manifest['sessions']}:
                continue
            for stage in STAGES:
                result = checked_stage(args.seed_from, stage, record)
                write(stage_path(output, stage, record['index']), result)
                seeded.append(dict(stage=stage, index=record['index'],
                                   source=str(stage_path(args.seed_from, stage, record['index'])),
                                   sha256=sha(stage_path(args.seed_from, stage, record['index']))))
        write(output / 'seed-reuse.json', dict(records=seeded, policy='reuse accepted first observations without resampling'))
    print(json.dumps(dict(status='passed', sessions=len(sessions), turns=3*len(sessions), shards=manifest['shards'])))


class Trace:
    """Capture the actual messages/responses; preserve upstream decision functions."""
    def __init__(self, llm, output):
        self.llm, self.calls, self.errors = llm, [], []
        self.output, self.index, self.turn = output, None, None

    def chat(self, **kwargs):
        messages = kwargs['messages']
        serial = []
        for message in messages:
            content = message['content']
            if isinstance(content, list):
                converted = []
                for x in content:
                    if x['type'] == 'text':
                        converted.append(dict(type='text', text=x['text']))
                    else:
                        url = x['image_url']['url']
                        raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                        image_sha = hashlib.sha256(raw).hexdigest()
                        asset = self.output / 'evidence/if_images' / (image_sha + '.jpg')
                        asset.parent.mkdir(parents=True, exist_ok=True)
                        if not asset.exists():
                            with asset.open('xb') as stream:
                                stream.write(raw)
                        elif sha(asset) != image_sha:
                            raise ValueError('Judge evidence image identity differs')
                        converted.append(dict(type='image_url', jpeg_sha256=image_sha, evidence_path=str(asset),
                                              url_sha256=hashlib.sha256(url.encode()).hexdigest(), encoding='upstream JPEG base64'))
                content = converted
            serial.append(dict(role=message['role'], content=content))
        params = kwargs['sampling_params']
        call = dict(messages=serial, sampling=dict(temperature=params.temperature, max_tokens=params.max_tokens,
                                                   min_tokens=params.min_tokens), status='running')
        self.calls.append(call)
        try:
            outputs = self.llm.chat(**kwargs)
            if not outputs or not outputs[0].outputs or not outputs[0].outputs[0].text.strip():
                raise RuntimeError('Empty actual judge response')
            result = outputs[0]
            call.update(status='passed', response=result.outputs[0].text,
                        prompt_token_ids=result.prompt_token_ids, output_token_ids=result.outputs[0].token_ids,
                        finish_reason=result.outputs[0].finish_reason)
            return outputs
        except Exception as exc:
            call.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            self.errors.append(call['error'])
            raise
        finally:
            call.update(index=self.index, turn=self.turn)
            write(self.output / 'evidence/if_calls' / str(self.index) / f'{len(self.calls):03d}.json', call)


def local_loader(cls, paths):
    original = cls.from_pretrained
    @classmethod
    def loader(_cls, name, *a, **kw):
        return original(str(paths.get(str(name), name)), *a, **kw)
    cls.from_pretrained = loader


class GroundingClient:
    """Exact upstream predict in the compatible environment on the same GPU."""
    def __init__(self, output, shard, gpu):
        from multiprocessing.connection import Client
        self.output, self.shard = output, shard
        self.socket = Path(os.environ['TMPDIR']) / 'grounding.sock'
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.0/compat:/home/chs/conda/envs/EdiVal/lib'
        command = ['/home/chs/conda/envs/EdiVal/bin/python', '-B', __file__,
                   'grounding-service', '--output', str(output), '--shard', str(shard), '--gpu', str(gpu)]
        self.process = subprocess.Popen(command, env=env)
        self.connection = None
        try:
            for _ in range(180):
                if self.process.poll() is not None:
                    raise RuntimeError(f'Grounding service exited {self.process.returncode}')
                if self.socket.exists():
                    self.connection = Client(str(self.socket), family='AF_UNIX')
                    ready = self.connection.recv()
                    if ready['status'] != 'ready':
                        raise RuntimeError(str(ready))
                    return
                time.sleep(1)
            raise RuntimeError('Grounding service startup timed out')
        except BaseException:
            self.close()
            raise

    def predict(self, model, image, caption, box_threshold, text_threshold, **kwargs):
        import torch
        request = dict(image=image.detach().cpu().numpy(), caption=caption,
                       box_threshold=box_threshold, text_threshold=text_threshold, **kwargs)
        self.connection.send(request)
        response = self.connection.recv()
        if response['status'] != 'passed':
            raise RuntimeError(response['error'])
        return torch.from_numpy(response['boxes']), torch.from_numpy(response['logits']), response['phrases']

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send({'stop': True})
            except (EOFError, BrokenPipeError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=20)


def grounding_service(args):
    import torch
    from multiprocessing.connection import Listener
    from groundingdino.util.inference import load_model, predict
    manifest = read(args.output / 'manifest.json')
    verify_official(Path(manifest['official_source']))
    if os.environ['CUDA_VISIBLE_DEVICES'] != str(args.gpu if args.gpu is not None else args.shard) or torch.cuda.device_count() != 1:
        raise ValueError('Grounding service must share exactly its assigned physical GPU')
    models = Path(manifest['model_root'])
    model = load_model(str(models / 'GroundingDINO-SwinT-OGC/GroundingDINO_SwinT_OGC.local.py'),
                       str(models / 'GroundingDINO-SwinT-OGC/groundingdino_swint_ogc.pth'), device='cuda')
    socket = Path(os.environ['TMPDIR']) / 'grounding.sock'
    with Listener(str(socket), family='AF_UNIX') as listener:
        with listener.accept() as connection:
            connection.send(dict(status='ready', pid=os.getpid(), gpu=args.shard))
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                if request.pop('stop', False):
                    break
                try:
                    request['image'] = torch.from_numpy(request['image'])
                    boxes, logits, phrases = predict(model=model, **request)
                    connection.send(dict(status='passed', boxes=boxes.detach().cpu().numpy(),
                                         logits=logits.detach().cpu().numpy(), phrases=phrases))
                except Exception as exc:
                    connection.send(dict(status='failed', error=f'{type(exc).__name__}: {exc}'))


def guarded(function, errors, label):
    def call(*a, **kw):
        try:
            return function(*a, **kw)
        except Exception as exc:
            errors.append(f'{label}: {type(exc).__name__}: {exc}')
            raise
    return call


class OfficialOutput:
    """Notice upstream caught errors without changing any valid scoring path."""
    def __init__(self, original, errors):
        self.original, self.errors, self.pending = original, errors, ''
    def write(self, text):
        self.original.write(text)
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if line.startswith(('Error querying VLM', 'Error in object detection',
                                'Warning: DINOv3 masked background similarity failed')):
                self.errors.append(line)
        return len(text)
    def flush(self):
        self.original.flush()
    def __getattr__(self, key):
        return getattr(self.original, key)


def paths_for(record, turn, cls):
    images = record['images']
    return cls(base=images[0]['path'], source=images[turn-1]['path'], target=images[turn]['path'])


def worker(args):
    import torch
    manifest = read(args.output / 'manifest.json')
    source, models = Path(manifest['official_source']), Path(manifest['model_root'])
    verify_official(source)
    for name, expected in manifest['runtime_files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen runtime changed: {name}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu if args.gpu is not None else args.shard) or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly the assigned physical GPU')
    sys.path.insert(0, str(source))
    records = [r for r in manifest['sessions'] if r['index'] in manifest['shards'][args.shard]]
    progress = dict(stage=args.stage, gpu=args.shard, state='loading', pid=os.getpid(), started_at=now(),
                    completed_sessions=0, completed_turns=0, total_sessions=len(records), errors=[])
    status_path = args.output / f'worker_{args.shard}_{args.stage}.json'
    write(status_path, progress)
    errors, trace, grounding_client = [], None, None
    original_stdout = sys.stdout
    try:
        if args.stage == 'if':
            from vllm import LLM
            from detector import instruction_detector as inst
            grounding_client = GroundingClient(args.output, args.shard, args.gpu if args.gpu is not None else args.shard)
            gd = grounding_client
            inst.predict = guarded(grounding_client.predict, errors, 'GroundingDINO predict')
            trace = Trace(LLM(model=str(models / 'Qwen2-VL-7B-Instruct'), trust_remote_code=True,
                             max_model_len=2048, limit_mm_per_prompt={'image': 2}, gpu_memory_utilization=0.5,
                             tensor_parallel_size=1, dtype='bfloat16', enforce_eager=True, seed=0), args.output)
        elif args.stage == 'metrics':
            from transformers import AutoImageProcessor, AutoModel
            from detector import quality_detector as vq, consistency_detector as cc
            from detector.utils import _make_json_serializable
            from groundingdino.util.inference import load_model
            mapping = {'facebook/dinov3-vitb16-pretrain-lvd1689m': models / 'DINOv3-ViT-B-16',
                       'google/vit-large-patch16-384': models / 'vit-large-patch16-384'}
            local_loader(AutoImageProcessor, mapping)
            local_loader(AutoModel, mapping)
            gd = load_model(str(models / 'GroundingDINO-SwinT-OGC/GroundingDINO_SwinT_OGC.local.py'),
                            str(models / 'GroundingDINO-SwinT-OGC/groundingdino_swint_ogc.pth'), device='cuda')
            cc.predict = guarded(cc.predict, errors, 'GroundingDINO predict')
            dino = cc.load_consistency_model(device='cuda')
            dino[0].forward = guarded(dino[0].forward, errors, 'DINOv3 forward')
            quality = vq.RAHF(vit_model=str(models / 'vit-large-patch16-384'), t5_model=str(models / 't5-base'))
            quality.load_state_dict(torch.load(models / 'RAHF/rahf_model.pt', map_location='cpu', weights_only=True), strict=True)
            quality.eval().to('cuda')
            vq.load_human_preference_inferencer = lambda device=None: None
            names = ['EvaluationModels', 'ImagePaths', 'evaluate_turn_multipass', 'evaluate_base_image_quality']
            ns = official_namespace(source, names, dict(evaluate_quality=vq.evaluate_quality, evaluate_consistency=cc.evaluate_consistency))
            bundle = ns['EvaluationModels'](gd, None, quality, dino, None)
        else:
            import yaml
            import hpsv3
            from hpsv3 import HPSv3RewardInferencer
            config_path = Path(hpsv3.__file__).parent / 'config/HPSv3_7B.yaml'
            config = yaml.safe_load(config_path.read_text())
            config['model_name_or_path'] = str(models / 'Qwen2-VL-7B-Instruct')
            config['output_dir'] = str(Path(os.environ['TMPDIR']) / 'hps-output')
            local_config = Path(os.environ['TMPDIR']) / 'hps-local.yaml'
            local_config.write_text(yaml.safe_dump(config))
            infer = HPSv3RewardInferencer(config_path=str(local_config),
                                        checkpoint_path=str(models / 'HPSv3/HPSv3.safetensors'), device='cuda:0')
        progress['state'] = 'running'
        sys.stdout = OfficialOutput(original_stdout, errors)
        for record in records:
            index = record['index']
            check_record(record)
            path = stage_path(args.output, args.stage, index)
            if path.exists():
                checked_stage(args.output, args.stage, record)
            else:
                result = dict(status='running', stage=args.stage, index=index, gpu=args.shard, started_at=now(),
                              input_identity=identity(record), manifest_sha256=sha(args.output / 'manifest.json'))
                if args.stage == 'if':
                    trace.calls, trace.errors = [], []
                    trace.index = index
                    turn_results = {}
                    for turn, annotation in enumerate(record['turns'], 1):
                        trace.turn = turn
                        start = len(trace.calls)
                        score, reason = inst.evaluate_instruction_following(
                            record['images'][turn-1]['path'], record['images'][turn]['path'],
                            formatted_instruction=annotation['format_instruction'], instruction=annotation['instruction'],
                            task_type=annotation['task_type'], grounding_model=gd, vlm_model=trace, output_reason=True)
                        turn_results[str(turn)] = dict(score=score, reason=reason, call_indices=list(range(start, len(trace.calls))))
                        if score not in (0, 1):
                            raise ValueError('Unexpected official IF score')
                    result.update(turns=turn_results, calls=trace.calls)
                    errors.extend(trace.errors)
                elif args.stage == 'metrics':
                    observed = checked_stage(args.output, 'if', record)
                    turn_results = {}
                    with torch.inference_mode():
                        base = ns['evaluate_base_image_quality'](record['images'][0]['path'], quality, None)
                        for turn, annotation in enumerate(record['turns'], 1):
                            old = observed['turns'][str(turn)]
                            ns['evaluate_instruction_following'] = lambda *a, _old=old, **kw: (_old['score'], _old['reason'])
                            turn_results[str(turn)] = ns['evaluate_turn_multipass'](
                                bundle, paths_for(record, turn, ns['ImagePaths']), annotation, turn, index)
                    result.update(base_image_quality=_make_json_serializable(base),
                                  turns=_make_json_serializable(turn_results), if_sha256=sha(stage_path(args.output, 'if', index)))
                else:
                    checked_stage(args.output, 'metrics', record)
                    image_paths = [x['path'] for x in record['images']]
                    with torch.inference_mode():
                        rewards = infer.reward(prompts=['']*4, image_paths=image_paths)
                    if rewards.shape != (4, 2) or not torch.isfinite(rewards).all():
                        raise ValueError('Invalid official HPS rewards')
                    raw = rewards.detach().float().cpu().tolist()
                    result.update(raw_rewards=raw, scores=[float(x[0]) for x in raw], prompts=['']*4,
                                  original_config_sha256=sha(config_path),
                                  metrics_sha256=sha(stage_path(args.output, 'metrics', index)))
                if errors:
                    result.update(status='failed', errors=errors, calls=trace.calls if trace else [])
                    write(args.output / 'failures' / f'{args.stage}_{index}.json', result)
                    raise RuntimeError('Upstream swallowed a component failure; preserved as technical failure')
                finite(result)
                result.update(status='passed', finished_at=now())
                write(path, result)
            progress.update(completed_sessions=progress['completed_sessions']+1,
                            completed_turns=progress['completed_turns']+3, current_index=index, updated_at=now())
            write(status_path, progress)
        progress.update(state='completed', finished_at=now(), peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        progress.update(state='failed', error=f'{type(exc).__name__}: {exc}', errors=errors,
                        current_calls=trace.calls if trace else [], finished_at=now())
        raise
    finally:
        sys.stdout = original_stdout
        if grounding_client is not None:
            grounding_client.close()
        write(status_path, progress)


def aggregate(args):
    manifest = read(args.output / 'manifest.json')
    source = Path(manifest['official_source'])
    verify_official(source)
    ns = official_namespace(source, ['LOCAL_TASK_TYPES', 'GLOBAL_TASK_TYPES', 'CONSISTENCY_TASK_TYPES',
                                     'QUALITY_TASK_TYPES', 'TASK_TYPES', 'TaskRateCollector'])
    collector, image_rates, results = ns['TaskRateCollector'](3), {}, {}
    stage_hashes, nulls, calls = {}, {}, 0
    for stage in STAGES:
        actual = {p.stem for p in (args.output / 'stages' / stage).glob('*.json')}
        if actual != {str(r['index']) for r in manifest['sessions']}:
            raise ValueError(f'Incomplete/extra stage coverage: {stage}')
    for record in manifest['sessions']:
        check_record(record)
        index = record['index']
        observed = {s: checked_stage(args.output, s, record) for s in STAGES}
        if observed['metrics']['if_sha256'] != sha(stage_path(args.output, 'if', index)):
            raise ValueError('Metrics depend on different IF evidence')
        if observed['hps']['metrics_sha256'] != sha(stage_path(args.output, 'metrics', index)):
            raise ValueError('HPS depends on different metric evidence')
        calls += len(observed['if']['calls'])
        for call in observed['if']['calls']:
            # Accepted seed evidence intentionally remains in its original run.
            if call['status'] != 'passed' or not call['response'].strip():
                raise ValueError('Failed/empty judge evidence')
            for message in call['messages']:
                if isinstance(message['content'], list):
                    for item in message['content']:
                        if item['type'] == 'image_url' and sha(item['evidence_path']) != item['jpeg_sha256']:
                            raise ValueError('Actual judge image bytes differ')
        final = dict(base_image_quality=observed['metrics']['base_image_quality'])
        final['base_image_quality']['base_image_human_preference_score'] = observed['hps']['scores'][0]
        image_rates[index] = []
        for turn, annotation in enumerate(record['turns'], 1):
            row = observed['metrics']['turns'][str(turn)]
            if row['instruction_following'] != observed['if']['turns'][str(turn)]['score']:
                raise ValueError('IF score changed during merge')
            row['human_preference_score'] = observed['hps']['scores'][turn]
            if (row['meta']['base_image_path'] != record['images'][0]['path']
                    or row['meta']['src_image_path'] != record['images'][turn-1]['path']
                    or row['meta']['target_image_path'] != record['images'][turn]['path']
                    or row['meta']['instruction'] != annotation['instruction']):
                raise ValueError('Official source/target/meta mapping differs')
            for key in ns['QUALITY_TASK_TYPES']:
                if row[key] is None:
                    raise ValueError(f'Missing quality score: {key}')
            collector.add_score(turn, annotation['task_type'], row['instruction_following'])
            collector.add_quality_scores(turn, {k: row[k] for k in ns['QUALITY_TASK_TYPES']})
            if annotation['eval_bg_consistency']:
                collector.add_consistency_scores(turn, row['object_details'], row['bg_details'])
            elif any(row[k] is not None for k in ['object_details', 'bg_details']):
                raise ValueError('CC must be skipped on official background-change rows')
            for key in ['object_dinov3_consistency_mean', 'object_l1_consistency_mean',
                        'background_l1_consistency', 'background_dinov3_consistency']:
                if row[key] is None:
                    nulls[key] = nulls.get(key, 0) + 1
            final[str(turn)] = row
            image_rates[index].append(row['instruction_following'])
        results[index] = final
        for stage in STAGES:
            stage_hashes[f'{stage}/{index}'] = sha(stage_path(args.output, stage, index))
    if len(collector.data['overall']) != manifest['turn_count']:
        raise ValueError('IF aggregation coverage differs')
    destination = args.output / 'results/multipass'
    if destination.exists():
        raise ValueError('Final outputs already exist; refusing overwrite')
    for index, final in results.items():
        write(destination / f'{index}_input_raw.json', final)
    write(destination / 'task_rate.json', collector.data)
    write(destination / 'image_rate.json', image_rates)
    def describe(values):
        return dict(count=len(values), mean=sum(values)/len(values) if values else None)
    summary = dict(protocol=PROTOCOL, status='completed', sessions=manifest['session_count'],
                   turns=manifest['turn_count'], if_overall=describe(collector.data['overall']),
                   per_turn={str(t): {k: describe(v) for k, v in collector.data[t].items()} for t in range(1, 4)},
                   cc_null_counts=nulls, judge_calls=calls,
                   note='Official multipass metrics; Lance generated with full real history. No combined IF/CC/VQ score invented.')
    write(args.output / 'summary.json', summary)
    write(args.output / 'validation.json', dict(status='passed', protocol=PROTOCOL, sessions=len(results),
                turns=manifest['turn_count'], hps_images=4*len(results), judge_calls=calls,
                manifest_sha256=sha(args.output / 'manifest.json'), stage_sha256=stage_hashes,
                final_sha256={p.name: sha(p) for p in destination.glob('*.json')},
                cc_null_counts=nulls, checked_at=now(),
                checks=['exact stage coverage without duplicates', 'immutable source images and instruction records',
                        'canonical original->current CC and previous->current IF', 'original official aggregation class',
                        'all quality and HPS scores finite', 'IF/metrics/HPS evidence dependency hashes']))
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'worker', 'aggregate', 'grounding-service'])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inference', type=Path, default=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/edival/full_512_bare_20261005_attention'))
    ap.add_argument('--dataset', type=Path, default=Path('/home/chs/dataset/EdiVal'))
    ap.add_argument('--official-source', type=Path, default=Path('/home/chs/tools/EdiVal') / REVISION)
    ap.add_argument('--model-root', type=Path, default=Path('/home/chs/model'))
    ap.add_argument('--setup', type=Path, default=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/setup/edival_official_20261006'))
    ap.add_argument('--indices', help='Acceptance-only explicit session IDs; omit for full release')
    ap.add_argument('--seed-from', type=Path, help='Reuse passed first observations, never resample')
    ap.add_argument('--stage', choices=STAGES)
    ap.add_argument('--shard', type=int)
    ap.add_argument('--gpu', type=int, choices=[0, 1], help='Physical GPU, independent of logical session shard')
    ap.add_argument('--num-shards', type=int, default=2)
    args = ap.parse_args()
    if args.num_shards < 1 or (args.shard is not None and args.shard < 0):
        ap.error('Positive shard count and nonnegative shard index required')
    if args.action == 'worker' and (args.stage is None or args.shard is None):
        ap.error('worker requires stage and shard')
    {'prepare': prepare, 'worker': worker, 'aggregate': aggregate, 'grounding-service': grounding_service}[args.action](args)


if __name__ == '__main__':
    main()
