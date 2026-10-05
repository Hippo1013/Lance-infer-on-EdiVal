"""CPU-only protocol, identity, typed missing values and aggregation."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
from .if_rules import parse_instruction
from .prompts import MULTITURN_VLM_PROMPT, MULTITURN_CU_VLM_PROMPT

PROTOCOL = 'mice-dual-judge-v2'
JUDGES = {'qwen': 'Qwen3.6-27B', 'gemma': 'gemma-4-31B-it'}
PURE_IF = {'subject_remove', 'position_change', 'count_change'}
TASKS = PURE_IF | {'subject_add', 'subject_replace', 'color_alter', 'material_alter', 'text_change', 'background_change'}
ROOT = Path(__file__).resolve().parents[3]
SETTINGS = {'temperature': 0.6, 'seed': 42, 'max_tokens': 2048,
            'max_model_len': 8192, 'dtype': 'bfloat16', 'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.88, 'max_num_seqs': 1, 'enable_thinking': False,
            'image_encoding': 'lossless PNG, original RGB resolution', 'votes_per_judge': 2,
            'vote_seed_rule': 'base seed + one-based vote index - 1',
            'cc_input': 'timm default evaluation geometry with identical mask geometry',
            'cc_no_component': 'null; upstream missing-value 0.5 retained separately',
            'invalid_if': 'null; upstream invalid-format 0 retained separately',
            'ga': 'mean of votes per prefix, then cumulative minimum per judge'}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def sampling_seed(vote):
    if type(vote) is not int or not 1 <= vote <= SETTINGS['votes_per_judge']:
        raise ValueError('Invalid vote index')
    return SETTINGS['seed'] + vote - 1

def request_identity(image_hashes, prompt, vote):
    return digest([image_hashes, prompt, SETTINGS, vote, sampling_seed(vote)])

def vote_mean(rows):
    if len(rows) != SETTINGS['votes_per_judge']:
        raise ValueError('Every configured vote is required')
    if any(row['status'] != 'ok' for row in rows):
        return result('unavailable', reason='Every vote must have a valid score',
                      components=[row['status'] for row in rows])
    return result('ok', sum(row['score'] for row in rows) / len(rows))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # This is an atomic durable output transaction, not a scratch artifact.
    pending = path.with_suffix(path.suffix + '.part')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(pending, path)

def source_pins():
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT/'configs/scoring_models.json', ROOT/'scripts/score_mice.py', ROOT/'scripts/run_mice_scoring.sh']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}

def result(status, value=None, **details):
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError('Nonfinite score')
    if status != 'ok' and value is not None: raise ValueError('Only ok results have numeric scores')
    return {'status': status, 'score': value, **details}

def parse_yes_no(text):
    # Extract explicit decisions, never a yes/no substring buried in reasoning.
    # Different chat templates may put the decision before or after an explanation.
    lines = [re.sub(r"[*`]", "", line).strip() for line in text.splitlines() if line.strip()]
    answers = set()
    token = r"[\"'“”‘’]?(yes|no)[\"'“”‘’]?"
    patterns = [
        rf"^{token}[.!]?\s*$",
        rf"^{token}[,.!:]\s+\S",  # 'No, the requested object is absent.'
        rf"^(?:(?:therefore|thus|so|hence)[,:]?\s+)?(?:(?:the|my)\s+)?"
        rf"(?:(?:final|correct)\s+)?(?:answer|response)\s*(?:is\s+|:\s*){token}[.!]?\s*$",
        rf"^(?:therefore|thus|so|hence)[,:]?\s+{token}[.!]?\s*$",
    ]
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.I)
            if match: answers.add(match[1].lower())
    if len(answers) != 1:
        raise ValueError('Expected one consistent explicit YES/NO decision')
    return answers.pop()

def parse_ga(text, turns):
    final = re.findall(r'<answer_final>\s*(yes|no)\s*</answer_final>', text, re.I)
    answers = re.findall(r'<answer_turn_(\d+)>\s*(yes|no)\s*</answer_turn_\1>', text, re.I)
    if len(final) != 1 or not answers: raise ValueError('Missing/duplicate GA answer tags')
    ids = [int(i) for i, _ in answers]; values = [v.lower() for _, v in answers]
    if ids != list(range(1, len(ids)+1)) or len(ids) > turns:
        raise ValueError('GA turn sequence invalid')
    if final[0].lower() == 'yes':
        if len(ids) != turns or any(v != 'yes' for v in values): raise ValueError('Inconsistent GA yes')
    elif values[-1] != 'no' or any(v != 'yes' for v in values[:-1]):
        raise ValueError('GA must stop at its first failure')
    return float(final[0].lower() == 'yes')

def ga_prompt(split, instructions, formatted):
    if split == 'cm':
        text = '\n'.join(f'第{i+1}轮：{v}' for i,v in enumerate(instructions))
        return MULTITURN_VLM_PROMPT.format(instructions_formatted=text)
    text = '\n'.join(f'第{i+1}轮：\n  - 模型接收的指令：{v}\n  - 显式参照指令：{f}' for i,(v,f) in enumerate(zip(instructions, formatted)))
    return MULTITURN_CU_VLM_PROMPT.format(instructions_formatted=text)

def valid_if(task, formatted):
    if task not in TASKS: return False
    parts = parse_instruction(task, formatted)
    if parts is None: return False
    if task == 'count_change' and not parts[1].isdigit(): return False
    if task == 'position_change' or (task == 'subject_add' and len(parts) == 3):
        if parts[1] not in {'left','right','above','below'}: return False
    return True

def pair_mean(a, b):
    if a['status'] != 'ok' or b['status'] != 'ok':
        return result('unavailable', reason='Both judges must have valid scores', components=[a['status'], b['status']])
    return result('ok', (a['score'] + b['score']) / 2)

def cumulative(rows):
    out=[]; previous=1.0; blocked=False
    for row in rows:
        if row['status'] != 'ok': blocked=True
        if blocked: out.append(result('unavailable', reason='A prefix is unresolved'))
        else:
            previous=min(previous,row['score']); out.append(result('ok',previous))
    return out

def mean_report(rows):
    values=[x['score'] for x in rows if x['status']=='ok']
    counts={s:sum(x['status']==s for x in rows) for s in sorted({x['status'] for x in rows})}
    return {'mean': sum(values)/len(values) if values else None, 'valid':len(values), 'total':len(rows), 'statuses':counts}
