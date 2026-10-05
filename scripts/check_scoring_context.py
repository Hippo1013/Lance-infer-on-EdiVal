#!/usr/bin/env python3
"""Audit Qwen token capacity for every GA prefix; optionally probe the largest."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image
from lance_mice.scoring.core import ROOT,SETTINGS,read,write,ga_prompt,parse_ga,digest,source_pins
from lance_mice.scoring.runner import open_image

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',action='store_true');a=p.parse_args()
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained('/home/chs/model/Qwen3.6-27B',local_files_only=True)
    def count(sizes,prompt):
        messages=[{'role':'user','content':[{'type':'image'} for _ in sizes]+[{'type':'text','text':prompt}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        base=len(processor.tokenizer.encode(text,add_special_tokens=False))
        tokens=processor._get_num_multimodal_tokens(image_sizes=[(h,w) for w,h in sizes]).num_image_tokens
        return base+sum(n-1 for n in tokens)
    comparisons=[]
    existing=ROOT/'outputs/scoring/acceptance_20261004_v4/qwen'
    for path in existing.glob('*/*/turn_*.json'):
        for trace in read(path)['traces']:
            estimated=count(trace['image_sizes'],trace['prompt'])
            if estimated!=trace['prompt_tokens']:raise ValueError('Processor/vLLM token count mismatch')
            comparisons.append(estimated)
    if not comparisons:raise ValueError('Missing real vLLM reference token counts')
    import json
    maximum=None;total=0
    for split in ('cm','cu'):
        for line in (Path('/home/chs/dataset/MICE-Bench')/split/'test_metadata.jsonl').read_text().splitlines():
            meta=json.loads(line);sid=split+'/'+Path(meta['image']).stem
            folder=ROOT/'outputs/mice/full_20261004_attention/run'/sid
            paths=[folder/'turn_0_input.png']+[folder/f'turn_{i}.png' for i in range(1,4)]
            sizes=[]
            for path in paths:
                with Image.open(path) as image:sizes.append(list(image.size))
            for turn in range(1,4):
                prompt=ga_prompt(split,meta['instruction'][:turn],meta['formatted_instruction'][:turn])
                n=count(sizes[:turn+1],prompt);total+=1
                row={'session_id':sid,'turn':turn,'prompt_tokens':n,'images':[str(x) for x in paths[:turn+1]],'image_sizes':sizes[:turn+1],'prompt':prompt}
                if maximum is None or n>maximum['prompt_tokens']:maximum=row
    value={'status':'passed','scope':'all 2160 Qwen GA prefixes; no full scoring',
           'reference_vllm_requests_matched':len(comparisons),'prefixes_checked':total,
           'context_limit':16384,'completion_budget':SETTINGS['max_tokens'],'maximum':maximum,
           'sources':source_pins()}
    if maximum['prompt_tokens']+SETTINGS['max_tokens']>16384:raise ValueError('Insufficient Qwen context')
    write(a.output/'context_audit.json',value);print({k:v for k,v in value.items() if k not in ['maximum','sources']},flush=True)
    print('maximum',maximum['session_id'],maximum['turn'],maximum['prompt_tokens'],flush=True)
    if a.probe:
        from lance_mice.scoring.judge import Judge
        from lance_mice.scoring.request_cache import RequestCache
        pin=next(x for x in read(ROOT/'configs/scoring_models.json')['models'] if x['name']=='Qwen3.6-27B')
        cache=RequestCache(a.output/'requests','/home/chs/model/Qwen3.6-27B',pin,digest(value))
        judge=Judge('/home/chs/model/Qwen3.6-27B',cache,context_limit=16384)
        raw=judge.ask([open_image(x) for x in maximum['images']],maximum['prompt'],'GA')
        score=parse_ga(raw,maximum['turn']);trace=judge.traces[0]
        assert trace['prompt_tokens']==maximum['prompt_tokens']
        assert trace['prompt_tokens']+trace['completion_tokens']<=16384
        write(a.output/'context_probe.json',{'status':'passed','session_id':maximum['session_id'],'turn':maximum['turn'],'GA_prefix':score,'trace':trace,'sources':source_pins(),'settings':SETTINGS,'context_limit':16384})
        print('largest real prefix passed',flush=True)
if __name__=='__main__':main()
