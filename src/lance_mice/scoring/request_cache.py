"""Immutable raw-response cache. Interpretation changes never select new answers."""
from pathlib import Path
from .core import SETTINGS, digest, read, write, request_identity, sampling_seed

def valid_request(trace, model):
    return (trace['model']==model
            and trace['sampling_seed']==sampling_seed(trace['vote'])
            and trace['temperature']==SETTINGS['temperature']
            and trace['request_hash']==request_identity(trace['image_hashes'],trace['prompt'],trace['vote']))

class RequestCache:
    def __init__(self,root,model,model_pin,producer_fingerprint):
        self.root=Path(root);self.model=str(model);self.pin=model_pin;self.producer=producer_fingerprint

    def binding(self,request_hash):
        return {'request_hash':request_hash,'model_path':self.model,'model_pin':self.pin,'settings':SETTINGS}

    def get(self,request_hash):
        binding=self.binding(request_hash);path=self.root/(digest(binding)+'.json')
        if not path.exists():return None
        row=read(path);checksum=row.pop('checksum')
        if digest(row)!=checksum or row['binding']!=binding:raise ValueError('Raw request cache mismatch')
        trace=row['trace']
        if trace['request_hash']!=request_hash or not valid_request(trace,self.model):raise ValueError('Cached input identity mismatch')
        if trace['finish_reason']!='stop' or not trace['raw'].strip():raise ValueError('Cached generation was not complete')
        return row

    def put(self,trace,provenance=None):
        if trace.get('finish_reason')!='stop' or not trace.get('raw','').strip() or trace.get('error'):
            raise ValueError('Cannot cache an incomplete generation')
        if not valid_request(trace,self.model):
            raise ValueError('Cannot cache mismatched request identity')
        binding=self.binding(trace['request_hash']);path=self.root/(digest(binding)+'.json')
        row={'binding':binding,'trace':trace,'producer_fingerprint':self.producer,
             'provenance':provenance or {'kind':'fresh model generation'}}
        existing=self.get(trace['request_hash'])
        if existing:
            if existing['trace']['raw']!=trace['raw']:raise ValueError('Conflicting raw responses; refusing to choose one')
            return
        write(path,{**row,'checksum':digest(row)})
