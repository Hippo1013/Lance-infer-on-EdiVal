"""Single-model vLLM judge adapter; shared request contract, separate evidence."""
import base64
import io
from .core import SETTINGS, parse_yes_no, request_identity, sampling_seed
from lance_mice.images import image_hash

class Judge:
    def __init__(self, model, cache, context_limit=None):
        self.model=str(model);self.traces=[];self.cache=cache;self.llm=None
        self.context_limit=context_limit or SETTINGS['max_model_len']
        self.vote=1

    def load(self):
        if self.llm is not None:return
        from vllm import LLM
        self.llm=LLM(model=self.model,dtype=SETTINGS['dtype'],tensor_parallel_size=1,
            gpu_memory_utilization=SETTINGS['gpu_memory_utilization'],max_model_len=self.context_limit,
            max_num_seqs=1,limit_mm_per_prompt={'image':4},enforce_eager=True,seed=SETTINGS['seed'],disable_log_stats=True)

    def ask(self, images, prompt, kind):
        images=[im.convert('RGB') for im in images]
        hashes=[image_hash(im) for im in images]
        seed=sampling_seed(self.vote)
        trace={'kind':kind,'model':self.model,'image_hashes':hashes,'image_sizes':[list(im.size) for im in images],
               'runtime_context_limit':self.context_limit,
               'vote':self.vote,'sampling_seed':seed,'temperature':SETTINGS['temperature'],
               'prompt':prompt,'request_hash':request_identity(hashes,prompt,self.vote)}
        cached=self.cache.get(trace['request_hash'])
        if cached:
            reused={**cached['trace'],'reused_from':{'producer_fingerprint':cached['producer_fingerprint'],'provenance':cached['provenance']}}
            self.traces.append(reused)
            return reused['raw']
        self.traces.append(trace)
        try:
            content=[]
            for im in images:
                data=io.BytesIO();im.save(data,format='PNG')
                content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()}})
            content.append({'type':'text','text':prompt})
            self.load()
            from vllm import SamplingParams
            out=self.llm.chat([{'role':'user','content':content}],
                sampling_params=SamplingParams(temperature=SETTINGS['temperature'],max_tokens=SETTINGS['max_tokens'],seed=seed),
                chat_template_kwargs={'enable_thinking':SETTINGS['enable_thinking']},use_tqdm=False)[0]
            reply=out.outputs[0]
            trace.update(raw=reply.text,finish_reason=reply.finish_reason,prompt_tokens=len(out.prompt_token_ids),completion_tokens=len(reply.token_ids))
            if not reply.text.strip() or reply.finish_reason != 'stop': raise ValueError('Empty or truncated judge response')
            self.cache.put(trace)
            return reply.text
        except Exception as exc:
            trace['error']=f'{type(exc).__name__}: {exc}';raise

    def one(self,image,prompt):
        text=self.ask([image],prompt,'IF')
        return text.strip().lower() if 'Output only the text content' in prompt else parse_yes_no(text)

    def two(self,first,second,prompt):
        return parse_yes_no(self.ask([first,second],prompt,'IF'))
