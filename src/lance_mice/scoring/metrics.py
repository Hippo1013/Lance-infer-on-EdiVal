"""Shared GPU detection and timm DINOv3 CC with explicit missing states."""
from __future__ import annotations
import io
import math
from pathlib import Path
from PIL import Image, ImageDraw
from lance_mice.images import image_hash
from .core import digest, read, write, result

def detection_key(image, target, threshold=.3, return_all=False, delete_large_box=False):
    return digest([image_hash(image), target, threshold, return_all, delete_large_box])

class Detector:
    def __init__(self, root, cache, fingerprint, readonly=False):
        self.cache=Path(cache); self.fingerprint=fingerprint; self.readonly=readonly
        self.used={}; self.model=None; self.root=Path(root)

    def __call__(self, image, target, threshold=.3, return_all=False, delete_large_box=False):
        key=detection_key(image,target,threshold,return_all,delete_large_box)
        path=self.cache/f'{key}.json'
        request={'image_hash':image_hash(image),'target':target,'threshold':threshold,'return_all':return_all,'delete_large_box':delete_large_box}
        if path.exists():
            row=read(path)
            if row['fingerprint']!=self.fingerprint or row['request']!=request or digest(row['detections'])!=row['checksum']:
                raise ValueError('Detection cache identity mismatch')
            answer=row['detections']
        else:
            if self.readonly: raise ValueError('Missing precomputed detection: '+key)
            answer=self.detect(image,target,threshold,return_all,delete_large_box)
            write(path,{'fingerprint':self.fingerprint,'request':request,'detections':answer,'checksum':digest(answer)})
        self.used[key]=request
        return answer

    def detect(self,image,target,threshold,return_all,delete_large_box):
        import torch
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T
        if self.model is None:
            cfg=SLConfig.fromfile(str(self.root/'GroundingDINO-SwinT-OGC/GroundingDINO_SwinT_OGC.local.py'));cfg.device='cuda'
            self.model=build_model(cfg)
            weights=torch.load(self.root/'GroundingDINO-SwinT-OGC/groundingdino_swint_ogc.pth',map_location='cpu',weights_only=True)
            keys=self.model.load_state_dict(clean_state_dict(weights['model']),strict=False)
            if keys.missing_keys or set(keys.unexpected_keys)-{'bert.embeddings.position_ids','bert.embeddings.token_type_ids','label_enc.weight'}:
                raise ValueError('Grounding weight mismatch')
            self.model=self.model.eval().cuda()
        transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        tensor,_=transform(image.convert('RGB'),None)
        names=target if isinstance(target,list) else [target]
        caption=' . '.join(x.strip().lower().replace('.','') for x in names)+' .'
        with torch.inference_mode():
            boxes,scores,labels=predict(self.model,tensor,caption,threshold,threshold)
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all(): raise ValueError('Nonfinite detections')
        answer={k:[] for k in ['label','score','box','center']}
        for (cx,cy,w,h),s,label in zip(boxes.tolist(),scores.tolist(),labels):
            if delete_large_box and w>.98 and h>.98: continue
            answer['label'].append(label);answer['score'].append(s)
            answer['box'].append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
            answer['center'].append([int(cx*image.width),int(cy*image.height)])
        if answer['score'] and not return_all:
            i=max(range(len(answer['score'])),key=lambda i:answer['score'][i]);answer={k:[v[i]] for k,v in answer.items()}
        return answer

class Consistency:
    def __init__(self,root,detector):
        self.root=Path(root);self.detector=detector;self.model=None

    def load(self):
        import timm
        if self.model is None:
            self.model=timm.create_model('vit_large_patch16_dinov3.lvd1689m',pretrained=False,
                checkpoint_path=str(self.root/'DINOv3-ViT-L-16/model.safetensors')).eval().cuda()
            self.transform=timm.data.create_transform(**timm.data.resolve_model_data_config(self.model),is_training=False)

    def features(self,images):
        import torch
        self.load()
        x=torch.stack([self.transform(im) for im in images]).cuda()
        with torch.inference_mode(): f=self.model.forward_features(x)
        prefix=self.model.num_prefix_tokens
        gh,gw=x.shape[-2]//16,x.shape[-1]//16
        if f.shape[1]-prefix != gh*gw or not torch.isfinite(f).all(): raise ValueError('Invalid DINO patch grid')
        return f[:,0],f[:,prefix:],(gh,gw)

    def mask_weights(self,mask,grid):
        import torch
        from torchvision.transforms import Resize,CenterCrop,ToTensor,InterpolationMode
        from timm.data.transforms import MaybeToTensor
        # Mirror timm geometry, using nearest interpolation only for the mask.
        for t in self.transform.transforms:
            if isinstance(t,Resize): mask=Resize(t.size,interpolation=InterpolationMode.NEAREST,max_size=t.max_size)(mask)
            elif isinstance(t,CenterCrop): mask=t(mask)
            elif isinstance(t,(ToTensor,MaybeToTensor)): break
            else: raise ValueError('Unrecognized timm geometry: '+type(t).__name__)
        w=torch.nn.functional.adaptive_avg_pool2d(ToTensor()(mask).unsqueeze(0).cuda(),grid).flatten()
        return w*(w>=.5)

    def __call__(self,src,target,meta):
        import torch
        import torch.nn.functional as F
        if target.size!=src.size: target=target.resize(src.size,Image.Resampling.LANCZOS)
        unchanged,all_objects=meta['unchanged_objects'],meta['all_objects']
        # Preserve upstream early eligibility rule, but distinguish absent measurements.
        if not unchanged or not all_objects:
            return result('not_applicable',reason='Upstream requires both unchanged_objects and all_objects',
                object=None,background=None,upstream_missing_value=.5)
        det=self.detector(src,unchanged,.35,True)
        values=[]
        for box in det['box']:
            x1,y1,x2,y2=[max(0,min(int(v*s),s)) for v,s in zip(box,[src.width,src.height,src.width,src.height])]
            if x2<=x1 or y2<=y1: raise ValueError('Degenerate detected crop')
            cls,patch,_=self.features([im.crop((x1,y1,x2,y2)) for im in (src,target)])
            sim=.5*F.cosine_similarity(cls[0:1],cls[1:2]).item()+.5*F.cosine_similarity(patch[0].mean(0)[None],patch[1].mean(0)[None]).item()
            values.append(sim)
        obj=sum(values)/len(values) if values else None
        bg=None; patches=0; reason='background not requested'; box_count=0
        if meta['bg_consistency']:
            ds=self.detector(src,all_objects,.35,True);dt=self.detector(target,all_objects,.35,True)
            boxes=ds['box']+dt['box'];box_count=len(boxes)
            if boxes:
                mask=Image.new('L',src.size,255);draw=ImageDraw.Draw(mask)
                for b in boxes:
                    xy=[max(0,min(int(v*s),s)) for v,s in zip(b,[src.width,src.height,src.width,src.height])]
                    if xy[2]<xy[0] or xy[3]<xy[1]: raise ValueError('Invalid background box')
                    draw.rectangle(xy,fill=0)
                _,tokens,grid=self.features([src,target]);weights=self.mask_weights(mask,grid)
                patches=int((weights>0).sum())
                if patches:
                    pooled=(F.normalize(tokens,dim=-1)*weights[None,:,None]).sum(1)/(weights.sum()+1e-6)
                    bg=F.cosine_similarity(pooled[0:1],pooled[1:2]).item();reason='ok'
                else:reason='no background patches above mask threshold'
            else:reason='no foreground detections for background mask'
        available=[x for x in [obj,bg] if x is not None]
        return result('ok' if available else 'unmeasurable',sum(available)/len(available) if available else None,
            object=obj,object_values=values,background=bg,background_reason=reason,background_patches=patches,
            background_boxes=box_count,upstream_missing_value=sum(available)/len(available) if available else .5)
