#!/usr/bin/env python3
"""GPU mathematical controls for the production CC adapter, outside benchmark scores."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image
import torch
from lance_mice.scoring.core import read,write
from lance_mice.scoring.metrics import Detector,Consistency
from lance_mice.scoring.runner import open_image
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=read(a.output/'manifest.json');torch.set_num_threads(8)
d=Detector('/home/chs/model',a.output/'detections',m['fingerprint'])
c=Consistency('/home/chs/model',d)
r=next(r for r in m['records'] if any(r['metadata']['unchanged_objects']))
t=next(i for i,x in enumerate(r['metadata']['unchanged_objects']) if x)
im=open_image(r['images'][0]);meta={k:r['metadata'][k][t] for k in ['unchanged_objects','all_objects','bg_consistency']}
control=c(im,im,meta)
assert control['status']=='ok' and control['score']>.9999
cls,patch,grid=c.features([im,im]);assert patch.shape[1]==grid[0]*grid[1] and c.model.num_prefix_tokens==5
# A 2:1 mask whose central 1:1 crop is entirely black must retain no patches.
mask=Image.new('L',(512,256),255)
mask.paste(0,(128,0,384,256));w=c.mask_weights(mask,grid)
assert w.count_nonzero().item()==0
write(a.output/'cc_controls.json',{'status':'passed','identical_image_CC':control,
    'prefix_tokens':c.model.num_prefix_tokens,'patch_shape':list(patch.shape),'grid':grid,
    'mask_center_crop_control':'passed'})
print('CC identical-image and mask geometry controls passed')
