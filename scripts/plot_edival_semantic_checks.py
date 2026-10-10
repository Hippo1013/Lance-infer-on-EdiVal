#!/usr/bin/env python3
"""Same-type preference and spatial weighting figures from accepted semantic tables."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
BLUE='#377EB8';GREEN='#33957B';ORANGE='#DB8A39';GRAY='#77818C'
def read(p):
    with gzip.open(p,'rt',encoding='utf-8',newline='') as f:yield from csv.DictReader(f)
def one(rows,**filters):
    rs=[r for r in rows if all(r.get(k)==str(v) for k,v in filters.items())]
    assert len(rs)==1,(filters,len(rs));return rs[0]
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,required=True);a=p.parse_args()
    root=a.input;out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert json.loads((root/'completion.json').read_text())['status']=='passed'
    font_manager.fontManager.addfont(str(a.font));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(a.font)).get_name(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    manifest=[];facts={}
    def save(fig,name):
        for ext in ['png','svg']:
            q=out/f'{name}.{ext}';fig.savefig(q,dpi=220,bbox_inches='tight',pad_inches=.16);manifest.append({'path':q.name,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
        plt.close(fig)
    rs=list(read(root/'tables/within_type_summary.csv.gz'))
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,typ,mod,indices,noun,color in [(axs[0],'text','text',[(1,2,2),(1,2,3),(1,3,3),(2,3,3)],'指令',BLUE),(axs[1],'image','combined',[(0,1,2),(0,1,3),(0,2,3),(1,2,3)],'图片',GREEN)]:
        labels=[];facts[typ]=[]
        for i,(older,newer,t) in enumerate(indices):
            r=one(rs,object_type=typ,modality=mod,older_index=older,newer_index=newer,generation_turn=t,metric='older_share')
            d=one(rs,object_type=typ,modality=mod,older_index=older,newer_index=newer,generation_turn=t,metric='older_density_share')
            y,lo,hi=[float(r[k])*100 for k in ['mean','ci_low','ci_high']]
            ax.errorbar(i-.08,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color,ms=7,capsize=3,label='总量比较' if i==0 else None)
            dy,dlo,dhi=[float(d[k])*100 for k in ['mean','ci_low','ci_high']]
            ax.errorbar(i+.08,dy,yerr=[[dy-dlo],[dhi-dy]],fmt='s',color=GRAY,ms=5,capsize=3,label='单个 token 比较' if i==0 else None)
            ax.annotate(f'{y:.1f}%',(i-.08,y),xytext=(0,-17),textcoords='offset points',ha='center',color=color)
            older_label=f'指令{older}' if typ=='text' else ('原图' if older==0 else f'生成图{older}')
            newer_label=f'指令{newer}' if typ=='text' else f'生成图{newer}'
            role_note='\n两者均为历史指令' if typ=='text' and (older,newer,t)==(1,2,3) else '\n两者均为历史生成图' if typ=='image' and older==1 else ''
            labels.append(f'第{t}轮\n{older_label} / {newer_label}'+role_note)
            facts[typ].append({'turn':t,'older':older_label,'newer':newer_label,'older_share_percent':y,'older_density_share_percent':dy,'ci_percent':[lo,hi]})
        ax.axhline(50,color='#ADB5BD',ls='--',lw=1);ax.set_xticks(range(4),labels);ax.set(ylabel='两者中较旧对象获得的份额（%）',ylim=(0,100),title='(a) 指令之间的分配' if typ=='text' else '(b) 图片之间的分配')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E8EBEF');ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right')
    fig.tight_layout(w_pad=3);save(fig,'10_within_type_preference')
    supports={}
    for r in read(root/'tables/image_region_profiles.csv.gz'):
        key=(r['session_id'],r['generation_turn'],r['object_role'],r['modality'],r['support_id'])
        if key not in supports:supports[key]=np.zeros((2,64))
        k=int(r['row'])*8+int(r['col']);supports[key][:,k]=[float(r['mean_conditional_share']),float(r['pooled_conditional_share'])]
    turnmaps=collections.defaultdict(list)
    for (sid,t,role,mod,support),v in supports.items():
        assert np.allclose(v.sum(1),1,atol=1e-9);turnmaps[sid,t,role,mod].append(v)
    sessionmaps=collections.defaultdict(list)
    for (sid,t,role,mod),v in turnmaps.items():sessionmaps[sid,role,mod].append(np.mean(v,axis=0))
    maps=collections.defaultdict(list)
    for (sid,role,mod),v in sessionmaps.items():maps[role,mod].append(np.mean(v,axis=0))
    roles=[('original_image','vit','原图 · ViT'),('original_image','vae','原图 · VAE'),('history_image','vit','历史图 · ViT'),('history_image','vae','历史图 · VAE')]
    fig,axs=plt.subplots(4,2,figsize=(9,13));maximum=max(np.mean(maps[role,mod],axis=0).max()*100 for role,mod,_ in roles);spatial={}
    for i,(role,mod,label) in enumerate(roles):
        assert len(maps[role,mod])==572
        record_maps=[v for (sid,t,r,m,support),v in supports.items() if r==role and m==mod]
        maxima=[float(np.mean([np.argmax(v[j])==0 for v in record_maps]))*100 for j in range(2)]
        mean=np.mean(maps[role,mod],axis=0)*100;spatial[role+'_'+mod]={'conditional_corner_percent':float(mean[0,0]),'pooled_corner_percent':float(mean[1,0]),'conditional_corner_max_record_percent':maxima[0],'pooled_corner_max_record_percent':maxima[1],'object_turn_records':len(record_maps),'conditional_map_percent':mean[0].tolist(),'pooled_map_percent':mean[1].tolist()}
        for j in range(2):
            ax=axs[i,j];im=ax.imshow(mean[j].reshape(8,8),cmap='YlGnBu',vmin=0,vmax=maximum,origin='upper');ax.set_title(label+(' · 生成记录等权' if j==0 else ' · 关注量加权')+f'\n左上格：{mean[j,0]:.2f}%',fontsize=11);ax.set_xticks([0,7],['左','右']);ax.set_yticks([0,7],['上','下'])
    fig.subplots_adjust(hspace=.6,wspace=.3,right=.83);cax=fig.add_axes([.87,.18,.022,.64]);fig.colorbar(im,cax=cax,label='区域在该图编码内的平均份额（%）');save(fig,'11_spatial_weighting')
    facts['spatial']=spatial
    marker=list(read(root/'tables/image_marker_turn.csv.gz'));markerfacts={};fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,field,yl in [(axs[0],'mean_raw_mass','结束标记占全部注意力的份额（%）'),(axs[1],'mean_block_share','结束标记在图像完整缓存块内的份额（%）')]:
        for i,(role,mod,label) in enumerate(roles):
            st=collections.defaultdict(list)
            for r in marker:
                if r['object_role']==role and r['modality']==mod and r['marker']=='end':st[r['session_id'],r['generation_turn']].append(float(r[field]))
            ss=collections.defaultdict(list)
            for (sid,t),v in st.items():ss[sid].append(np.mean(v))
            values=[np.mean(v)*100 for v in ss.values()];assert len(values)==572
            mean=float(np.mean(values));ax.bar(i,mean,color=GREEN if mod=='vit' else ORANGE,width=.6);ax.text(i,mean,f'{mean:.2f}',ha='center',va='bottom');markerfacts.setdefault(role+'_'+mod,{})[field+'_percent']=mean
        ax.set_xticks(range(4),[x[2].replace(' · ','\n') for x in roles]);ax.set(ylabel=yl,title='(a) 对全部注意力的贡献' if field=='mean_raw_mass' else '(b) 在图像块中的分配');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E8EBEF');ax.set_axisbelow(True);ax.margins(y=.17)
    fig.tight_layout(w_pad=3);save(fig,'12_image_end_markers');facts['image_end_markers']=markerfacts
    (out/'semantic_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n');(out/'semantic_figure_manifest.json').write_text(json.dumps({'figures':manifest,'source_completion_sha256':hashlib.sha256((root/'completion.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'status':'rendered','figures':len(manifest),'output':str(out)}))
if __name__=='__main__':main()
