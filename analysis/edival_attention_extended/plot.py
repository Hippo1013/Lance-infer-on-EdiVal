#!/usr/bin/env python3
"""Publication figures generated on the data server from the extended atlas."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image,ImageOps,PngImagePlugin
from core import mean_valid
from run import ROOT,PROJECT,TASK,sha,atomic_json

OUT=PROJECT/'outputs/attention_reports'/TASK
BLUE='#377EB8';GREEN='#33957B';ORANGE='#DB8A39';PURPLE='#9474B4'


def main(font):
    OUT.mkdir(parents=True,exist_ok=True)
    font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white'})
    with np.load(ROOT/'summaries/full_atlas.npz',allow_pickle=False) as z:
        atlas={k:z[f'mean_{i}'] for i,k in enumerate(z['keys'].tolist())}
        support={k:z[f'n_{i}'] for i,k in enumerate(z['keys'].tolist())}
    names=json.loads((ROOT/'metadata/metric_names.json').read_text())
    summary=json.loads((ROOT/'summaries/scalar_summary.json').read_text());ss={x['metric']:x for x in summary}
    review=json.loads((ROOT/'summaries/candidate_review.json').read_text())
    figures=[];facts={}
    def arr(t,key,mode='process'):
        return atlas[f't{t}__{mode}'][...,names[str(t)].index(key)]
    def sav(fig,name,note):
        fig.suptitle(note,fontsize=14)
        for ext in ['png','svg']:
            p=OUT/(name+'.'+ext);fig.savefig(p,dpi=190,bbox_inches='tight',pad_inches=.2)
            figures.append(dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,note=note))
        plt.close(fig)
    def heat(ax,a,title,kind='mass',vmin=None,vmax=None):
        a=np.asarray(a)*100 if kind in ['mass','share','delta'] else np.asarray(a)
        if kind=='delta':
            vmax=max(float(np.nanmax(np.abs(a))),.01) if vmax is None else vmax;vmin=-vmax;cmap='RdBu_r'
        else:cmap='YlGnBu'
        im=ax.imshow(a.T,origin='lower',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set(title=title,xlabel='去噪步骤',ylabel='模型层')
        ax.set_xticks([0,9,19,29],['1','10','20','30']);ax.set_yticks([0,11,23,35],['1','12','24','36'])
        cb=ax.figure.colorbar(im,ax=ax,fraction=.048,pad=.03);cb.ax.set_title('%' if kind in ['mass','share'] else ('百分点' if kind=='delta' else '数值'),fontsize=8)
        return im
    def clean(ax):
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E8EBEF',lw=.7);ax.set_axisbelow(True)
    # The complete atlas stays on the server; selected comparable views in text.
    fig,axs=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
    for row,(key,label) in enumerate([('category_current_instruction:mass','当前指令'),('category_history_instructions:mass','历史指令合计'),('category_original_image:mass','原图')]):
        mx=max(np.nanmax(arr(t,key)) for t in [1,2,3])*100
        for col,t in enumerate([1,2,3]):heat(axs[row,col],arr(t,key),f'{label} · 第{t}轮',vmin=0,vmax=mx)
    sav(fig,'13_process_allocation','图13｜生成过程中的关注总量：头平均，再对会话等权')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for col,(t,key,label) in enumerate([(2,'pair_T1_T2:older_share','第2轮：指令1 / 指令2'),(3,'pair_T1_T2:older_share','第3轮：历史指令1 / 2'),(3,'pair_I1_combined_I2_combined:older_share','第3轮：历史生成图1 / 2')]):
        a=arr(t,key);heat(axs[0,col],a,label,'share',0,100)
        ax=axs[1,col];ax.plot(np.arange(1,31),mean_valid(a,1)*100,color=BLUE,label='记录等权');ax.plot(np.arange(1,31),mean_valid(arr(t,key,'weighted_process'),1)*100,color=ORANGE,label='位置内关注量加权')
        ax.axhline(50,color='#888',ls='--',lw=1);ax.set(xlabel='去噪步骤',ylabel='较旧对象在两者中的份额（%）',ylim=(0,100));ax.legend(frameon=False);clean(ax)
    sav(fig,'14_relative_recency','图14｜同类对象的新旧分配：50%表示两者相同')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,(key,label) in enumerate([('T1:entropy','指令1 · token熵'),('I0_vae:entropy','原图 · VAE区域熵')]):
        for col,t in enumerate([1,2,3]):heat(axs[row,col],arr(t,key),f'{label} · 第{t}轮','entropy',0,1)
    sav(fig,'15_process_concentration','图15｜内部集中度：归一化熵越接近1越均匀')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for row,g in enumerate(['T1','I0_vae']):
        key=f'paired__1_3__{g}__process';cols=[f'1_3__{g}:delta_mass',f'1_3__{g}:delta_entropy',f'1_3__{g}:TV']
        # Pair files retain their own metric columns in alphabetical order.
        with np.load(ROOT/'paired'/sorted(json.loads((ROOT/'metadata/design.json').read_text())['split'])[0]/f'1_3__{g}.npz') as z:pn=z['names'].tolist()
        for col,(k,label,kind) in enumerate(zip(cols,['第3轮－第1轮总量','第3轮－第1轮熵','内部形状差异'],['delta','delta','share'])):
            heat(axs[row,col],atlas[key][...,pn.index(k)],f'{g} · {label}',kind,vmin=0 if kind=='share' else None,vmax=100 if kind=='share' else None)
    sav(fig,'16_paired_process','图16｜同一对象的跨轮变化：总量、集中度与具体分布')
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for ax,(key,label,lim) in zip(axs.ravel(),[('T1:mass','第3轮指令1 · 绝对份额',None),('pair_T1_T2:older_share','第3轮历史指令 · 较旧份额',100),('I0_vit:left_share','第3轮原图ViT · 左上内部份额',100),('I0_vae:marker_end_mass','第3轮原图VAE · 结束标记绝对份额',None)]):
        a=arr(3,key,'head')*100;im=ax.imshow(a,origin='lower',aspect='auto',cmap='YlGnBu',vmin=0,vmax=lim)
        ax.set(title=label,xlabel='同一层内的头编号',ylabel='模型层');ax.set_xticks([0,3,7,11,15],['1','4','8','12','16']);ax.set_yticks([0,11,23,35],['1','12','24','36']);fig.colorbar(im,ax=ax,fraction=.045,pad=.03).ax.set_title('%',fontsize=9)
    sav(fig,'17_head_preferences','图17｜各层16个头的分配差异：步骤平均、会话等权')
    fig,axs=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    for ax,(g,label) in zip(axs,[('T1','指令1'),('I0_vit','原图ViT'),('I0_vae','原图VAE')]):
        for k,color in [(1,BLUE),(4,ORANGE)]:
            a=atlas[f't3__{g}:top{k}_head_share'];ax.plot(np.arange(1,31),mean_valid(a,1)*100,color=color,label=f'最强{k}个头')
            ax.axhline(k/16*100,color=color,ls=':',lw=1)
        ax.set(title=f'第3轮 · {label}',xlabel='去噪步骤',ylabel='该层对象关注量的贡献（%）',ylim=(0,100));ax.legend(frameon=False);clean(ax)
    sav(fig,'18_head_contribution','图18｜少数头的贡献：每个会话各自排名，虚线为均分参考')
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,(g,label) in zip(axs[0],[('T1','旧指令1'),('T3','当前指令3'),('I0_vit','原图ViT')]):heat(ax,arr(3,g+':query_R'),label+' · 有效查询参与程度','share',0,100)
    with np.load(ROOT/'summaries/session_scalars.npz') as z:sn=z['metric_names'].tolist();sv=z['values'];sids=z['session_ids'].tolist()
    for ax,(g,label) in zip(axs[1],[('T1','旧指令1'),('T3','当前指令3'),('I0_vit','原图ViT')]):
        x=sv[:,sn.index(f't3__{g}:mass')]*100;y=sv[:,sn.index(f't3__{g}:query_R')]*100
        ax.scatter(x,y,s=9,alpha=.35,color=BLUE,rasterized=True);ax.set(xlabel='对象总份额（%）',ylabel='有效查询参与程度（%）',title=label,ylim=(0,100));clean(ax)
    sav(fig,'19_query_participation','图19｜读取的广度：有效参与程度不是实际Query数量比例')
    fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    for row,mod in enumerate(['vit','vae']):
        for col,(im,t,variant) in enumerate([(0,3,0),(0,3,1),(1,3,0),(1,3,1)]):
            a=atlas[f't{t}__I{im}_{mod}:space_overall'][variant].reshape(8,8)*100
            vmax=max(np.nanmax(atlas[f't3__I{k}_{mod}:space_overall'])*100 for k in [0,1])
            implot=axs[row,col].imshow(a,cmap='YlOrRd',vmin=0,vmax=vmax,origin='upper')
            axs[row,col].set(title=f'{"原图" if im==0 else "历史图1"} · {mod.upper()}\n{"记录等权" if variant==0 else "实际关注量加权"}',xlabel='区域列',ylabel='区域行');axs[row,col].set_xticks([0,3,7],['1','4','8']);axs[row,col].set_yticks([0,3,7],['1','4','8']);fig.colorbar(implot,ax=axs[row,col],fraction=.046,pad=.02).ax.set_title('%',fontsize=8)
    sav(fig,'20_spatial_weighting','图20｜第3轮固定坐标分布：两种平均方式并列')
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for col,(mod,label) in enumerate([('vit','ViT'),('vae','VAE')]):
        heat(axs[0,col],arr(3,f'I0_{mod}:left_share'),label+' · 左上格内部份额','share',0,100)
        heat(axs[1,col],arr(3,f'I0_{mod}:marker_end_mass'),label+' · 结束标记绝对份额','mass',0,None)
    sav(fig,'21_spatial_markers','图21｜真实空间格与非空间结束标记：分母与单位分别标明')
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for mod,color in [('vit',GREEN),('vae',ORANGE)]:
        a=atlas[f't3__I0_{mod}:step_motion_TV'];axs[0].plot(np.arange(2,31),mean_valid(a,1)*100,color=color,label=mod.upper())
        with np.load(ROOT/'paired'/sids[0]/f'1_3__I0_{mod}.npz') as z:pn=z['names'].tolist()
        a=atlas[f'paired__1_3__I0_{mod}__step'];axs[1].plot(np.arange(1,31),a[...,pn.index(f'1_3__I0_{mod}:TV')]*100,color=color,label=mod.upper())
        k=f'paired__1_3__I0_{mod}:TV';v=sv[:,sn.index(k)]*100;axs[2].hist(v,bins=25,histtype='step',density=True,color=color,label=mod.upper(),lw=1.8)
    axs[0].set(title='相邻去噪步骤',xlabel='后一个步骤',ylabel='原图区域分布差异（%）')
    axs[1].set(title='同一原图：第1轮→第3轮',xlabel='去噪步骤',ylabel='区域分布差异（%）')
    axs[2].set(title='会话间变化范围',xlabel='第1轮→第3轮区域分布差异（%）',ylabel='分布密度')
    for ax in axs:ax.legend(frameon=False);clean(ax)
    sav(fig,'22_spatial_motion','图22｜热点变化：分布差异0表示相同，100%表示完全分离')
    # Actual image overlay: inputs are read only on server, never copied locally.
    PngImagePlugin.MAX_TEXT_CHUNK=4*1024*1024
    case=next(x for x in review['cases'] if x['metric']=='I0_vit:left_share')
    chosen=[('典型',case['cases']['typical']),('较强',case['cases']['strong']),('反向',case['cases']['opposite'])]
    chosen=[(label,s) for label,s in chosen if s is not None]
    fig,axs=plt.subplots(len(chosen),4,figsize=(13,3.4*len(chosen)),squeeze=False,constrained_layout=True)
    for row,(label,sid) in enumerate(chosen):
        path=PROJECT/'outputs/sixrun_20261007/experiment/inference/edival_chat/run'/sid
        with Image.open(path/'turn_0_input.png') as image:img=ImageOps.exif_transpose(image).convert('RGB')
        meta=json.loads((ROOT/'objects'/sid/'turn_3.json').read_text())
        pixel_hash=hashlib.sha256(f'RGB:{img.width}:{img.height}:'.encode()+img.tobytes()).hexdigest()
        assert pixel_hash==meta['image_hashes'][0]
        # Full source image with verified resize-only geometry in both routes.
        for mod in ['vit','vae']:
            geom=next(g for g in meta['geometry'] if g['image_index']==0 and g['kind']==mod)
            w,h=geom['source_size'];geometry=geom['geometry']
            assert img.size==(w,h)
            assert geometry['source_crop_box']==[0,0,w,h]
            assert geometry['post_resize_crop_box']==[0,0,*geometry['resized_size']]
        with np.load(ROOT/'arrays'/sid/'turn_3.npz') as z:
            prof={k:z[f'profile_{i}'] for i,k in enumerate(z['profile_names'].tolist())}
        for col,(mod,variant) in enumerate([('vit',0),('vit',1),('vae',0),('vae',1)]):
            ax=axs[row,col];a=prof[f'I0_{mod}:space_overall'][variant].reshape(8,8)*100
            # Encoder resize preserves this grid's full-image coordinate range.
            ax.imshow(img,extent=(0,8,8,0));im=ax.imshow(a,extent=(0,8,8,0),cmap='inferno',vmin=0,vmax=35,alpha=.58,interpolation='nearest')
            ax.set(title=f'{label} · {sid}\n{mod.upper()} · {"等权" if variant==0 else "关注量加权"}');ax.set_xticks([]);ax.set_yticks([])
            fig.colorbar(im,ax=ax,fraction=.047,pad=.02).ax.set_title('%',fontsize=8)
    sav(fig,'23_spatial_cases','图23｜原图空间分布案例：按冻结窗口指标选取，统一色标')
    # Text profile change: same actual token identity, three editing rounds.
    k='paired__1_3__T1:TV';v=sv[:,sn.index(k)];sid=sids[int(np.nanargmin(abs(v-np.nanmedian(v))))]
    fig,axs=plt.subplots(3,1,figsize=(12,8),constrained_layout=True)
    texts=[]
    for ax,t in zip(axs,[1,2,3]):
        with np.load(ROOT/'arrays'/sid/f'turn_{t}.npz') as z:pi=z['profile_names'].tolist().index('T1:token_overall');a=z[f'profile_{pi}']
        meta=json.loads((ROOT/'objects'/sid/f'turn_{t}.json').read_text())['tokens']['T1'];texts.append(meta['text'])
        x=np.arange(a.shape[-1]);ax.bar(x-.2,a[0]*100,.4,color=BLUE,label='记录等权');ax.bar(x+.2,a[1]*100,.4,color=ORANGE,label='关注量加权');ax.set_xticks(x,[p.replace('\n',' ') for p in meta['pieces']],rotation=20,ha='right');ax.set(title=f'{sid} · 第{t}轮同一指令1',ylabel='指令内部份额（%）',ylim=(0,100));ax.legend(frameon=False);clean(ax)
    assert len(set(texts))==1;facts['token_case']={'session_id':sid,'instruction':texts[0],'selection_metric':k,'selection':'closest to median paired distribution change'}
    sav(fig,'24_token_changes','图24｜同一指令的token关注变化：集中程度与具体内容分别观察')
    fig,axs=plt.subplots(1,2,figsize=(13,3.8),constrained_layout=True)
    labels=['较旧历史指令','较旧历史图','原图ViT左上格','原图VAE结束标记']
    for ci,c in enumerate(review['cases']):
        for split_label,offset,color in [('exploration',-.15,BLUE),('confirmation',.15,ORANGE)]:
            r=next(r for r in review['rows'] if r['id']==c['id'] and r['split']==split_label);d=r['summary'];ax=axs[0] if ci<2 else axs[1];y=ci if ci<2 else ci-2
            ax.errorbar(d['mean']*100,y+offset,xerr=[[max(0,(d['mean']-d['ci_low'])*100)],[max(0,(d['ci_high']-d['mean'])*100)]],fmt='o',color=color,capsize=3,label=('探索部分' if split_label=='exploration' else '复核部分') if y==0 else None)
        ax=axs[0] if ci<2 else axs[1];ax.axvline(c['reference']*100,color='#aaa',ls=':',lw=1)
    axs[0].set_yticks([0,1],labels[:2]);axs[1].set_yticks([0,1],labels[2:]);axs[0].set(xlabel='较旧对象在两者中的份额（%）',title='新旧相对分配');axs[1].set(xlabel='左上格内部份额 / 标记绝对份额（%）',title='空间与标记：两个不同分母')
    for ax in axs:ax.legend(frameon=False);clean(ax)
    sav(fig,'25_candidate_confirmation','图25｜冻结局部窗口的复核：点为均值，误差线为聚类95%区间')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    with np.load(ROOT/'summaries/full_atlas.npz') as z:
        ak=z['keys'].tolist();coverage=z[f'below_half_{ak.index("t3__process")}']
    for ax,(key,label) in zip(axs,[('pair_T1_T2:older_share','历史指令1 / 2'),('pair_I1_combined_I2_combined:older_share','历史生成图1 / 2')]):
        heat(ax,coverage[...,names['3'].index(key)],label+' · 较旧对象份额低于50%的会话比例','share',0,100)
    sav(fig,'26_session_coverage','图26｜过程位置上的会话覆盖：先平均头，再判断会话方向')
    atomic_json(OUT/'figure_manifest.json',dict(task_id=TASK,script_sha256=sha(Path(__file__)),font_sha256=sha(font),figures=figures))
    atomic_json(OUT/'figure_facts.json',facts)
    print('figures generated',len(figures)//2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--font',type=Path,required=True);main(p.parse_args().font)
