#!/usr/bin/env python3
"""Read accepted EdiVal descriptive tables; render the first three report sections.

Run on the data server. Inputs are immutable; all new evidence stays under --output.
Only final PNG/SVG figures are intended for transfer to the local report directory.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import pathlib
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BLUE = '#377EB8'
GREEN = '#33957B'
ORANGE = '#DB8A39'
PURPLE = '#9474B4'
GRAY = '#77818C'
CATS = ['current_instruction', 'history_instructions', 'original_image',
        'history_images', 'target_image', 'context_other', 'generation_markers']
LABELS = ['当前指令', '历史指令', '原图', '历史生成图', '正在生成的图', '其他上下文', '生成标记']
COLORS = [BLUE, '#94B9D7', '#9BCBBA', GREEN, PURPLE, '#CED2D6', '#626B75']
ROLES = [('current_instruction', 'text'), ('history_instruction', 'text'),
         ('original_image', 'vit'), ('original_image', 'vae'),
         ('history_image', 'vit'), ('history_image', 'vae')]
ROLE_LABELS = ['当前指令 · token', '历史指令 · token', '原图 · ViT区域',
               '原图 · VAE区域', '历史图 · ViT区域', '历史图 · VAE区域']
ROLE_COLORS = [BLUE, '#8DB4D2', GREEN, ORANGE, GREEN, ORANGE]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def number(row, key):
    return float(row[key]) if row[key] != '' else np.nan


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f)


def one(rows, **filters):
    found = [r for r in rows if all(r.get(k) == str(v) for k, v in filters.items())]
    if len(found) != 1:
        raise ValueError((filters, len(found)))
    return found[0]


def average(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def role_average(rows, role, modality, get_value):
    """Equal objects per session/turn, equal present turns, equal sessions."""
    st = collections.defaultdict(list)
    for r in rows:
        if r['object_role'] == role and r['modality'] == modality:
            st[r['session_id'], r['generation_turn']].append(get_value(r))
    ss = collections.defaultdict(list)
    for (s, _), values in st.items():
        ss[s].append(average(values))
    return average([average(v) for v in ss.values()])


def clean(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#B7BEC6')
    ax.tick_params(length=3, color='#B7BEC6')
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color='#E8EBEF', lw=.7)


def line_ci(ax, rows, color, label, metric, scale=100, marker='o', linestyle='-'):
    rows = sorted(rows, key=lambda r: int(r['generation_turn']))
    x = np.array([int(r['generation_turn']) for r in rows])
    y = np.array([number(r, 'mean') for r in rows]) * scale
    lo = np.array([number(r, 'ci_low') for r in rows]) * scale
    hi = np.array([number(r, 'ci_high') for r in rows]) * scale
    ax.errorbar(x, y, yerr=[y-lo, hi-y], color=color, label=label,
                marker=marker, ls=linestyle, capsize=3, lw=1.7, ms=5)
    ax.set_xticks([1, 2, 3], ['第1轮', '第2轮', '第3轮'])
    ax.set_xlim(.8, 3.2)
    clean(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--font', type=pathlib.Path, required=True)
    args = ap.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if root == out or root in out.parents:
        raise ValueError('Report output must be separate from accepted inputs')
    out.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(args.font))
    family = font_manager.FontProperties(fname=str(args.font)).get_name()
    plt.rcParams.update({'font.family': family, 'font.size': 11,
                         'axes.titlesize': 13, 'axes.labelsize': 11,
                         'axes.unicode_minus': False, 'legend.fontsize': 10,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'svg.fonttype': 'path'})
    completion = json.loads((root/'completion.json').read_text())
    assert completion['status'] == 'passed' and completion['input_turns'] == 1716
    assert sha(root/completion['artifact_manifest_path']) == completion['artifact_manifest_sha256']
    tables = {}
    names = ['composition_summary', 'trajectory_summary', 'paired_summary',
             'concentration_summary', 'category_turn', 'group_turn',
             'paired_changes', 'concentration_turn', 'object_trajectories']
    for name in names:
        tables[name] = list(read_rows(root/'tables'/f'{name}.csv.gz'))
    comp, conc = tables['composition_summary'], tables['concentration_summary']
    evidence = {'source_task': completion['task_id'],
                'source_candidate_sha256': completion['independently_validated_candidate_sha256'],
                'source_manifest_sha256': completion['artifact_manifest_sha256'],
                'script_sha256': sha(pathlib.Path(__file__)),
                'python': platform.python_version(), 'numpy': np.__version__,
                'matplotlib': matplotlib.__version__, 'font_sha256': sha(args.font),
                'tables': {n: sha(root/'tables'/f'{n}.csv.gz') for n in names},
                'figures': [], 'notes': []}
    facts = {}

    def save(fig, name, source, note):
        for ext in ['png', 'svg']:
            p = out/f'{name}.{ext}'
            fig.savefig(p, dpi=210, bbox_inches='tight', pad_inches=.16)
            evidence['figures'].append({'path': p.name, 'sha256': sha(p),
                                        'bytes': p.stat().st_size, 'source': source, 'note': note})
        plt.close(fig)

    # 1: Complete composition plus an enlarged text-only view.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1.35, 1]})
    turns = ['1', '2', '3', 'overall']
    bottom = np.zeros(4)
    for cat, label, color in zip(CATS, LABELS, COLORS):
        values = [number(one(comp, table='category_turn', category=cat,
                             generation_turn=t, metric='mass'), 'mean')*100 for t in turns]
        axs[0].bar(range(4), values, bottom=bottom, color=color, label=label,
                   width=.63, edgecolor='white', linewidth=.45)
        for j, v in enumerate(values):
            if v >= 7:
                axs[0].text(j, bottom[j]+v/2, f'{v:.1f}', ha='center', va='center',
                            fontsize=10, color='white' if cat in ['history_images', 'generation_markers'] else '#202730')
        bottom += values
    assert np.allclose(bottom, 100, atol=.01)
    axs[0].set(ylim=(0, 100), ylabel='注意力总占比（%）', title='(a) 全部关注对象')
    axs[0].set_xticks(range(4), ['第1轮', '第2轮', '第3轮', '三轮整体'])
    clean(axs[0])
    for cat, label, color in zip(CATS[:2], LABELS[:2], COLORS[:2]):
        rows = [one(comp, table='category_turn', category=cat, generation_turn=t, metric='mass') for t in ['1','2','3']]
        line_ci(axs[1], rows, color, label, 'mass')
    axs[1].set(title='(b) 指令部分的放大视图', ylabel='注意力总占比（%）', ylim=(0, 1.9))
    axs[1].legend(frameon=False, loc='upper right')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(.49, -.02), frameon=False)
    fig.subplots_adjust(bottom=.23, wspace=.30)
    save(fig, '01_attention_allocation', 'composition_summary', 'Seven disjoint categories; no renormalization. Text intervals: 95% original-image cluster bootstrap.')
    facts['allocation_percent'] = {label: [round(number(one(comp, table='category_turn', category=cat, generation_turn=t, metric='mass'),'mean')*100,4) for t in turns] for cat,label in zip(CATS,LABELS)}

    # 2: Per-token values and their context-length-adjusted counterpart.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, metric, scale, title, xlabel in [
            (axs[0], 'per_token_mass', 100, '(a) 每个 token 的平均份额', '单个 token 的注意力（%，对数刻度）'),
            (axs[1], 'enrichment', 1, '(b) 相对均匀分配的倍数', '相对倍数（对数刻度）')]:
        for i,(cat,color) in enumerate(zip(CATS,COLORS)):
            row=one(comp,table='category_turn',category=cat,generation_turn='overall',metric=metric)
            v,lo,hi=[number(row,k)*scale for k in ['mean','ci_low','ci_high']]
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,ms=7,capsize=3)
            ax.annotate(f'{v:.4f}%' if metric=='per_token_mass' else f'{v:.2f}倍',
                        (v,i),xytext=(9,0),textcoords='offset points',va='center',fontsize=10)
        ax.set_xscale('log');ax.set(title=title,xlabel=xlabel)
        ax.set_yticks(range(7), LABELS);clean(ax,'x')
        ax.set_xlim((.003,20) if metric=='per_token_mass' else (.09,900))
    axs[0].invert_yaxis()
    axs[0].set_xticks([.01,.1,1,10],['0.01','0.1','1','10'])
    axs[1].set_xticks([.1,1,10,100],['0.1','1','10','100'])
    axs[1].axvline(1,color='#A4ADB7',ls='--',lw=1)
    fig.subplots_adjust(wspace=.25,bottom=.18,left=.15)
    save(fig,'02_token_density','composition_summary','Overall densities are present-only; session equal. Uniform baseline=1. All seven categories retained.')

    # 3: Same identities followed across dialogue turns, without filling future objects.
    fig, axs=plt.subplots(2,2,figsize=(11.5,7),sharex=True)
    traj=tables['trajectory_summary']
    for j,(kind,mod,indices,colors,names_) in enumerate([
            ('text','text',[1,2,3],[BLUE,'#7CADD0','#ADCDE3'],['指令1','指令2','指令3']),
            ('image','combined',[0,1,2],[GREEN,'#79B49E','#BDD8CB'],['原图','第1轮生成图','第2轮生成图'])]):
        for i,(metric,scale,ylabel) in enumerate([('mass',100,'注意力总占比（%）'),('per_token_mass',100,'单个 token 的注意力（%）')]):
            ax=axs[i,j]
            for index,color,label in zip(indices,colors,names_):
                rows=[r for r in traj if r['object_type']==kind and r['modality']==mod and r['object_index']==str(index) and r['metric']==metric]
                line_ci(ax,rows,color,label,metric,scale)
            ax.set_ylabel(ylabel);ax.set_ylim(bottom=0)
            if i==0:ax.set_title('(a) 同一条指令' if j==0 else '(b) 同一张图像');ax.legend(frameon=False,fontsize=9)
    fig.subplots_adjust(hspace=.20,wspace=.28,bottom=.08)
    save(fig,'03_object_trajectories','trajectory_summary','Missing future objects remain absent. Image=ViT+VAE. Points=session means; bars=95% bootstrap intervals.')

    # 4: Per-session paired distributions, not only differences of average lines.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    pair_specs=[('text','text',[(1,1,2),(1,2,3),(1,1,3),(2,2,3)],BLUE),
                ('image','combined',[(0,1,2),(0,2,3),(0,1,3),(1,2,3)],GREEN)]
    pairfacts=[]
    for ax,(kind,mod,specs,color) in zip(axs,pair_specs):
        values=[];labs=[]
        for index,a,b in specs:
            arr=np.array([number(r,'delta_mass')*100 for r in tables['paired_changes'] if r['object_type']==kind and r['modality']==mod and r['object_index']==str(index) and r['turn_from']==str(a) and r['turn_to']==str(b)])
            assert len(arr)==572
            values.append(arr)
            obj=f'指令{index}' if kind=='text' else '原图' if index==0 else f'第{index}轮生成图'
            labs.append(f'{obj}：第{a}→{b}轮')
            sr=one(tables['paired_summary'],object_type=kind,modality=mod,object_index=index,turn_from=a,turn_to=b,metric='delta_mass')
            pairfacts.append({'object':obj,'from':a,'to':b,'mean_pp':float(arr.mean()),'declining':int(np.sum(arr < -1e-10)), 'ci_pp':[number(sr,'ci_low')*100,number(sr,'ci_high')*100]})
        bp=ax.boxplot(values,orientation='horizontal',tick_labels=labs,whis=(10,90),showfliers=False,patch_artist=True,widths=.48)
        for patch in bp['boxes']:patch.set(facecolor=color,alpha=.25,edgecolor=color)
        for med in bp['medians']:med.set(color=color,lw=2)
        ax.scatter([v.mean() for v in values],range(1,5),marker='D',s=28,color=color,zorder=4,label='平均变化')
        ax.axvline(0,color='#8F99A3',ls='--',lw=1)
        ax.set_xlabel('注意力占比变化（百分点；左侧为下降）')
        ax.set_title('(a) 指令的逐会话变化' if kind=='text' else '(b) 图像的逐会话变化')
        ax.invert_yaxis();clean(ax,'x')
        ax.legend(frameon=False,loc='lower left',fontsize=9)
    fig.subplots_adjust(wspace=.50,bottom=.18,left=.13)
    save(fig,'04_paired_changes','paired_changes','Boxes=P25-P75; whiskers=P10-P90; line=median; diamond=mean. Every row has 572 paired sessions; omitted tails remain in numeric counts.')
    facts['paired_changes']=pairfacts

    # 5: Concentration measures on each individual object's own support.
    fig,axs=plt.subplots(1,3,figsize=(13,4.7),sharey=True)
    cm=tables['concentration_turn']
    metrics=[('normalized_entropy',1,'(a) 归一化熵','越接近1，分布越均匀'),
             ('top10pct_share',100,'(b) 前10%元素的份额','越高，越集中（%）'),
             ('n90_fraction',100,'(c) 覆盖90%注意力所需元素','元素占比越低，越集中（%）')]
    facts['concentration']=[]
    for ax,(metric,scale,title,xlabel) in zip(axs,metrics):
        for i,((role,mod),color) in enumerate(zip(ROLES,ROLE_COLORS)):
            row=one(conc,grouping='object_role',object_role=role,modality=mod,generation_turn='overall',metric=metric,variant='main')
            v,lo,hi,p10,p90=[number(row,k)*scale for k in ['mean','ci_low','ci_high','p10','p90']]
            ax.plot([p10,p90],[i,i],color=color,alpha=.25,lw=7,solid_capstyle='round')
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=color,ms=5,capsize=4)
            baseline=role_average(cm,role,mod,lambda r: 1 if metric=='normalized_entropy' else int(r['top10pct_k'])/int(r['support_N']) if metric=='top10pct_share' else np.ceil(.9*int(r['support_N']))/int(r['support_N']))*scale
            ax.scatter(baseline,i,marker='|',color='#858E98',s=110,linewidth=1.5)
            facts['concentration'].append({'role':role,'modality':mod,'metric':metric,'mean':v/scale,'uniform_baseline':baseline/scale})
        ax.set_yticks(range(6),ROLE_LABELS)
        ax.set(title=title,xlabel=xlabel,xlim=(0,1.04) if scale==1 else (0,104))
        ax.axhline(1.5,color='#E2E6EA',lw=1,ls=':');clean(ax,'x')
    axs[0].invert_yaxis()
    handles=[Line2D([0],[0],marker='o',color=GRAY,lw=1,label='均值与95%区间'),
             Line2D([0],[0],color=GRAY,alpha=.25,lw=7,label='会话间P10–P90范围'),
             Line2D([0],[0],marker='|',color=GRAY,lw=0,markersize=10,label='均匀分配参考')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.56,-.01))
    fig.subplots_adjust(left=.16,wspace=.17,bottom=.22)
    save(fig,'05_concentration_overview','concentration_summary + concentration_turn','Per-position metrics then equal objects/turns/sessions. Text and region supports are different; uniform top fraction uses ceil(0.1*N)/N.')

    # 6: Changes and the pre-specified low-mass sensitivity view.
    fig,axs=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    panels=[('text','(a) 指令内部',[('current_instruction',BLUE,'当前指令'),('history_instruction','#8DB4D2','历史指令')]),
            ('original_image','(b) 原图内部',[('vit',GREEN,'ViT区域'),('vae',ORANGE,'VAE区域')]),
            ('history_image','(c) 历史生成图内部',[('vit',GREEN,'ViT区域'),('vae',ORANGE,'VAE区域')])]
    for ax,(kind,title,series) in zip(axs,panels):
        for key,color,label in series:
            role,mod=(key,'text') if kind=='text' else (kind,key)
            for variant,style in [('main','-'),('mass_gt_1e-6','--')]:
                rows=[r for r in conc if r['grouping']=='object_role' and r['object_role']==role and r['modality']==mod and r['generation_turn']!='overall' and r['metric']=='top10pct_share' and r['variant']==variant]
                line_ci(ax,rows,color,label if variant=='main' else '_nolegend_','top10pct_share',linestyle=style)
        ax.set(title=title,ylim=(0,100));ax.legend(frameon=False,loc='upper left',fontsize=9)
    axs[0].set_ylabel('前10%元素的注意力份额（%）')
    fig.legend(handles=[Line2D([0],[0],color=GRAY,label='实线：全部正注意力记录'),Line2D([0],[0],color=GRAY,ls='--',label='虚线：去掉总份额≤0.0001%的记录')],loc='lower center',ncol=2,frameon=False)
    fig.subplots_adjust(wspace=.13,bottom=.23)
    save(fig,'06_concentration_turns','concentration_summary','Pre-specified sensitivity: M>1e-6. History-image role at turn3 averages I1/I2 inside each session before population aggregation.')

    # Deterministic illustrative case, chosen by middle-of-distribution statistics.
    # This is not a success/failure filter or a selection of an extreme attention map.
    session_vectors={}
    cats=tables['category_turn']
    for sid in sorted({r['session_id'] for r in cats}):
        v=[]
        for cat in ['current_instruction','original_image','history_images']:
            v.append(number(one(cats,session_id=sid,generation_turn=3,category=cat),'mass'))
        for st,idx,mod in [('text',3,'text'),('image',0,'vit'),('image',0,'vae')]:
            v.append(number(one(cm,session_id=sid,generation_turn=3,support_type=st,object_index=idx,modality=mod),'normalized_entropy'))
        session_vectors[sid]=v
    sids=list(session_vectors);matrix=np.array(list(session_vectors.values()))
    med=np.median(matrix,axis=0);iqr=np.diff(np.quantile(matrix,[.25,.75],axis=0),axis=0)[0]
    distances=np.mean(np.abs(matrix-med)/np.where(iqr>0,iqr,1),axis=1)
    sid=sids[int(np.argmin(distances))]
    facts['example']={'session_id':sid,'turn':3,'selection':'Minimum average absolute deviation from six final-turn population medians, scaled by IQR; deterministic lexicographic tie break. Variables: current/original/history mass; T3 entropy; I0 ViT/VAE entropy.', 'distance':float(np.min(distances))}
    textrows=[r for r in read_rows(root/'tables/text_token_profiles.csv.gz') if r['session_id']==sid and r['generation_turn']=='3']
    fig,axs=plt.subplots(3,1,figsize=(11.5,8))
    facts['example']['instructions']=[]
    for index,ax,color in zip([1,2,3],axs,[BLUE,'#6099C3','#8DB4D2']):
        rows=sorted([r for r in textrows if r['object_index']==str(index)],key=lambda r:int(r['token_ordinal']))
        shares=np.array([number(r,'mean_conditional_share')*100 for r in rows]);assert np.isclose(shares.sum(),100)
        labels=[r['token_piece'].replace('\n','↵') or f"#{r['token_ordinal']}" for r in rows]
        xs=np.arange(len(rows));ax.bar(xs,shares,color=color,width=.7)
        ax.axhline(100/len(rows),color='#929CA5',ls='--',lw=1)
        ax.set_xticks(xs,labels,rotation=28 if len(rows)>13 else 0,ha='right' if len(rows)>13 else 'center',fontsize=10)
        title=rows[0]['instruction_text']
        ax.set_title(f'指令{index}：{title}',loc='left',fontsize=12)
        ax.set_ylabel('组内份额（%）');ax.set_ylim(0,max(shares.max()*1.20,18));clean(ax)
        facts['example']['instructions'].append({'instruction':index,'text':title,'tokens':len(rows),'peak_piece':labels[int(shares.argmax())],'peak_percent':float(shares.max())})
    fig.subplots_adjust(hspace=.72,left=.09,bottom=.08,top=.94)
    save(fig,'07_token_example','text_token_profiles',f'Session {sid}, turn3. Equal mean of per-position conditional shares. Dashed=uniform. Actual saved offsets, not retokenized words; illustrated case only.')

    imgrows=[r for r in read_rows(root/'tables/image_region_profiles.csv.gz') if r['session_id']==sid and r['generation_turn']=='3']
    fig,axs=plt.subplots(2,3,figsize=(10.7,7.0),layout='constrained')
    maps={}
    for mod in ['vit','vae']:
        for index in range(3):
            a=np.full((8,8),np.nan)
            for r in imgrows:
                if r['modality']==mod and r['object_index']==str(index):a[int(r['row']),int(r['col'])]=number(r,'mean_conditional_share')*100
            assert np.isclose(np.nansum(a),100)
            maps[mod,index]=a
    vmax=max(np.nanmax(a) for a in maps.values())
    for i,mod in enumerate(['vit','vae']):
        for j in range(3):
            ax=axs[i,j]
            im=ax.imshow(maps[mod,j],vmin=0,vmax=vmax,cmap='YlGnBu',interpolation='nearest')
            ax.set_xticks([0,3,7],[1,4,8]);ax.set_yticks([0,3,7],[1,4,8])
            ax.set_xlabel('网格列');ax.set_ylabel(f'{mod.upper()} · 网格行' if j==0 else '网格行')
            if i==0:ax.set_title(['原图','第1轮生成图','第2轮生成图'][j])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.83,pad=.025,label='单个区域占本图本编码的注意力（%）')
    save(fig,'08_region_example','image_region_profiles',f'Session {sid}, turn3. Six maps share a color scale. Each is normalized inside its own image/encoding; no original image downloaded; no semantic localization claim.')
    facts['example']['map_peak_percent']=float(vmax)
    # Population spatial maps preserve the same equal object/turn/session hierarchy.
    # No selection by the brightest region; use all 572 sessions and all valid maps.
    spatial=collections.defaultdict(lambda:np.full((8,8),np.nan))
    for r in read_rows(root/'tables/image_region_profiles.csv.gz'):
        key=(r['session_id'],int(r['generation_turn']),int(r['object_index']),r['modality'])
        spatial[key][int(r['row']),int(r['col'])]=number(r,'mean_conditional_share')
    assert len(spatial)==6864
    assert all(np.isclose(np.nansum(a),1) for a in spatial.values())
    spatialfacts={};population_maps={}
    for role in ['original_image','history_image']:
        for mod in ['vit','vae']:
            st=collections.defaultdict(list)
            peaks=[]
            for (s,t,index,m),a in spatial.items():
                if m==mod and ((role=='original_image' and index==0) or (role=='history_image' and index>0)):
                    st[s,t].append(a)
                    peaks.append(int(np.nanargmax(a)))
            ss=collections.defaultdict(list)
            for (s,t),values in st.items():ss[s].append(np.mean(values,axis=0))
            session_maps=np.array([np.mean(values,axis=0) for s,values in sorted(ss.items())])
            pop=np.mean(session_maps,axis=0);population_maps[role,mod]=pop*100
            loc=np.unravel_index(int(np.argmax(pop)),(8,8))
            spatialfacts[role+'_'+mod]={'sessions':len(ss),'support_maps':len(peaks),
                'peak_row_col_1based':[int(x)+1 for x in loc],
                'top_left_mean_percent':float(pop[0,0]*100),
                'top_left_session_p10_p90_percent':(np.quantile(session_maps[:,0,0],[.1,.9])*100).tolist(),
                'top_left_peak_map_fraction':float(np.mean(np.array(peaks)==0)),
                'mean_map_sum':float(pop.sum())}
    fig,axs=plt.subplots(2,2,figsize=(9.5,7.5),layout='constrained')
    vmax=max(np.max(a) for a in population_maps.values())
    for i,mod in enumerate(['vit','vae']):
        for j,role in enumerate(['original_image','history_image']):
            a=population_maps[role,mod];ax=axs[i,j]
            im=ax.imshow(a,vmin=0,vmax=vmax,cmap='YlGnBu',interpolation='nearest')
            ax.text(0,0,f'{a[0,0]:.1f}%',ha='center',va='center',color='white' if a[0,0]>vmax*.55 else '#202730',fontsize=8.5)
            ax.set_xticks([0,3,7],[1,4,8]);ax.set_yticks([0,3,7],[1,4,8])
            ax.set_xlabel('网格列');ax.set_ylabel(f'{mod.upper()} · 网格行')
            if i==0:ax.set_title('原图 · 三轮整体' if j==0 else '历史生成图 · 存在时整体')
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.85,pad=.025,label='单个区域的平均组内份额（%）')
    save(fig,'09_population_regions','image_region_profiles','All 572 sessions, equal objects within turn, present turns within session, then equal sessions. Shared scale; uniform region baseline 1/64=1.5625%.')
    facts['population_regions']=spatialfacts
    for name in ['text_token_profiles','image_region_profiles']:
        evidence['tables'][name]=sha(root/'tables'/f'{name}.csv.gz')
    accepted=json.loads((root/completion['artifact_manifest_path']).read_text())
    accepted_hashes={entry['path']:entry['sha256'] for entry in accepted['files']}
    for name,digest in evidence['tables'].items():
        assert accepted_hashes[f'tables/{name}.csv.gz']==digest, name
    evidence['validation']={'read_tables_match_accepted_manifest':len(evidence['tables']),
        'allocation_sums_percent_tolerance':.01,'paired_sessions_per_series':572,
        'spatial_support_maps':len(spatial),'spatial_map_conditional_sums_passed':True,
        'population_sessions_per_role':572,'unresolved_issues':[]}
    facts['low_mass_and_zero']={}
    for role,mod in ROLES:
        facts['low_mass_and_zero'][role+'_'+mod]={k:role_average(cm,role,mod,lambda r,k=k:number(r,k)) for k in ['low_mass_fraction','zero_mass_fraction']}
    evidence['notes']=['No original input or accepted output was modified.',
        'Bootstrap intervals are taken from accepted summaries; no new resampling.',
        'Supplemental baselines and case selection are descriptive, equally weighted, and documented in report_facts.json.',
        'All PNGs and SVGs must be visually reviewed before document delivery.']
    (out/'report_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    evidence['facts_sha256']=sha(out/'report_facts.json')
    (out/'figure_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'rendered','figures':len(evidence['figures']),'output':str(out),'example_session':sid,'source_candidate_sha256':evidence['source_candidate_sha256']},ensure_ascii=False))


if __name__=='__main__':
    main()
