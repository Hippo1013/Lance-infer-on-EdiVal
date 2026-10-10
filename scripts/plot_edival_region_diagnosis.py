#!/usr/bin/env python3
"""Render server-only diagnostic evidence into portable PNG/SVG figures."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT=Path('/home/chs/exp0_attention/Lance-infer-on-EdiVal/outputs/attention_analysis/edival_region_bias_20261009_v1')
FONT='/home/chs/tools/fonts/NotoSansCJKsc-Regular.otf'
BLUE='#3D79B3'; GREEN='#399D88'; ORANGE='#D99243'; INK='#283845'; GRAY='#EDF1F4'


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    font_manager.fontManager.addfont(FONT)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
        'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':INK})
    audit=json.loads((ROOT/'cache_order_audit.json').read_text())
    assert audit['status']=='reproduced' and audit['observer_vae_spatial_ids']==[22,41,42,43]
    fig,ax=plt.subplots(figsize=(13.6,5.5));ax.set_xlim(-1.9,7);ax.set_ylim(-.55,4.25);ax.axis('off')
    labels=[['起始标记','图块 0','图块 1','图块 2','图块 3','结束标记'],
            ['起始标记','结束标记','图块 0','图块 1','图块 2','图块 3'],
            ['其他上下文','空间 0','空间 1','空间 2','空间 3','其他上下文']]
    for ri,(y,title) in enumerate([(3.25,'输入中的顺序'),(1.85,'实际缓存顺序'),(.45,'原观测器的标签')]):
        ax.text(-.6,y,title,ha='right',va='center',fontweight='bold')
        for col,label in enumerate(labels[ri]):
            color=(ORANGE if label=='结束标记' else BLUE if label.startswith('图块') else '#8795A3') if ri<2 else (GREEN if 1<=col<=4 else '#8795A3')
            ax.add_patch(FancyBboxPatch((col,y-.3),.9,.6,boxstyle='round,pad=.035',linewidth=0,facecolor=color,alpha=.16))
            ax.text(col+.45,y,label,ha='center',va='center',color=color,fontsize=11)
    for source,dest in [(0,0),(1,2),(2,3),(3,4),(4,5),(5,1)]:
        ax.annotate('',xy=(dest+.45,2.20),xytext=(source+.45,2.88),arrowprops={'arrowstyle':'->','color':ORANGE if source==5 else '#A6B2BB','lw':1.3,'alpha':.9})
    ax.annotate('结束标记误入第一个空间位置',xy=(1.45,.8),xytext=(.6,1.14),color='#B46E28',fontsize=11,
                arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.annotate('最后一个图像 token 被排除',xy=(5.45,.8),xytext=(3.5,1.14),color='#B46E28',fontsize=11,
                arrowprops={'arrowstyle':'->','color':ORANGE})
    ax.set_title('VAE 缓存顺序与空间标签的错位',loc='left',fontsize=16,fontweight='bold',pad=15)
    ax.text(-1.85,-.35,'示意图使用4个图像 token；本批数据实际为1024个。两台服务器的已安装方法均通过CPU身份测试复现。',fontsize=10,color='#596A78')
    save(fig,'10_vae_cache_mapping')
    d=json.loads((ROOT/'combined.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(13.5,10.5),layout='constrained')
    cohorts=[(['t1_I0_vit','t2_I0_vit','t3_I0_vit'],[1/3]*3),(['t2_I1_vit','t3_I1_vit','t3_I2_vit'],[.5,.25,.25])]
    def scalar(metric):return [100*sum(d[k]['scalar'][metric]*w for k,w in zip(ks,ws)) for ks,ws in cohorts]
    a=axs[0,0];x=np.arange(2);width=.3
    for offset,metric,name,color in [(-.15,'conditional_topleft','先归一化，再平均',BLUE),(.15,'pooled_topleft','先累计份额，再归一化',GREEN)]:
        v=scalar(metric);bars=a.bar(x+offset,v,width,color=color,label=name)
        a.bar_label(bars,labels=[f'{v:.2f}%' for v in v],padding=4,fontsize=11)
    a.axhline(100/64,color=ORANGE,linestyle='--',lw=1,label='均匀参考：1.56%')
    a.set(xticks=x,xticklabels=['原图','历史生成图'],ylim=(0,23),ylabel='左上格的组内份额（%）')
    a.set_title('A  平均方式的影响',loc='left',fontweight='bold');a.legend(frameon=False,fontsize=9,loc='upper left')
    a=axs[0,1]
    steps=np.arange(1,31)
    for k,label,color in [('t1_I0_vit','第1轮原图',BLUE),('t3_I2_vit','第3轮最新历史图',GREEN)]:
        ar=np.asarray(d[k]['step_sums']);ratio=np.divide(ar[0],ar[1],out=np.full(30,np.nan),where=ar[1]>0)
        a.plot(steps,100*ratio,color=color,label=label,lw=2)
    a.axhline(100/64,color=ORANGE,linestyle='--',lw=1);a.set(ylim=(0,12),xlabel='去噪步（执行顺序）',ylabel='累计份额中的左上格比例（%）')
    a.set_title('B  生成过程中的持续偏向',loc='left',fontweight='bold');a.legend(frameon=False,fontsize=10)
    for a,k,title in [(axs[1,0],'t1_I0_vit','C  第1轮原图：层与头'),(axs[1,1],'t3_I2_vit','D  第3轮最新历史图：层与头')]:
        ar=np.asarray(d[k]['layer_head_sums']);ratio=np.divide(ar[0],ar[1],out=np.full((36,16),np.nan),where=ar[1]>0)
        cmap=plt.colormaps['YlGnBu'].copy();cmap.set_bad('#DDDDDD')
        im=a.imshow(ratio*100,aspect='auto',vmin=0,vmax=100,cmap=cmap,interpolation='nearest')
        a.set(xticks=[0,3,7,11,15],xticklabels=[1,4,8,12,16],yticks=[0,5,11,17,23,29,35],yticklabels=[1,6,12,18,24,30,36],xlabel='注意力头（从1计数）',ylabel='模型层（从1计数）')
        a.set_title(title,loc='left',fontweight='bold')
        fig.colorbar(im,ax=a,label='该层、头的累计份额中左上格比例（%）',shrink=.9)
    fig.suptitle('ViT 左上格偏向：归一化、生成阶段与层／头分布',fontsize=16,fontweight='bold')
    fig.supxlabel('A：会话等权；B–D：在标明范围内累计实际注意力后计算比例。灰格为累计图像份额为0。',fontsize=10,color='#596A78')
    save(fig,'11_vit_bias_diagnosis')
    head_diagnostics={}
    for k in ['t1_I0_vit','t3_I2_vit']:
        ar=np.asarray(d[k]['layer_head_sums'])
        ratio=np.divide(ar[0],ar[1],out=np.zeros((36,16)),where=ar[1]>0)
        mask=ratio>.5
        head_diagnostics[k]=dict(threshold=.5,n_heads=int(mask.sum()),
            fraction_of_image_attention=float(ar[1][mask].sum()/ar[1].sum()),
            fraction_of_corner_attention=float(ar[0][mask].sum()/ar[0].sum()))
    (ROOT/'head_mass_diagnostics.json').write_text(json.dumps(head_diagnostics,indent=2)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'figures').iterdir()}
    (ROOT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
