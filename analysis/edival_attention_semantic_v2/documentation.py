"""Machine-readable metric contract, table dictionary and reproduction guide."""
import csv,gzip,json,pathlib,shutil,sys
from common import *
from metrics import METRICS

KEYS={
 'image_marker_turn':['session_id','generation_turn','image_index','modality','marker'],
 'within_type_pairs':['session_id','generation_turn','object_type','modality','older_index','newer_index'],
 'within_type_summary':['generation_turn','object_type','modality','older_index','newer_index','metric'],
 'group_turn':['session_id','generation_turn','group_name'],
 'image_combined_turn':['session_id','generation_turn','image_index'],
 'category_turn':['session_id','generation_turn','category'],
 'category_modality_turn':['session_id','generation_turn','category','modality'],
 'object_trajectories':['session_id','generation_turn','object_id'],
 'paired_changes':['session_id','object_id','turn_from','turn_to'],
 'trajectory_matrix':['session_id','generation_turn','object_type','object_index','modality'],
 'concentration_turn':['session_id','generation_turn','support_id'],
 'text_token_profiles':['session_id','generation_turn','support_id','token_ordinal'],
 'image_region_profiles':['session_id','generation_turn','support_id','row','col'],
 'composition_summary':['table','group_name','category','modality','generation_turn','metric'],
 'trajectory_summary':['object_type','object_index','modality','generation_turn','metric'],
 'paired_summary':['object_type','object_index','modality','turn_from','turn_to','metric'],
 'concentration_summary':['grouping','support_type','modality','object_index','object_role','generation_turn','metric','variant']}

def type_of(x):
    if x in ['True','False']:return 'boolean'
    try:int(x);return 'integer'
    except ValueError:pass
    try:float(x);return 'number'
    except ValueError:return 'string'

def build():
    spec=dict(task_id=TASK,dataset='EdiVal',input_protocol='lance-history-chat-v1',calculation_dtype='float64',array_storage_dtype='float32',
        input_axes=['step','layer','head'],n_positions_total=17280,input_observation='FP16 detached Q/K; FP32 saved statistics; positive CFG branch; target latent queries already averaged',
        group_mean='group_stats[...,stat_names.index("mean")], then FP64 mean over 30*36*16 positions; do not normalize original groups',
        density='mass / actual group token_count; null when count=0',enrichment='mass / (token_count / total_key_tokens); null when count=0',
        combined_density='(vit mass + vae mass)/(vit count + vae count)',
        exclusive_categories={'current_instruction':'Tt','history_instructions':'Tj, j<t','original_image':'I0_vit + I0_vae','history_images':'Ii_vit + Ii_vae, 1<=i<t','target_image':'target_image','context_other':'context_other','generation_markers':'generation_markers'},
        support='one Tj actual body-key tokens OR one image/encoding cells where region_token_count>0; zeros remain structural elements; N=0 invalid',
        metrics={
         'support_mass':'M=sum(x); mean over all positions, including M=0',
         'entropy_nats':'H=-sum(p*ln(p)); p=x/M only for M>0; 0 ln 0=0',
         'normalized_entropy':'H/ln(N); null for N=1',
         'effective_count':'exp(H)','effective_fraction':'exp(H)/N',
         'top1_share':'largest p','top3_share':'sum largest min(3,N) p','top5_share':'sum largest min(5,N) p',
         'top10pct_share':'sum largest ceil(0.1*N) p',
         'n90':'smallest k with descending cumulative share >= 0.9 - 1e-12','n90_fraction':'n90/N'},
        aggregation='compute nonlinear metric per position; average only metric-valid positions; within-role objects equally weighted within session/turn; then turns within session, then sessions; present_only for densities and concentration',
        validity='support_mass valid at M=0; conditional metrics only M>0; normalized_entropy additionally N>1; separate denominator columns for each metric',
        low_mass_threshold=1e-6,low_mass_definition='0<M<=1e-6; low_mass_fraction denominator=all 17280; low_mass_fraction_positive denominator=positive positions',
        sensitivity='M>1e-6; same metric-specific N validity',
        near_zero_tolerance=1e-12,near_zero_interpretation='floating point comparison only; not a scientific effect threshold',
        numeric_tolerances={'source_probability_sum':1e-3,'source_support_group_mass':1e-3,'npz_json_group_mean':5e-5,'n90_cumulative':1e-12,'derived_bounds_roundoff':1e-12,'array_fp32':'absolute error <= max(1e-7,6.1e-8*abs(FP64 value)) in independent samples'},
        adjustments='no smoothing, pseudocounts, normalization of original full-key groups, clipping, or source repair; bound checks allow documented FP64 rounding only',
        profiles={'mean_raw_mass':'mean_b x[b,k]','mean_conditional_share':'mean_{b:M[b]>0} x[b,k]/M[b]','pooled_conditional_share':'mean_b x[b,k] / mean_b M[b]','per_token_raw_mass':'mean_raw_mass / region_token_count; empty cell null'},
        bootstrap={'reps':2000,'seed':20261008,'unit':'original_image_cluster','percentiles':[2.5,97.5],
         'point_weighting':'session equal','resampling':'clusters uniformly with replacement, include every session of each selected cluster; sampled session sums divided by sampled valid-session counts',
         'stream':'NumPy default_rng(SeedSequence([seed, uint64 little-endian first eight SHA256 bytes of str(summary key)])); sorted clusters; batch 100',
         'interpretation':'uncertainty across observed sessions/original images; not multiple random model seeds'},
        missing={'structural_category':'mass=0 count=0 present=false; density=null; reason=structural_zero','not_yet_present_object':'no trajectory or pair; matrix null with reason=not_yet_present',
         'zero_mass':'conditional metrics and profiles null, reason=zero_support_mass','N1':'normalized_entropy null, reason=single_element_support','empty_cell':'not in N; density and conditional shares null, reason=empty_region'},
        arrays={'axes':['step','layer','head','support','metric'],'metric_values':'float32; invalid entries zero placeholders; use metric_valid boolean mask',
         'metric_valid':'per-position per-metric boolean validity','positive_mass':'M>0','low_mass':'0<M<=1e-6','support_N':'structural count per support','identity':'source_id and source_sha256',
         'ids':'step_ids, timesteps, layer_ids, head_ids, support_ids, metric_names'},
        target_image='whole group only; no internal concentration support',std_distinction='input group_stats std is population std across Query; summary std is sample ddof=1 across session values',
        offsets='actual original offsets; substring gives character fragment, not new tokenization or decoded token; overlapping offsets retained',
        no_analysis='no scoring, inference, GPU use, success filtering, head/stage conclusions, causal interpretation, or formal figures')
    atomic_json(ROOT/'metadata/metric_spec.json',spec)
    tables={}
    for path in sorted((ROOT/'tables').glob('*.csv.gz')):
        name=path.name.removesuffix('.csv.gz');types={};nulls={};n=0
        with gzip.open(path,'rt',encoding='utf8',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames
            for row in reader:
                n+=1
                for k,v in row.items():
                    if v=='':nulls[k]=True
                    else:types.setdefault(k,set()).add(type_of(v))
        desc={}
        for k in fields:
            typ=types.get(k,{'string'})
            if typ<={'integer','number'}:typ={'number'} if 'number' in typ else {'integer'}
            root_metric=next((m for m in METRICS if k==m or k==m+'_sensitivity'),None)
            if root_metric=='entropy_nats':unit='nat'
            elif root_metric in ['effective_count','n90']:unit='element count'
            elif root_metric in ['normalized_entropy','effective_fraction','n90_fraction']:unit='dimensionless'
            elif root_metric is not None:unit='probability'
            elif k.endswith('_n_valid') or k in ['support_N','n_positions_total','n_positions_positive','n_positions_low_mass','n_positions_conditional','token_count','total_key_tokens','region_token_count','top3_k','top5_k','top10pct_k']:unit='count'
            elif k in ['per_token_mass','delta_per_token_mass','per_token_raw_mass']:unit='probability per Key token'
            elif 'mass' in k or 'share' in k:unit='probability'
            else:unit='dimensionless'
            desc[k]=dict(types=sorted(typ),nullable=bool(nulls.get(k)),null_encoding='empty CSV field',unit=unit,
                null_reason='see metric-specific *_null_reason, density_null_reason, conditional_null_reason or matrix null_reason; union summary dimensions may be inapplicable; empty sensitivity=zero valid sensitivity positions; summary dispersion/CI undefined when insufficient valid sessions',
                description=('metric '+k.split('_sensitivity')[0]+'; definition in metric_spec.json' if k in METRICS else 'actual source-linked index, identity, validity, denominator or summary field'))
        tables[name]=dict(path=str(path.relative_to(ROOT)),rows=n,primary_key=KEYS[name],fields=desc,
            views='exclusive complete Key allocation' if name in ['group_turn','category_turn'] else ('additional nonexclusive image display; do not add to complete Key allocation' if name in ['image_combined_turn','category_modality_turn'] else 'source-linked derived view'),
            grouping='session equal; summary rows declare generation_turn, aggregation_level, conditioning, n_valid, n_clusters and bootstrap parameters')
    # Summary value units follow metric, not the column name alone.
    for name in ['composition_summary','trajectory_summary','paired_summary','concentration_summary']:
        for k in ['mean','std','p10','p25','p50','p75','p90','ci_low','ci_high','min','max']:
            tables[name]['fields'][k]['unit']='unit of metric column: nat for entropy_nats; count for effective_count/n90; probability for mass/share; dimensionless for normalized metrics/enrichment'
    atomic_json(ROOT/'schemas/tables.json',dict(task_id=TASK,format='UTF-8 CSV.GZ, strict JSON and pickle-free NPZ',boolean_encoding=['True','False'],tables=tables,
        notes={'source_id':'session_id/turn_t, joins metadata/source_manifest.jsonl.gz; paired tables use source_id_from/to; summaries link via group dimensions and their input table',
            'source_image_clusters':'metadata/source_image_clusters.csv.gz, primary key session_id; original_image_hash determines resampling unit',
            'generation_turn':'1,2,3 or overall in summaries','all_null_columns':'type remains documented semantic string; nullable'}))
    readme='''# EdiVal attention descriptive numerical archive

Task: edival_semantic_20261009_v2. This archive contains the three contracted descriptive analyses for all 572 sessions and 1716 turns. It contains numerical tables and reusable arrays; no inference, scoring or formal figures are generated.

Machine identity and input protocol are retained in metadata/run.json and source_manifest.jsonl.gz. Source files remain on their producing hosts. Every original attention file is bound to its receipt by size and full SHA256. The immutable candidate manifest binds the scientific artifacts checked by independent formulas and actual cache-identity audits. The final artifact manifest adds acceptance and resource evidence without changing reviewed artifacts.

Read CSV using Python gzip.open(..., "rt", encoding="utf8", newline="") and csv.DictReader. An empty field means null, with metric-specific reasons and validity fields. Booleans are True/False. Read arrays with numpy.load(..., allow_pickle=False). Consult schemas/tables.json for keys and columns and metadata/metric_spec.json for all formulas and denominator rules.

concentration_turn has one wide row per support (10296 rows), with metric, metric_n_valid, metric_null_reason, metric_sensitivity, and metric_sensitivity_n_valid columns. Array metric_values/metric_valid axes are step, layer, head, support, metric. Invalid array entries are zero placeholders and must be masked. Calculations and scalar tables use FP64; position arrays use FP32, with measured storage-error evidence in input_audit and independent_samples. The original saved query std is distinct from summary session std.

Original groups and the seven categories each cover the whole Key allocation. Image combined and modality category tables are additional display views. Their masses must not be added to the exclusive tables. Missing historical categories have structural zero mass, while not-yet-present individual objects have missing matrix entries. Density and concentration summaries condition on presence. Nonlinear concentration metrics are calculated at each position before averaging. Both conditional profile means and pooled shares are preserved with distinct names.

Summaries first average objects within a role/session/turn, then available turns within each session, then sessions equally. Bootstrap resamples original-image clusters (570 distinct clusters), preserving session-weighted point estimates. Arrays preserve positions only for reuse and validation; positions are not independent statistical samples.

Reproduction source is /home/chs/exp0_attention/Lance-infer-on-EdiVal/analysis/edival_attention_semantic_v2 and an immutable snapshot is metadata/analysis_source. The environment is /usr/bin/python 3.12.3 and NumPy 2.4.1, with package versions and hashes recorded. No additional environment was needed. From the project root, use the authenticated receipt-bound peer service implemented in scripts/serve_semantic_attention.py and a fresh system directory created by mktemp -d, then run:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python analysis/edival_attention_semantic_v2/pipeline.py --temp <mktemp-directory> --workers 8
/usr/bin/python analysis/edival_attention_semantic_v2/tests.py
/usr/bin/python analysis/edival_attention_semantic_v2/independent.py --temp <mktemp-directory>
/usr/bin/python analysis/edival_attention_semantic_v2/documentation.py
/usr/bin/python analysis/edival_attention_semantic_v2/validate.py
```

To reproduce after completion, create a new task output root rather than overwriting accepted artifacts. Change common.ROOT and the task-specific service credential/endpoint settings consistently, retaining the source identity and formulas. Do not overwrite immutable candidate artifacts. Remote NPZ files are downloaded one at a time per worker, fully verified, and removed immediately. Remove the mktemp directory after verification. The completed task services and credentials are temporary and intentionally removed after the acceptance handshake.
'''
    part=ROOT/'README.md.part';part.write_text(readme,encoding='utf8');os.replace(part,ROOT/'README.md')

if __name__=='__main__':build();print('Metric spec, schema and README written')
