"""Derive paper-only tables and game-context analysis; never fit a neural model.

All model results are immutable saved predictions. Deployment archetypes are a
descriptive training-fitted clustering of one fixed-time CT snapshot per round.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PROCESSED = ROOT/'data/processed'
LOG = ROOT/'results/paper_model'
FRAMES_CACHE = ROOT/'data/parser_cache/frames'
MODELS = ['knn','vanilla_transformer','improved_transformer']
SEEDS = [2026,2027,2028]
REGIONS = [f'R{i}' for i in range(12)]

def js(name, value):
    (BASE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def match_bootstrap(x):
    x=np.asarray(x,float);rng=np.random.default_rng(2026)
    vals=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return np.quantile(vals,[.025,.975]).tolist()

def data_analysis():
    with np.load(PROCESSED/'dataset.npz',allow_pickle=False) as archive:
        d={k:archive[k] for k in archive.files}
    frames=d['frames'];starts=np.unique(d['start']);ends=np.r_[starts[1:],len(frames)]
    meta=pd.DataFrame({k:d[k] for k in ['start','group','demo','round','ct_team','t_team','split']}).drop_duplicates('start').set_index('start')
    snapshots=[]
    for start,end in zip(starts,ends):
        f=frames[start:end];clock=f[:,28]*120
        entry=meta.loc[start].to_dict()
        zero=int(np.abs(clock).argmin())
        equip0=float(f[zero,30:198].reshape(12,14)[:,2].sum()*50000)
        for time in [15.,45.,75.]:
            pos=int(np.abs(clock-time).argmin());gap=float(abs(clock[pos]-time))
            if gap>.55:continue
            row=f[pos];count=np.rint(row[:12]*5).astype(int);alive=int(round(row[24]*5))
            if alive<1:continue
            assert count.sum()==alive
            share=count/alive
            snapshots.append(dict(**entry,target_time=int(time),actual_time=float(clock[pos]),
                ct_alive=alive,smoke_active=int(round(row[198:222].reshape(12,2)[:,0].sum()*5)),
                equipment_start=equip0,known_team=entry['ct_team']!='unknown',
                occupancy_entropy_bits=float(-np.sum(share[share>0]*np.log2(share[share>0]))),
                **dict(zip(REGIONS,share)),**{f'count_R{i}':int(v) for i,v in enumerate(count)}))
    snapshots=pd.DataFrame(snapshots)
    snapshots.to_csv(BASE/'fixed_time_round_snapshots.csv',index=False)
    early=snapshots.query('target_time==15 and ct_alive==5 and known_team').copy()
    train=early[early.split=='train']
    X=train[REGIONS].to_numpy();selection=[];fits={}
    with threadpool_limits(2):
        for k in range(2,9):
            km=KMeans(n_clusters=k,n_init=20,random_state=2026).fit(X)
            selection.append(dict(k=k,training_rounds=len(train),silhouette=float(silhouette_score(X,km.labels_))))
            fits[k]=km
    selected=max(selection,key=lambda r:r['silhouette'])['k'];km=fits[selected]
    # Spatially order the learned prototypes for stable, interpretable pattern IDs.
    centers=km.cluster_centers_;coords=np.asarray(json.loads((PROCESSED/'regions.json').read_text())['centers_scaled'])
    dominant=centers.argmax(1);ordering=np.lexsort((dominant,centers@coords[:,0]));newid=np.empty(selected,int);newid[ordering]=np.arange(selected)
    early['pattern']=newid[km.predict(early[REGIONS].to_numpy())]
    pd.DataFrame(selection).to_csv(BASE/'deployment_cluster_selection.csv',index=False)
    proto=pd.DataFrame(centers[ordering],columns=REGIONS);proto.insert(0,'pattern',np.arange(selected));proto.to_csv(BASE/'deployment_prototypes.csv',index=False)
    q=np.quantile(train.equipment_start,[1/3,2/3]);early['equipment_tier']=np.where(early.equipment_start<=q[0],'Lower',np.where(early.equipment_start<=q[1],'Middle','Upper'))
    early.to_csv(BASE/'early_deployments.csv',index=False)
    support=early.groupby('ct_team').agg(matches=('group','nunique'),rounds=('round','size'))
    selected_teams=support.query('matches>=2 and rounds>=10').sort_index().index
    counts=pd.crosstab([early.ct_team,early.group],early.pattern).reindex(columns=range(selected),fill_value=0)
    proportions=counts.div(counts.sum(1),axis=0)
    profiles=proportions.groupby('ct_team').mean().loc[selected_teams]
    result=profiles.rename(columns={i:f'P{i}' for i in range(selected)}).join(support)
    result.to_csv(BASE/'team_pattern_profiles.csv')
    regionprofiles=early.groupby(['ct_team','group'])[REGIONS].mean().groupby('ct_team').mean().loc[selected_teams]
    regionprofiles.join(support).to_csv(BASE/'team_region_profiles.csv')
    erows=[]
    for tier,sub in early.groupby('equipment_tier'):
        c=pd.crosstab(sub.group,sub.pattern).reindex(columns=range(selected),fill_value=0)
        p=c.div(c.sum(1),axis=0).mean(0)
        for pattern,value in p.items():erows.append(dict(equipment_tier=tier,pattern=int(pattern),proportion=float(value),rounds=len(sub),matches=sub.group.nunique()))
    pd.DataFrame(erows).to_csv(BASE/'equipment_pattern_profiles.csv',index=False)
    # Temporal redistribution on the same surviving rounds, equal match weights.
    common=snapshots[snapshots.known_team].groupby(['group','round']).target_time.nunique();keep=set(common[common==3].index)
    paired=snapshots[snapshots.apply(lambda x:(x.group,x['round']) in keep,axis=1)].copy()
    temporal=paired.groupby(['target_time','group'])[REGIONS+['occupancy_entropy_bits','ct_alive']].mean().groupby('target_time').mean()
    temporal['rounds']=paired.groupby('target_time').size();temporal['matches']=paired.groupby('target_time').group.nunique()
    temporal.to_csv(BASE/'paired_time_profiles.csv')
    # Equipment-conditioned region shifts at fixed live time and fixed own population.
    diffs=[]
    for group,sub in early.groupby('group'):
        lo=sub[sub.equipment_tier=='Lower'];hi=sub[sub.equipment_tier=='Upper']
        if len(lo)<2 or len(hi)<2:continue
        for region in REGIONS:diffs.append(dict(group=group,region=region,delta_upper_minus_lower=float(hi[region].mean()-lo[region].mean()),lower_rounds=len(lo),upper_rounds=len(hi)))
    pd.DataFrame(diffs).to_csv(BASE/'equipment_region_match_differences.csv',index=False)
    # Frozen, train-fitted pattern assignment provides a descriptive test stratum.
    labels=early[['group','round','pattern']].drop_duplicates(['group','round'])
    labels.to_csv(BASE/'round_pattern_assignment.csv',index=False)
    summary=dict(rounds=len(starts),fixed_time_records=len(snapshots),early_15s_five_alive_known_rounds=len(early),
        early_matches=early.group.nunique(),early_training_rounds=len(train),
        archetypes=selected,clustering_features='12 observed CT region occupancy proportions; fixed 15 live seconds, all 5 CT alive',
        cluster_fit='training matches only; K selected by training silhouette over K=2..8; seed2026; n_init20',
        equipment_tertile_thresholds=q.tolist(),equipment_semantics='CT total current equipment value at freeze end, reconstructed from own-player attributes; not cash or income',
        selected_team_count=len(selected_teams),team_min_matches=2,team_min_rounds=10,
        temporal_paired_rounds=len(keep),temporal_paired_matches=paired.group.nunique(),
        descriptive_scope='All 53 matches including training; one fixed-time observation per eligible round; not predictive holdout performance or causal tactical value')
    js('behavior_summary.json',summary)
    # Training-only empirical position-density backdrop from parsed coordinates.
    density=np.zeros((160,160));extent=np.array([-2400.,1900.,-1400.,3400.]);sources=[]
    cache=FRAMES_CACHE
    train_groups=set(d['group'][d['split']=='train'])
    for path in sorted(cache.glob('*.npz')):
        if path.name.split('_')[0] not in train_groups:continue
        with np.load(path,allow_pickle=False) as z:
            pts=np.concatenate([z['ct_xyz'][z['ct_alive']],z['t_xyz'][z['t_alive']]])
        hist,*_=np.histogram2d(pts[:,1],pts[:,0],bins=160,range=[[extent[2],extent[3]],[extent[0],extent[1]]]);density+=hist;sources.append(path.name)
    if len(sources):
        from scipy.ndimage import gaussian_filter
        smooth=gaussian_filter(density,1.0);smooth=np.log1p(smooth);smooth/=smooth.max()
        np.savez_compressed(BASE/'training_position_density.npz',density=smooth,extent=extent,training_matches=np.asarray(sources))
        js('training_position_density_source.json',dict(matches=len(sources),files=sources,semantics='training trajectory density, not navmesh or map artwork'))
    def record_path(path):
        try:return path.relative_to(ROOT).as_posix()
        except ValueError:return str(path)
    js('analysis_sources.json',dict(dataset=record_path(PROCESSED/'dataset.npz'),dataset_sha256=hashlib.sha256((PROCESSED/'dataset.npz').read_bytes()).hexdigest(),log=record_path(LOG),seeds=SEEDS,models=MODELS,path_base='repository root for relative paths'))

def prediction_analysis():
    scores=pd.read_csv(LOG/'all_test_scores.csv',dtype={'group':str,'demo':str,'sample_id':str});scores=scores[scores.model.isin(MODELS)].copy()
    metrics=['energy_score','exact_count_nll','count_brier','presence_brier','pair_presence_brier','count_mae']
    macro=scores.groupby(['model','seed','group'])[metrics].mean();perseed=macro.groupby(['model','seed']).mean()
    rows=[]
    for model in MODELS:
        for metric in metrics:
            vals=perseed.loc[model,metric].dropna()
            if len(vals):rows.append(dict(model=model,metric=metric,mean=float(vals.mean()),sample_std=float(vals.std(ddof=1)),seeds=len(vals)))
    pd.DataFrame(rows).to_csv(BASE/'paper_results.csv',index=False)
    pair=macro.xs('improved_transformer',level='model')-macro.xs('vanilla_transformer',level='model')
    bymatch=pair.groupby('group').mean();bymatch.rename(columns=lambda k:'delta_'+k).to_csv(BASE/'match_comparison.csv')
    # Full-score strata use the same match/seed protocol as the main table.
    condition=[]
    for n,sub in scores.groupby('n'):
        p=sub.groupby(['model','seed','group'])[metrics].mean().groupby('model').mean()
        for metric in metrics:
            a,b=p.loc['improved_transformer',metric],p.loc['vanilla_transformer',metric]
            condition.append(dict(n=int(n),metric=metric,improved=float(a),vanilla=float(b),relative_gain_percent=float(100*(b-a)/b) if b!=0 else np.nan,cutoffs_per_seed=int(len(sub)/9),matches=sub.group.nunique()))
    pd.DataFrame(condition).to_csv(BASE/'population_conditioned_performance.csv',index=False)
    labels=pd.read_csv(BASE/'round_pattern_assignment.csv',dtype={'group':str});strat=scores.merge(labels,on=['group','round'],how='inner')
    strat=strat[strat.elapsed>=15].copy()
    rows=[]
    for pattern,sub in strat.groupby('pattern'):
        p=sub.groupby(['model','seed','group'])[metrics].mean().groupby('model').mean()
        row=dict(pattern=int(pattern),improved_es=float(p.loc['improved_transformer','energy_score']),vanilla_es=float(p.loc['vanilla_transformer','energy_score']),
            relative_gain_percent=float(100*(1-p.loc['improved_transformer','energy_score']/p.loc['vanilla_transformer','energy_score'])),
            improved_nll=float(p.loc['improved_transformer','exact_count_nll']),vanilla_nll=float(p.loc['vanilla_transformer','exact_count_nll']),
            nll_relative_gain_percent=float(100*(1-p.loc['improved_transformer','exact_count_nll']/p.loc['vanilla_transformer','exact_count_nll'])),
            matches=sub.group.nunique(),rounds=sub[['group','round']].drop_duplicates().shape[0],cutoffs_per_method_seed=int(len(sub)/9),minimum_elapsed_seconds=15)
        for metric in ['energy_score','exact_count_nll']:
            values=sub.groupby(['model','seed','group'])[metric].mean().unstack('model')
            delta=values.improved_transformer-values.vanilla_transformer
            row[metric+'_better_seeds']=int((delta.groupby('seed').mean()<0).sum())
            row[metric+'_delta_ci_low'],row[metric+'_delta_ci_high']=match_bootstrap(delta.groupby('group').mean().to_numpy())
        rows.append(row)
    pd.DataFrame(rows).to_csv(BASE/'deployment_conditioned_performance.csv',index=False)
    # Copy only small, already saved arrays needed for mechanism redraws.
    selected=json.loads((LOG/'diagnostic_selection.json').read_text());js('diagnostic_selection.json',selected)
    caseids=selected['case_rows'];rows=selected['rows'];payload={}
    with np.load(LOG/'seed_2026/improved_transformer/diagnostics/cases.npz',allow_pickle=False) as z:payload.update({k:z[k] for k in z.files})
    with np.load(LOG/'seed_2026/improved_transformer/test_predictions.npz',allow_pickle=False) as z:
        for j in caseids:payload[f'case{j}_draws']=z['samples'][rows[j]['test_position']]
    np.savez_compressed(BASE/'paper_case_data.npz',**payload)
    with np.load(LOG/'seed_2026/improved_transformer/diagnostics/traces.npz',allow_pickle=False) as z:
        np.savez_compressed(BASE/'paper_trace_data.npz',**{k:z[k] for k in z.files})
    pd.DataFrame(json.loads((PROCESSED/'regions.json').read_text())['centers_scaled'],columns=['x','y','z']).to_csv(BASE/'region_centroids.csv',index_label='region')
    probes=[]
    for seed in SEEDS:
        table=pd.read_csv(LOG/f'seed_{seed}/improved_transformer/diagnostics/interventions.csv',dtype={'group':str})
        for group,sub in table.groupby('group'):
            probes.append(dict(seed=seed,group=group,cutoffs=len(sub),delta_readout_nll=float((sub.no_readout_nll-sub.baseline_nll).mean()),delta_gate_nll=float((sub.neutral_gate_nll-sub.baseline_nll).mean()),delta_distance_nll=float((sub.no_distance_nll-sub.baseline_nll).mean())))
    pd.DataFrame(probes).to_csv(BASE/'frozen_mechanism_probes.csv',index=False)
    js('performance_summary.json',dict(test_matches=14,test_rounds=293,test_cutoffs_per_method_seed=27034,seeds=SEEDS,
        es_gain_vs_vanilla=100*(1-perseed.loc['improved_transformer'].energy_score.mean()/perseed.loc['vanilla_transformer'].energy_score.mean()),
        es_gain_vs_knn=100*(1-perseed.loc['improved_transformer'].energy_score.mean()/perseed.loc['knn'].energy_score.mean()),
        nll_gain_vs_vanilla=100*(1-perseed.loc['improved_transformer'].exact_count_nll.mean()/perseed.loc['vanilla_transformer'].exact_count_nll.mean()),
        match_es_gain_ci=match_bootstrap(-bymatch.energy_score),match_nll_gain_ci=match_bootstrap(-bymatch.exact_count_nll)))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',type=Path,default=PROCESSED)
    parser.add_argument('--log',type=Path,default=LOG)
    parser.add_argument('--predictions-only',action='store_true')
    parser.add_argument('--frames-cache',type=Path,default=FRAMES_CACHE,help='Optional parsed-coordinate cache for rebuilding the preserved training density')
    args=parser.parse_args();PROCESSED=args.processed.resolve();LOG=args.log.resolve();FRAMES_CACHE=args.frames_cache.resolve()
    BASE.mkdir(parents=True,exist_ok=True)
    if not args.predictions_only:data_analysis()
    prediction_analysis()
    print((BASE/'behavior_summary.json').read_text())
    print((BASE/'performance_summary.json').read_text())
