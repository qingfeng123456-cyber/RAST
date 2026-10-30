"""Publication plots from saved analysis arrays; no neural runtime required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE=Path(__file__).resolve().parent
ANALYSIS=HERE.parent/'analysis'
BLUE='#0072B2';ORANGE='#D55E00';TEAL='#009E73';PURPLE='#8064A2'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.bbox':'tight','savefig.pad_inches':.045})

def table(name,**kw):return pd.read_csv(ANALYSIS/name,**kw)
def js(name):return json.loads((ANALYSIS/name).read_text(encoding='utf-8'))
def arrays(name):
    with np.load(ANALYSIS/name,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def panel(ax,letter):ax.text(-.12,1.04,f'({letter})',transform=ax.transAxes,fontsize=9,va='bottom')
def heat(ax,x,vmin=0,vmax=None,cmap='viridis',**kwargs):
    palette=plt.get_cmap(cmap).copy();palette.set_bad('#dedede')
    return ax.imshow(np.ma.masked_invalid(x),aspect='auto',origin='lower',interpolation='nearest',cmap=palette,vmin=vmin,vmax=vmax,**kwargs)
def bg(ax):
    p=ANALYSIS/'training_position_density.npz'
    if p.exists():
        support=arrays(p.name);den=np.ma.masked_where(support['density']<.2,support['density'])
        ax.imshow(den,origin='lower',extent=support['extent'],cmap='Greys',vmin=.2,vmax=1,alpha=.23,zorder=0)
    ax.set_aspect('equal');ax.set_xlim(-2400,1900);ax.set_ylim(-1400,3400)


def fig_context_shifts():
    summary=js('behavior_summary.json');e=table('equipment_pattern_profiles.csv');temp=table('paired_time_profiles.csv').set_index('target_time');d=table('equipment_region_match_differences.csv',dtype={'group':str})
    k=summary['archetypes'];tiers=['Lower','Middle','Upper'];matrix=e.pivot(index='equipment_tier',columns='pattern',values='proportion').reindex(tiers)
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.75),gridspec_kw={'width_ratios':[.9,1.15,1.25]},layout='constrained')
    im=heat(axes[0],matrix.to_numpy(),0,max(.5,float(matrix.to_numpy().max())))
    axes[0].set(xticks=range(k),xticklabels=[f'P{i}' for i in range(k)],yticks=range(3),yticklabels=[f'{tier}\n{int(e[e.equipment_tier==tier].iloc[0]["rounds"])} rounds' for tier in tiers],xlabel='Deployment archetype',ylabel='Initial equipment tertile');fig.colorbar(im,ax=axes[0],label='Match-balanced usage',fraction=.055,pad=.025)
    im=heat(axes[1],temp[[f'R{i}' for i in range(12)]].to_numpy(),0,.4)
    axes[1].set(xticks=range(12),xticklabels=range(12),yticks=range(3),yticklabels=[f'{int(t)} s' for t in temp.index],xlabel='Region',ylabel='Matched round time');axes[1].tick_params(axis='x',labelsize=6);fig.colorbar(im,ax=axes[1],label='CT survivor occupancy share',fraction=.055,pad=.025)
    ax=axes[2];ax.axvline(0,color='.55',linestyle='--',linewidth=.8)
    records=[]
    for i in range(12):
        values=d[d.region==f'R{i}'].delta_upper_minus_lower.to_numpy();rng=np.random.default_rng(2026+i)
        vals=values[rng.integers(len(values),size=(10000,len(values)))].mean(1);lo,hi=np.quantile(vals,[.025,.975]);mean=values.mean()
        ax.errorbar([mean],[i],xerr=[[max(mean-lo,0)],[max(hi-mean,0)]],fmt='o',markersize=3,capsize=1.5,color=BLUE)
        records.append(dict(region=f'R{i}',delta=float(mean),ci_low=float(lo),ci_high=float(hi),matches=len(values)))
    pd.DataFrame(records).to_csv(ANALYSIS/'equipment_region_difference_summary.csv',index=False)
    ax.set(yticks=range(12),yticklabels=[f'R{i}' for i in range(12)],xlabel='Upper − lower equipment\noccupancy share',ylabel='Region');ax.grid(axis='x',alpha=.15)
    for a,l in zip(axes,'abc'):panel(a,l)
    q=summary['equipment_tertile_thresholds'];caption=(f'Context-conditioned deployment and temporal redistribution. Equipment is CT total equipment value at freeze end, with lower/middle/upper cutoffs {q[0]:.0f}/{q[1]:.0f}, fitted on training rounds. '
      f'(a) Match-balanced archetype usage at 15 s with five CT alive; (b) normalized survivor occupancy at 15/45/75 s on the same {summary["temporal_paired_rounds"]} rounds from {summary["temporal_paired_matches"]} matches; '
      f'(c) within-match upper-minus-lower equipment occupancy differences at 15 s, restricted to matches with at least two rounds in each tier; whiskers are match-bootstrap 95% intervals. '
      'All displayed profiles use equal match weighting.')
    return fig,caption,['equipment_pattern_profiles.csv','paired_time_profiles.csv','equipment_region_match_differences.csv','equipment_region_difference_summary.csv','behavior_summary.json']

def fig_match_comparison():
    m=table('match_comparison.csv',dtype={'group':str});p=js('performance_summary.json');s=table('population_conditioned_performance.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.15),gridspec_kw={'width_ratios':[1,1.25]},layout='constrained');ax=axes[0]
    x=-m.delta_energy_score;y=-m.delta_exact_count_nll;ax.axvline(0,color='.6',linestyle='--',linewidth=.8);ax.axhline(0,color='.6',linestyle='--',linewidth=.8)
    ax.scatter(x,y,color=BLUE,s=28,alpha=.9)
    for i,(xx,yy) in enumerate(zip(x,y)):ax.annotate(str(i+1),(xx,yy),xytext=((3 if i%2 else -7),(5 if i%3 else -8)),textcoords='offset points',fontsize=6.5)
    mx,my=x.mean(),y.mean();cx,cy=p['match_es_gain_ci'],p['match_nll_gain_ci']
    ax.errorbar([mx],[my],xerr=[[mx-cx[0]],[cx[1]-mx]],yerr=[[my-cy[0]],[cy[1]-my]],fmt='D',color=ORANGE,markersize=5,capsize=3,label='Match mean / 95% CI')
    ax.set(xlabel='ES reduction vs. vanilla',ylabel='Joint NLL reduction vs. vanilla');ax.legend(loc='lower right',frameon=True,fontsize=6.5);ax.margins(.2)
    metrics=['energy_score','exact_count_nll','pair_presence_brier','count_mae'];matrix=s.pivot(index='n',columns='metric',values='relative_gain_percent')[metrics]
    ax=axes[1];lim=max(float(np.nanmax(abs(matrix.to_numpy()))),1);im=heat(ax,matrix.to_numpy(),-lim,lim,cmap='RdBu')
    ax.set(xticks=range(4),xticklabels=['ES','Joint NLL','Pair Brier','Count MAE'],yticks=range(5),yticklabels=range(1,6),xlabel='Metric',ylabel='Known hidden-alive population N')
    ax.tick_params(axis='x',labelsize=7)
    for i in range(5):
        for j in range(4):ax.text(j,i,f'{matrix.iloc[i,j]:.1f}%' if np.isfinite(matrix.iloc[i,j]) else '—',ha='center',va='center',fontsize=7,color='white' if abs(matrix.iloc[i,j])>.65*lim else 'black')
    fig.colorbar(im,ax=ax,label='Relative reduction (%)',fraction=.04,pad=.025)
    for a,l in zip(axes,'ab'):panel(a,l)
    caption=('Held-out predictive gains of the region-aligned model over the vanilla Transformer. (a) Each point is one of 14 held-out matches, after equal averaging across the three seeds; positive coordinates favor the final model. '
     'The diamond and marginal intervals summarize equally weighted matches (10,000 match resamples, seed 2026), conditional on these trained runs. '
     '(b) Gains at each known hidden-alive population, using the same within-match then across-match and seed aggregation. All five populations and all displayed metrics are retained. '
     'The evaluation contains 27,034 cutoff histories from 293 rounds.')
    return fig,caption,['match_comparison.csv','population_conditioned_performance.csv','performance_summary.json']

def fig_readout():
    selection=js('diagnostic_selection.json');c=arrays('paper_case_data.npz');ids=selection['case_rows'][:2];probes=table('frozen_mechanism_probes.csv',dtype={'group':str});mat=probes.pivot(index='group',columns='seed',values='delta_readout_nll')
    fig=plt.figure(figsize=(7.2,3.7),layout='constrained');gs=fig.add_gridspec(1,3,width_ratios=[1,1,.9]);axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    data=[np.where(c[f'case{j}_readout_remaining'][:,None]>0,c[f'case{j}_readout_attention'].mean(1)-1/12,np.nan) for j in ids];lim=max(np.nanmax(np.abs(x)) for x in data)
    for i,(j,a) in enumerate(zip(ids,data)):
        im=heat(axes[i],a,-lim,lim,cmap='RdBu_r');axes[i].set(xticks=range(0,12,2),xticklabels=range(0,12,2),yticks=range(0,11,2),yticklabels=range(0,11,2),xlabel=f'Memory region · case {j}',ylabel='Allocated region / AR step')
    fig.colorbar(im,ax=axes[:2],label='Readout weight − 1/12',fraction=.025,pad=.02)
    ax=axes[2];im=heat(ax,mat.to_numpy(),0,float(mat.to_numpy().max()),cmap='magma_r')
    ax.set(xticks=range(3),xticklabels=['2026','2027','2028'],yticks=range(14),yticklabels=[str(i+1) for i in range(14)],xlabel='Training seed',ylabel='Held-out match');ax.tick_params(axis='x',rotation=35)
    fig.colorbar(im,ax=ax,label='Frozen readout bypass ΔNLL',fraction=.045,pad=.02)
    for a,l in zip(axes,'abc'):panel(a,l)
    caption=(f'Regional readout mechanisms of the final model. (a,b) Head-mean attention minus uniform for predefined cases {ids}; gray rows have zero remaining population, and the final region is determined by conservation. '
      '(c) Exact NLL after frozen readout bypass minus the unmodified checkpoint, averaged within each of the 14 matches, for all three seeds on the same 128 prespecified cutoffs. '
      'Positive values show dependence of the learned predictor on its readout. Attention panels use seed 2026, validation-selected epoch 5 after 13 completed epochs. '
      'Readout panels are teacher-forced conditional likelihood traces; the right panel is a frozen-checkpoint intervention.')
    return fig,caption,['paper_case_data.npz','frozen_mechanism_probes.csv','diagnostic_selection.json']

def fig_joint_cases():
    sel=js('diagnostic_selection.json');c=arrays('paper_case_data.npz');coords=table('region_centroids.csv');ids=sel['case_rows'][:3]
    fig,axes=plt.subplots(3,3,figsize=(7.2,5.55),layout='constrained');N=max(sel['rows'][j]['n'] for j in ids);images={}
    for col,j in enumerate(ids):
        draws=c[f'case{j}_draws'];values=[draws.mean(0),draws.std(0),c[f'case{j}_truth']]
        for row,vals in enumerate(values):
            ax=axes[row,col];bg(ax);im=ax.scatter(coords.x,coords.y,c=vals,s=52,cmap='viridis',vmin=0,vmax=[N,N/2,N][row],edgecolors='.35',linewidths=.45,zorder=2);images[row]=im
            for i,p in coords.iterrows():ax.annotate(str(i),(p.x,p.y),xytext=(2,3),textcoords='offset points',fontsize=5.8)
            ax.set(xticks=[-1500,0,1500],yticks=[-1000,1000,3000]);ax.tick_params(labelsize=6)
            if row==0:ax.text(.5,1.03,f'Case {j}: N={sel["rows"][j]["n"]}',ha='center',transform=ax.transAxes,fontsize=8)
            if row==2:ax.set_xlabel('Game X (units)')
            if col==0:ax.set_ylabel(['Predictive mean','Predictive SD','Observed count'][row]+'\nGame Y (units)')
    for row in range(3):fig.colorbar(images[row],ax=list(axes[row]),label=['Count','Count SD','Count'][row],fraction=.018,pad=.015,shrink=.85)
    caption=(f'Final-model occupancy and uncertainty for the unchanged predefined cases {ids}. Columns identify cases; rows show the mean and standard deviation of 256 free-running count-vector samples, then the single observed target count. '
     'All count vectors conserve the known hidden-alive population. Coordinates are training-fitted region centroids over a light backdrop of empirical training trajectory density. '
     'Common count/SD limits are set by the displayed populations. Cases use representative seed 2026, validation-selected epoch 5.')
    return fig,caption,['paper_case_data.npz','region_centroids.csv','training_position_density.npz','diagnostic_selection.json']

def fig_space_time():
    sel=js('diagnostic_selection.json');c=arrays('paper_case_data.npz');j=sel['case_rows'][0];time=c[f'case{j}_time'];att=c[f'case{j}_temporal_attention'].mean(1)
    edges=np.arange(np.floor(time.min()/5)*5,np.ceil(time.max()/5)*5+5,5);bins=np.clip(np.searchsorted(edges,time,side='right')-1,0,len(edges)-2);mass=np.zeros((len(att),12,len(edges)-1))
    for ix,cell in enumerate(bins):mass[:,:,cell]+=att[:,:,ix]
    np.savez_compressed(ANALYSIS/'space_time_attention_bins.npz',edges=edges,mass=mass,case=j)
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.6),layout='constrained');spatial=c[f'case{j}_spatial_attention'].mean(1)
    for col,layer in enumerate([0,len(att)-1]):
        ax=axes[0,col];im=ax.pcolormesh(edges,np.arange(13)-.5,mass[layer],cmap='viridis',vmin=0,vmax=mass.max(),shading='flat');ax.axvline(0,color='white',linestyle='--',linewidth=.7);ax.set(xlim=(time.min(),time.max()),xlabel='Seconds relative to freeze end',ylabel=f'Region · layer {layer+1}');fig.colorbar(im,ax=ax,label='Attention mass / 5 s',fraction=.035,pad=.02)
        ax=axes[1,col];im=heat(ax,spatial[layer],0,spatial.max());ax.set(xlabel='Key region',ylabel=f'Query region · layer {layer+1}',xticks=range(0,12,2),yticks=range(0,12,2));fig.colorbar(im,ax=ax,label='Spatial attention weight',fraction=.035,pad=.02)
    for a,l in zip(axes.flat,'abcd'):panel(a,l)
    caption=(f'Spatial and complete-history temporal attention of the final model for predefined case {j}, seed 2026. Top: last-query temporal attention, averaged over all heads and summed into fixed 5-second bins, for the first and last encoder layers. '
     'Negative time is the buy phase; the displayed history ends at the observed cutoff. Bottom: head-mean spatial attention at that last real time. Raw tokens, timestamps and all four layer/head arrays are saved.')
    return fig,caption,['paper_case_data.npz','space_time_attention_bins.npz','diagnostic_selection.json']

def fig_joint_law():
    sel=js('diagnostic_selection.json');c=arrays('paper_case_data.npz');ids=sel['case_rows'][:4]
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.65),layout='constrained');records=[]
    for ax,j,letter in zip(axes.flat,ids,'abcd'):
        draws=c[f'case{j}_draws'].astype(int);n=int(c[f'case{j}_truth'].sum());a,b=np.argsort(draws.var(0))[-2:]
        law=np.zeros((n+1,n+1));np.add.at(law,(draws[:,a],draws[:,b]),1/len(draws))
        impossible=np.add.outer(np.arange(n+1),np.arange(n+1))>n;law[impossible]=np.nan
        im=heat(ax,law,0,1);ax.set_aspect('equal')
        truth=c[f'case{j}_truth'];ax.scatter([truth[b]],[truth[a]],marker='x',s=42,color=ORANGE,linewidths=1.5)
        ax.set(xticks=range(n+1),yticks=range(n+1),xlabel=f'Count in R{b}',ylabel=f'Count in R{a}');panel(ax,letter)
        for aa in range(n+1):
            for bb in range(n+1):
                records.append(dict(case=j,n=n,region_y=int(a),region_x=int(b),count_y=aa,count_x=bb,probability=None if impossible[aa,bb] else float(law[aa,bb]),feasible=not bool(impossible[aa,bb])))
    fig.colorbar(im,ax=list(axes.flat),label='Joint predictive probability',shrink=.75,fraction=.025,pad=.025)
    pd.DataFrame(records).to_csv(ANALYSIS/'joint_count_law_source.csv',index=False)
    caption=(f'Two-region joint count laws of the final model in predefined cases {ids}, estimated from 256 free-running samples. In each case the two regions with the largest predictive count variance are shown. '
     'Gray cells have total count greater than the known hidden-alive population; orange crosses mark the single observed target. The shared probability scale is 0–1. Seed2026, validation-selected epoch 5.')
    return fig,caption,['paper_case_data.npz','joint_count_law_source.csv','diagnostic_selection.json']

def main(name=None):
    names=[name] if name else ['fig_context_shifts','fig_match_comparison','fig_readout','fig_joint_cases','fig_space_time','fig_joint_law']
    manifest_path=ANALYSIS/'figure_manifest.json';manifest=js(manifest_path.name) if manifest_path.exists() else {}
    for key in names:
        fig,caption,sources=globals()[key]()
        for fmt in ['pdf','svg','png']:fig.savefig(HERE/f'{key}.{fmt}',dpi=350,bbox_inches='tight',pad_inches=.045)
        plt.close(fig);manifest[key]=dict(caption=caption,sources=sources,formats=['pdf','svg','png'],model='improved_transformer',inference_or_training=False)
        print(key)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ANALYSIS/'FIGURE_CAPTIONS.md').write_text('\n\n'.join(f'## {key}\n\n{row["caption"]}\n\nSources: '+', '.join(row['sources']) for key,row in manifest.items()),encoding='utf-8')
if __name__=='__main__':main()
