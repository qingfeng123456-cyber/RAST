"""Recompute publication summaries from the saved cutoff scores, without ML imports."""
import argparse
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'results/paper_model');args=p.parse_args()
    run=args.run.resolve()
    cfg=json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
    scores=pd.read_csv(run/'all_test_scores.csv',dtype={'group':str,'demo':str,'sample_id':str})
    assert set(scores.model)==set(cfg['models'])
    assert set(scores.seed)==set(cfg['seeds'])
    identities=None
    for (model,seed),group in scores.groupby(['model','seed']):
        assert not group.sample_id.duplicated().any()
        current=set(group.sample_id)
        if identities is None:identities=current
        else:assert identities==current
    metrics=[m for m in ['energy_score','exact_count_nll','count_brier','presence_brier','pair_presence_brier','count_mae'] if m in scores]
    by_match=scores.groupby(['model','seed','group'])[metrics].mean().reset_index()
    by_seed=by_match.groupby(['model','seed'])[metrics].mean().reset_index()
    long=by_seed.melt(id_vars=['model','seed'],value_vars=metrics,var_name='metric',value_name='value').dropna()
    summary=long.groupby(['model','metric']).value.agg(mean='mean',sample_std='std',seeds='count').reset_index()
    by_seed.to_csv(run/'recomputed_per_seed.csv',index=False)
    summary.to_csv(run/'recomputed_summary.csv',index=False)
    print(summary.to_string(index=False))
