"""Verify derived paper artifacts without training or neural inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'behavior_summary.json').read_text())
early=pd.read_csv(ROOT/'early_deployments.csv',dtype={'group':str})
assert len(early)==summary['early_15s_five_alive_known_rounds']==885
assert len(early[['group','round']].drop_duplicates())==len(early)
assert (early.ct_alive==5).all() and early.group.nunique()==51
assert len(early[early.split=='train'])==535
regions=[f'R{i}' for i in range(12)]
np.testing.assert_allclose(early[regions].sum(1),1,atol=1e-7)
proto=pd.read_csv(ROOT/'deployment_prototypes.csv').set_index('pattern')
assert len(proto)==summary['archetypes']==2
np.testing.assert_allclose(proto.sum(1),1,atol=1e-7)
teams=pd.read_csv(ROOT/'team_pattern_profiles.csv');columns=[c for c in teams if c.startswith('P')]
assert len(teams)==21 and (teams.matches>=2).all() and (teams['rounds']>=10).all()
np.testing.assert_allclose(teams[columns].sum(1),1,atol=1e-7)
time=pd.read_csv(ROOT/'paired_time_profiles.csv')
assert set(time.target_time)=={15,45,75} and (time['rounds']==667).all()
np.testing.assert_allclose(time[regions].sum(1),1,atol=1e-7)
perf=pd.read_csv(ROOT/'deployment_conditioned_performance.csv')
assert (perf.minimum_elapsed_seconds==15).all() and perf['rounds'].sum()==246
assert (perf.energy_score_better_seeds==3).all()
with np.load(ROOT/'paper_case_data.npz',allow_pickle=False) as c:
    for key in c.files:
        if key.endswith('_draws'):
            n=int(c[key.replace('_draws','_truth')].sum())
            assert c[key].shape==(256,12) and (c[key].sum(-1)==n).all()
figures=json.loads((ROOT/'figure_manifest.json').read_text())
for name,row in figures.items():
    assert row['model']=='improved_transformer' and not row['inference_or_training']
    for ext in row['formats']:assert (ROOT.parent/'figures'/f'{name}.{ext}').stat().st_size>1000
    for source in row['sources']:assert (ROOT/source).is_file()
report=dict(status='PASS',neural_training=False,neural_inference=False,figures=len(figures),formats=['pdf','svg','png'],
    early_rounds=885,training_clustering_rounds=535,team_profiles=21,paired_temporal_rounds=667,
    deployment_evaluation_minimum_live_seconds=15)
(ROOT/'analysis_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
