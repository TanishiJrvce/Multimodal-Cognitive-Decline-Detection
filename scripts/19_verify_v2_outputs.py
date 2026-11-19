"""Validate complete V2 experiment provenance and independently reload every checkpoint."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from mcd.models import FusionNet,apply_modality_mask
from mcd.protocol import load_candidates,load_embeddings,VARIANTS
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'data/results/v2'
def main():
 torch.set_num_threads(1);d,names=load_candidates(ROOT);W,B=load_embeddings(ROOT,d,['V_joint']);X=d[names].to_numpy(float);checks=[];j=pd.read_csv(R/'training_results.csv');assert len(j)==125
 for z in j.itertuples():
  key=f'{z.experiment}_s{z.seed}_f{z.fold}_v2';a=json.loads((R/'audit'/f'{key}.json').read_text());g=pd.read_csv(R/'scores'/f'{key}.csv');cp=torch.load(ROOT/a['checkpoint'],map_location='cpu',weights_only=True);prep=cp['preprocessor'];inner=cp['inner_preprocessor']
  assert cp['experiment']==z.experiment and cp['model_id']==key
  assert set(inner['fit_ids'])==set(a['inner_train']);assert set(prep['fit_ids'])==set(a['outer_train']);assert not set(a['inner_train'])&set(a['inner_validation']);assert not set(a['outer_train'])&set(a['outer_test'])
  lookup=d.set_index('file').speaker
  assert not set(lookup.loc[a['inner_train']])&set(lookup.loc[a['inner_validation']]);assert not set(lookup.loc[a['outer_train']])&set(lookup.loc[a['outer_test']])
  assert set(g[g.split=='test'].file)==set(a['outer_test']);assert g.scored_by_model.eq(key).all();assert len(g)==275
  x=np.where(np.isfinite(X),X,np.array(prep['median']))[:,prep['indices']];x=(x-np.array(prep['mean']))/np.array(prep['scale']);f,w,b=apply_modality_mask(x,W,B,set(cp['mod_names']));m=FusionNet(len(prep['indices']));m.load_state_dict(cp['state_dict']);m.eval()
  with torch.no_grad():logit,ps=m(*[torch.tensor(v,dtype=torch.float32) for v in [f,w,b]]);prob=torch.sigmoid(logit).numpy();score=ps.numpy()
  aligned=g.set_index('file').loc[d.file];err=max(np.max(abs(prob-aligned.group_prob)),np.max(abs(score-aligned.prog_score)));assert err<1e-6
  test=g[g.split=='test'];assert abs(roc_auc_score(test.label,test.group_prob)-z.auc)<1e-10;assert abs(balanced_accuracy_score(test.label,test.group_prob>.5)-z.bac)<1e-10
  if z.experiment=='V_joint':assert z.ranking_gradient_steps>0 and z.pair_exposures>0
  checks.append(dict(model_id=key,n_outer_test=len(test),inner_fit_verified=True,outer_fit_verified=True,speaker_disjoint=True,checkpoint_reloaded=True,max_score_error=err,ranking_gradient_steps=int(z.ranking_gradient_steps)))
 states=pd.read_csv(R/'states_assignments_V_joint.csv');assert (states.model_id==states.scored_by_model).all();assert len(states)==6875
 # Every state-analysis input must equal its own model's actual score table, including training rows.
 for key,g in states.groupby('model_id'):
  own=pd.read_csv(R/'scores'/f'{key}.csv').set_index('file');assert np.allclose(g.prog_score,own.loc[g.file].prog_score,rtol=0,atol=1e-12)
 pd.DataFrame(checks).to_csv(R/'verification_checks.csv',index=False);print(f'PASS: {len(checks)} checkpoints reloaded; all predictions, memberships, fit IDs, metrics and own-model state inputs verified.')
if __name__=='__main__':main()
