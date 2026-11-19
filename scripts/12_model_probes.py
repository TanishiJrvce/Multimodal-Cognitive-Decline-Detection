"""Fixed-specification speaker-level logistic probes and OOF score structure analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Re-use shared globals defined in 11_group_analysis
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "group_analysis",
    Path(__file__).resolve().parent / "11_group_analysis.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
from group_analysis import *  # noqa: F401,F403 — intentional star import of shared state
from sklearn.metrics import roc_curve
from scipy.stats import rankdata

def auc_boot(y,s,n=5000):
 i1=np.flatnonzero(y==1);i0=np.flatnonzero(y==0);out=[]
 for _ in range(n):
  idx=np.r_[RNG.choice(i1,len(i1)),RNG.choice(i0,len(i0))];out.append(roc_auc_score(y[idx],s[idx]))
 return np.quantile(out,[.025,.975])
def predict(X,y,seed):
 p=np.zeros(len(y))
 for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
  model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=seed))
  model.fit(X[tr],y[tr]);p[te]=model.predict_proba(X[te])[:,1]
 return p
def probes():
 df,_=load();sp=df.groupby('speaker')[FEATURES+Q+['label','female']].mean();y=sp.label.values.astype(int)
 families={'Timing3':TIMING,'Language5':LING,'Voice6':VOICE,'Spectral23':SPEC,'TimingLanguage8':TIMING+LING,'All37':FEATURES,'Quality6':Q,'TimingQuality9':TIMING+Q,'AllQuality43':FEATURES+Q,'Gender1':['female']}
 preds={};rows=[];seeds=[]
 for name,cols in families.items():
  X=sp[cols].values;p=np.stack([predict(X,y,s) for s in range(5)]);preds[name]=p
  for s in range(5):seeds.append(dict(model=name,seed=s,auc=roc_auc_score(y,p[s]),bac=balanced_accuracy_score(y,p[s]>.5),brier=brier_score_loss(y,p[s])))
  mean=p.mean(0);lo,hi=auc_boot(y,mean);fpr,tpr,t=roc_curve(y,mean);idx=np.flatnonzero(fpr<=.2)[-1]
  rows.append(dict(model=name,n_features=len(cols),n_speakers=len(y),auc=roc_auc_score(y,mean),auc_low=lo,auc_high=hi,bac=balanced_accuracy_score(y,mean>.5),brier=brier_score_loss(y,mean),sensitivity_at_empirical_80specificity=tpr[idx],specificity_at_that_threshold=1-fpr[idx],threshold=t[idx]))
  print('PROBE',name,rows[-1],flush=True)
 save(pd.DataFrame(rows),'supplementary_probe_summary');save(pd.DataFrame(seeds),'supplementary_probe_runs')
 out=pd.DataFrame({'speaker_id':['S%03d'%i for i in range(len(y))],'label':y})
 for name,p in preds.items():
  for s in range(5):out[f'{name}_seed{s}']=p[s]
 save(out,'supplementary_oof_predictions')
 # Paired conditional bootstrap difference of two frozen OOF score vectors.
 contrasts=[];i1=np.flatnonzero(y==1);i0=np.flatnonzero(y==0)
 for l,r in [('All37','Quality6'),('TimingLanguage8','Timing3'),('AllQuality43','Quality6'),('TimingQuality9','Quality6')]:
  z=[];left=preds[l].mean(0);right=preds[r].mean(0)
  for _ in range(5000):
   ix=np.r_[RNG.choice(i1,len(i1)),RNG.choice(i0,len(i0))];z.append(roc_auc_score(y[ix],left[ix])-roc_auc_score(y[ix],right[ix]))
  contrasts.append(dict(left=l,right=r,auc_difference=roc_auc_score(y,left)-roc_auc_score(y,right),low=np.quantile(z,.025),high=np.quantile(z,.975)))
 save(pd.DataFrame(contrasts),'supplementary_probe_contrasts')
 # Full-pipeline permutation: refit scaler and classifier after label shuffle, seed 0 only.
 null=[]
 for name in ['Timing3','All37']:
  X=sp[families[name]].values;observed=roc_auc_score(y,preds[name][0]);values=[]
  for b in range(499):
   yp=RNG.permutation(y);v=roc_auc_score(yp,predict(X,yp,0));values.append(v);null.append(dict(model=name,iteration=b,auc=v))
  pv=(1+sum(v>=observed for v in values))/500
  print('PIPELINE PERMUTATION',name,observed,pv,flush=True)
  save(pd.DataFrame([dict(model=name,observed_seed0_auc=observed,permutation_p=pv,null_mean=np.mean(values),null_q95=np.quantile(values,.95))]),'permutation_'+name)
 save(pd.DataFrame(null),'supplementary_pipeline_permutations')

def score_structure():
 a=read('data/results/states_assignments.csv');df,m=load();o=read('data/processed/recording_order.csv').rename(columns={'recording':'file','split':'original_split'})
 # Score scales differ by fold. Percentiles use the training score distribution only.
 parts=[]
 for (seed,fold),g in a.groupby(['seed','fold']):
  training=np.sort(g.loc[g.split=='train','prog_score'].values);g=g.copy();g['score_percentile']=(np.searchsorted(training,g.prog_score,side='left')+.5*np.array([np.sum(training==x) for x in g.prog_score]))/len(training);parts.append(g)
 a=pd.concat(parts);v=a[a.split=='val'];p=v.groupby('file').agg(score=('score_percentile','mean'),score_sd=('score_percentile','std'),high_state_fraction=('state','mean'),label=('label','first'),speaker=('speaker','first')).reset_index();joined=p.merge(df.drop(columns=['speaker','label','split']),on='file')
 sp=joined.groupby('speaker')[['score','label']].mean();y=sp.label.to_numpy(int);s=sp.score.values;lo,hi=auc_boot(y,s)
 rows=[dict(level='speaker',auc=roc_auc_score(y,s),low=lo,high=hi,n=len(y))];rows.append(dict(level='recording',auc=roc_auc_score(joined.label,joined.score),low=np.nan,high=np.nan,n=len(joined)));save(pd.DataFrame(rows),'progression_score_group_discrimination')
 print('PROGRESSION SCORE AUC',rows,flush=True)
 # Each ordered person's scores are simultaneously held out in exactly one fold per seed.
 pairrows=[];personrows=[]
 for mode in ['train','val']:
  aa=a[a.split==mode].merge(o[['speaker','file','rank']],on=['speaker','file']);bucket=aa.groupby(['seed','fold','speaker','rank']).agg(score=('prog_score','mean'),state=('state','mean')).reset_index()
  for (seed,fold,speaker),g in bucket.groupby(['seed','fold','speaker']):
   g=g.sort_values('rank')
   if len(g)<2:continue
   concord=[]
   for r1,r2 in itertools.combinations(g.to_dict('records'),2):
    hit=float(r2['score']>r1['score'])+.5*float(r2['score']==r1['score']);concord.append(hit);pairrows.append(dict(mode=mode,seed=seed,fold=fold,speaker=speaker,hit=hit,rankgap=r2['rank']-r1['rank'],state_change=r2['state']-r1['state']))
   personrows.append(dict(mode=mode,seed=seed,fold=fold,speaker=speaker,concordance=np.mean(concord),endpoint_hit=float(g.iloc[-1].score>g.iloc[0].score),n_pairs=len(concord)))
 per=pd.DataFrame(personrows);pairs=pd.DataFrame(pairrows);summ=[]
 for mode in ['train','val']:
  # Average runs within person before bootstrap; people are the sampling unit.
  zz=per[per['mode']==mode].groupby('speaker').agg(concordance=('concordance','mean'),endpoint_hit=('endpoint_hit','mean'));ci=boot_mean(zz.concordance.values);ci2=boot_mean(zz.endpoint_hit.values)
  summ.append(dict(mode=mode,n_speakers=len(zz),concordance=zz.concordance.mean(),low=ci[0],high=ci[1],endpoint_hit=zz.endpoint_hit.mean(),endpoint_low=ci2[0],endpoint_high=ci2[1],model_person_evaluations=int((per['mode']==mode).sum())))
 save(pd.DataFrame(summ),'temporal_generalization');print('TEMPORAL',summ,flush=True)
 # Rank-permutation test on five-seed OOF consensus score, within person.
 zz=o[['speaker','file','rank']].merge(p[['file','score']],on='file').groupby(['speaker','rank']).score.mean().reset_index();arr=[]
 for speaker,g in zz.groupby('speaker'):
  if len(g)<2:continue
  vals=g.sort_values('rank').score.values;arr.append(vals)
 def conc(vals):return np.mean([float(vals[j]>vals[i])+.5*float(vals[j]==vals[i]) for i in range(len(vals)) for j in range(i+1,len(vals))])
 obs=np.mean([conc(x) for x in arr]);null=np.array([np.mean([conc(RNG.permutation(x)) for x in arr]) for _ in range(10000)]);save(pd.DataFrame([dict(n_speakers=len(arr),concordance=obs,permutation_p=(1+sum(null>=obs))/(1+len(null)),null_mean=null.mean(),null_q95=np.quantile(null,.95))]),'oof_consensus_temporal_permutation')
 # OOF feature associations, speakers equal-weighted; quality-adjusted rank residuals.
 z=joined.groupby('speaker')[FEATURES+Q+['score']].mean().dropna(subset=Q);cov=stats.rankdata(z[Q].values,axis=0);C=np.column_stack([np.ones(len(z)),StandardScaler().fit_transform(cov)]);inv=np.linalg.pinv(C);ry=stats.rankdata(z.score);resy=ry-C@(inv@ry);rows=[]
 for c in FEATURES+Q:
  rx=stats.rankdata(z[c]);rho,pv=stats.spearmanr(z[c],z.score);resx=rx-C@(inv@rx);partial=stats.pearsonr(resx,resy).statistic if c not in Q else np.nan
  # bootstrap speakers for unadjusted association
  vals=[]
  for _ in range(2000):
   ix=RNG.integers(0,len(z),len(z));vals.append(stats.spearmanr(z[c].values[ix],z.score.values[ix]).statistic)
  rows.append(dict(feature=c,rho=rho,p=pv,low=np.nanquantile(vals,.025),high=np.nanquantile(vals,.975),partial_rho_quality=partial,n_speakers=len(z)))
 rr=pd.DataFrame(rows);rr['q_bh_43']=bh(rr.p);save(rr,'oof_progression_feature_associations');print('SCORE features',rr.sort_values('p').head(10).round(4).to_dict('records'),flush=True)
 # OOF state probabilities and entropy. Five held-out states per recording, no forced alignment beyond low/high.
 save(pd.DataFrame([dict(n_recordings=len(p),always_low=int((p.high_state_fraction==0).sum()),always_high=int((p.high_state_fraction==1).sum()),variable=int(p.high_state_fraction.between(.01,.99).sum()),mean_percentile_sd=p.score_sd.mean())]),'oof_state_uncertainty')
 # Group enrichment separately in train and validation, first average within speaker/run.
 es=[]
 for (seed,fold,mode),g in a.groupby(['seed','fold','split']):
  # P(high-state | group), each speaker contributes equally within group.
  gg=g.groupby('speaker').agg(high=('state','mean'),label=('label','first'));u=gg.groupby('label').high.mean()
  es.append(dict(seed=seed,fold=fold,mode=mode,dementia_high_fraction=u.loc[1],comparison_high_fraction=u.loc[0],difference=u.loc[1]-u.loc[0]))
 save(pd.DataFrame(es),'state_enrichment_by_run')
 # Full rank and state agreement across models; folds are not independent samples.
 raw=a.pivot(index='file',columns=['seed','fold'],values='prog_score');lab=a.pivot(index='file',columns=['seed','fold'],values='state');rows=[]
 for c1,c2 in itertools.combinations(raw.columns,2):rows.append(dict(seed_a=c1[0],fold_a=c1[1],seed_b=c2[0],fold_b=c2[1],score_rank_rho=stats.spearmanr(raw[c1],raw[c2]).statistic,state_ari=adjusted_rand_score(lab[c1],lab[c2])))
 save(pd.DataFrame(rows),'cross_model_agreement')
 # Anonymised reproducible OOF summary.
 p=p.drop(columns='speaker');p['recording_id']=['R%03d'%i for i in range(len(p))];save(p.drop(columns='file'),'anonymized_oof_score_summary')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['probes','scores','all'],default='all');args=ap.parse_args()
 if args.stage in ['probes','all']:probes()
 if args.stage in ['scores','all']:score_structure()
