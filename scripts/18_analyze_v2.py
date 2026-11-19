"""Speaker-aware analyses of completed V2 models and same-model score tables."""
from pathlib import Path
import sys,json,itertools,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from mcd.protocol import load_candidates,QUALITY
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/results/v2';RNG=np.random.default_rng(20261120)
spec=importlib.util.spec_from_file_location('verified_v1',ROOT/'scripts/17_verify_committed_v1.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def save(d,n):pd.DataFrame(d).to_csv(OUT/(n+'.csv'),index=False)
def ci(x):
 x=np.asarray(x);return np.quantile(x[RNG.integers(0,len(x),(5000,len(x)))].mean(1),[.025,.975])
def main():
 d,names=load_candidates(ROOT);j=pd.read_csv(OUT/'training_results.csv');assert len(j)==125
 allscores=pd.concat([pd.read_csv(f) for f in sorted((OUT/'scores').glob('*.csv'))],ignore_index=True);p=allscores[allscores.split=='test'].copy();assert len(p)==6875
 p.to_csv(OUT/'oof_predictions.csv',index=False);m=[]
 for (exp,seed,fold),g in p.groupby(['experiment','seed','fold']):
  sp=g.groupby('speaker').agg(label=('label','first'),prob=('group_prob','mean'));m.append(dict(experiment=exp,seed=seed,fold=fold,auc=roc_auc_score(g.label,g.group_prob),bac=balanced_accuracy_score(g.label,g.group_prob>.5),speaker_auc=roc_auc_score(sp.label,sp.prob),speaker_bac=balanced_accuracy_score(sp.label,sp.prob>.5)))
 metrics=pd.DataFrame(m);metrics.to_csv(OUT/'verified_metrics.csv',index=False);summary=metrics.groupby('experiment')[['auc','bac','speaker_auc','speaker_bac']].agg(['mean','std']);summary.to_csv(OUT/'verified_summary.csv')
 pivot=metrics.pivot(index=['seed','fold'],columns='experiment',values='auc');diff=pivot.V_joint-pivot.V_cls;se=diff.std()*np.sqrt(1/25+1/4);crit=stats.t.ppf(.975,24)
 contrast=dict(mean_difference=diff.mean(),wins=int((diff>0).sum()),positive_seed_means=int((diff.groupby(level=0).mean()>0).sum()),seed_min=diff.groupby(level=0).mean().min(),seed_max=diff.groupby(level=0).mean().max(),heuristic_low=diff.mean()-crit*se,heuristic_high=diff.mean()+crit*se);save([contrast],'joint_classification_contrast')
 # All same-model training scores are available; compute train-CDF percentiles honestly.
 parts=[]
 for _,g in allscores[allscores.experiment=='V_joint'].groupby('model_id'):
  assert g.scored_by_model.eq(g.model_id).all();tr=np.sort(g[g.split=='train'].prog_score.to_numpy());g=g.copy();g['percentile']=(np.searchsorted(tr,g.prog_score,side='left')+.5*(np.searchsorted(tr,g.prog_score,side='right')-np.searchsorted(tr,g.prog_score,side='left')))/len(tr);parts.append(g)
 norm=pd.concat(parts);norm.to_csv(OUT/'joint_normalized_scores.csv',index=False)
 order=pd.read_csv(ROOT/'data/processed/recording_order.csv').rename(columns={'recording':'file'}).dropna(subset=['rank']);temporal=[]
 for mode in ['train','test']:
  z=v.temporal_records(allscores[allscores.split==mode],order);z['split']=mode;temporal.append(z)
 per=pd.concat(temporal);per.to_csv(OUT/'temporal_person_records.csv',index=False);ts=[]
 for (exp,mode),g in per.groupby(['experiment','split']):
  z=g.groupby('speaker')[['concordance','endpoint_hit']].mean();lo,hi=ci(z.concordance);el,eh=ci(z.endpoint_hit);ts.append(dict(experiment=exp,split=mode,n_speakers=len(z),evaluations=len(g),concordance=z.concordance.mean(),low=lo,high=hi,endpoint=z.endpoint_hit.mean(),endpoint_low=el,endpoint_high=eh))
 save(ts,'temporal_summary')
 joint=p[p.experiment=='V_joint'].merge(order[['file','speaker','rank']],on=['file','speaker']);bucket=joint.groupby(['speaker','seed','rank']).prog_score.mean().reset_index();null=np.zeros(10000);observed=[]
 for speaker,g in bucket.groupby('speaker'):
  mat=g.pivot(index='seed',columns='rank',values='prog_score').sort_index(axis=1).to_numpy()
  if mat.shape[1]<2:continue
  def conc(m):return np.mean([np.mean((m[:,j]>m[:,i])+.5*(m[:,j]==m[:,i])) for i in range(m.shape[1]) for j in range(i+1,m.shape[1])])
  observed.append(conc(mat));values=np.array([conc(mat[:,perm]) for perm in itertools.permutations(range(mat.shape[1]))]);null+=values[RNG.integers(0,len(values),len(null))]
 null/=len(observed);obs=np.mean(observed);permutation=dict(n_speakers=len(observed),concordance=obs,one_sided_p=(1+(null>=obs-1e-12).sum())/10001,two_sided_p=(1+(abs(null-.5)>=abs(obs-.5)-1e-12).sum())/10001);save([permutation],'temporal_permutation')
 # Conditional OOF speaker-ensemble uncertainty and matched quality comparator.
 bp=pd.read_csv(OUT/'baseline_oof.csv');ensembles={}
 for exp,g in p.groupby('experiment'):ensembles[exp]=g.groupby(['speaker','file']).agg(label=('label','first'),prob=('group_prob','mean')).groupby('speaker').agg(label=('label','first'),prob=('prob','mean'))
 q=bp[bp.model=='B7_quality_RF'];ensembles['quality_RF']=q.groupby(['speaker','file']).agg(label=('label','first'),prob=('group_prob','mean')).groupby('speaker').agg(label=('label','first'),prob=('prob','mean'))
 speakers=sorted(ensembles['V_joint'].index);y=ensembles['V_joint'].loc[speakers].label.to_numpy(int);i1=RNG.integers(0,(y==1).sum(),(5000,(y==1).sum()));i0=RNG.integers(0,(y==0).sum(),(5000,(y==0).sum()));ens=[];draws={}
 for name,g in ensembles.items():
  s=g.loc[speakers].prob.to_numpy();draws[name]=v.auc_draws(y,s,i1,i0);lo,hi=np.quantile(draws[name],[.025,.975]);ens.append(dict(model=name,auc=roc_auc_score(y,s),low=lo,high=hi,bac=balanced_accuracy_score(y,s>.5)))
 save(ens,'ensemble_summary');save([dict(left=l,right=r,mean_difference=next(x['auc'] for x in ens if x['model']==l)-next(x['auc'] for x in ens if x['model']==r),low=np.quantile(draws[l]-draws[r],.025),high=np.quantile(draws[l]-draws[r],.975)) for l,r in [('V_joint','V_cls'),('V_joint','quality_RF'),('V_cls','quality_RF')]],'ensemble_contrasts')
 # Score associations on 141 speaker averages, quality-adjusted analyses complete cases.
 features=json.loads((ROOT/'data/processed/feature_list.json').read_text());f=pd.read_csv(ROOT/'data/processed/features_screened.csv');q=pd.read_csv(ROOT/'data/processed/quality.csv');ling=pd.read_csv(ROOT/'data/processed/features_linguistic.csv')
 scores=norm[norm.split=='test'].groupby('file').percentile.mean().rename('score');z=f.join(scores,on='file').merge(q,on='file').merge(ling[['file','n_words']],on='file');z['log_duration']=np.log(z.duration_s);covs=['log_duration','snr_db','bw99_hz','loudness_dbfs','peak','ch_corr'];sp=z.groupby('speaker')[features+covs+['score','label','n_words']].mean();sp=sp.dropna(subset=covs);C=np.column_stack([np.ones(len(sp)),stats.rankdata(sp[covs],axis=0)]);pinv=np.linalg.pinv(C);ry=stats.rankdata(sp.score);ry-=C@(pinv@ry);assoc=[]
 for c in features+covs:
  rho,pv=stats.spearmanr(sp[c],sp.score);rx=stats.rankdata(sp[c]);rx-=C@(pinv@rx);partial=stats.pearsonr(rx,ry).statistic if c not in covs else np.nan;assoc.append(dict(feature=c,rho=rho,p=pv,partial_rho_quality=partial,n_speakers=len(sp)))
 save(assoc,'score_associations');gscore=norm[norm.split=='test'].groupby(['speaker','file']).agg(score=('percentile','mean'),label=('label','first')).groupby('speaker').agg(score=('score','mean'),label=('label','first'));save([dict(auc=roc_auc_score(gscore.label,gscore.score),n_speakers=len(gscore))],'progression_group_auc')
 # Aggregate actual OOF RF SHAP values from all 25 models, not one explanation subset.
 sv=pd.read_csv(OUT/'rf_shap_oof.csv');cols=[c for c in names if c in sv];absolute=sv[cols].abs().fillna(0);absolute['speaker']=sv.speaker;means=absolute.groupby('speaker')[cols].mean();ix=RNG.integers(0,len(means),(5000,len(means)));boot=means.to_numpy()[ix].mean(1);ranks=np.argsort(np.argsort(-boot,axis=1),axis=1)+1
 shaprows=[]
 for i,c in enumerate(cols):shaprows.append(dict(feature=c,mean_abs=means[c].mean(),low=np.quantile(boot[:,i],.025),high=np.quantile(boot[:,i],.975),top5_fraction=(ranks[:,i]<=5).mean(),selection_fraction=float(sv[c].notna().mean())))
 save(shaprows,'shap_feature_summary')
 timing={'speech_rate','articulation_rate','pause_count','pause_total_s','pause_mean_s','pause_ratio'};voice={'f0_mean','f0_std','f0_range','jitter','shimmer','hnr'};language=set(ling.columns)-{'file'};families={'Timing':timing,'Language':language,'Voice':voice,'Spectral':set(names)-timing-voice-language};familyrows=[];densities={}
 for fam,cs in families.items():
  cs=[c for c in cols if c in cs];idx=[cols.index(c) for c in cs];mass=boot[:,idx].sum(1);share=mass/boot.sum(1);num=sv[cs].notna().sum(1);density=sv[cs].abs().sum(1)/num.replace(0,np.nan);by=density.groupby(sv.speaker).mean().reindex(means.index);densities[fam]=by.to_numpy()[ix].mean(1)
  counts=num.groupby([sv.seed,sv.fold]).first();familyrows.append(dict(family=fam,n_features_min=int(counts.min()),n_features_max=int(counts.max()),share=means[cs].to_numpy().mean(0).sum()/means.to_numpy().mean(0).sum(),share_low=np.quantile(share,.025),share_high=np.quantile(share,.975),per_feature_mean=by.mean(),density_low=np.quantile(densities[fam],.025),density_high=np.quantile(densities[fam],.975)))
 save(familyrows,'shap_family_summary');ratio=densities['Timing']/densities['Spectral'];save([dict(ratio=next(x['per_feature_mean'] for x in familyrows if x['family']=='Timing')/next(x['per_feature_mean'] for x in familyrows if x['family']=='Spectral'),low=np.quantile(ratio,.025),high=np.quantile(ratio,.975))],'shap_density_ratio')
 # Independently verify source domain statistics are still anchored to the same measurements.
 ac=pd.read_csv(ROOT/'data/processed/features_acoustic.csv');identity=np.abs(ac.speech_rate-ac.articulation_rate*(1-ac.pause_ratio));assert identity.max()<1e-10
 domain=f[['file','speaker']].merge(ac,on='file').merge(q[['file','duration_s']],on='file').merge(order[['file','rank']],on='file');domain['log_r']=np.log(domain.speech_rate);domain['log_a']=np.log(domain.articulation_rate);domain['log_nonpause']=np.log1p(-domain.pause_ratio);domain['log_duration']=np.log(domain.duration_s)
 bucket=domain.groupby(['speaker','rank'])[['log_r','log_a','log_nonpause','log_duration']].mean();changes=[]
 for speaker,g in bucket.groupby(level=0):
  if len(g)<2:continue
  x=g.iloc[-1]-g.iloc[0];changes.append(x.to_dict())
 changes=pd.DataFrame(changes);assert len(changes)==31;assert np.max(abs(changes.log_r-changes.log_a-changes.log_nonpause))<1e-12;rate=[];sign=RNG.choice([-1,1],(10000,len(changes)))
 for c in changes:
  x=changes[c].to_numpy();lo,hi=ci(x);pv=(1+(abs((sign*x).mean(1))>=abs(x.mean())).sum())/10001;rate.append(dict(component=c,mean_log_change=x.mean(),low=lo,high=hi,percent=100*np.expm1(x.mean()),p=pv))
 ps=np.array([x['p'] for x in rate]);inds=np.argsort(ps);adj=np.minimum.accumulate((ps[inds]*4/np.arange(1,5))[::-1])[::-1];corrected=np.empty(4);corrected[inds]=np.minimum(adj,1)
 for x,qv in zip(rate,corrected):x['q']=qv
 save(rate,'rate_decomposition')
 selection=pd.read_csv(OUT/'states_selection_V_joint.csv');ks=selection.loc[selection.groupby(['seed','fold']).bic.idxmin()].sort_values(['seed','fold']);ks.to_csv(OUT/'states_kselect.csv',index=False);states=pd.read_csv(OUT/'states_assignments_V_joint.csv');state_test=states[states.split=='test'];counts=state_test.groupby('file').best_k.agg(lambda s:int((s>1).sum()));save([dict(n_recordings=len(counts),no_split_any_seed=int((counts==0).sum()),some_split=int(counts.between(1,4).sum()),split_all_five=int((counts==5).sum()))],'state_opportunities')
 summary_dict=dict(model_summary={exp:{c+'_'+st:float(summary.loc[exp,(c,st)]) for c in ['auc','bac','speaker_auc','speaker_bac'] for st in ['mean','std']} for exp in summary.index},contrast=contrast,temporal=ts,permutation=permutation,ensemble=ens,rate=rate,k_counts={str(int(k)):int(n) for k,n in ks.k.value_counts().items()},attribution_ratio=float(next(x['per_feature_mean'] for x in familyrows if x['family']=='Timing')/next(x['per_feature_mean'] for x in familyrows if x['family']=='Spectral')),n_verified_models=125,n_oof_predictions=len(p),n_same_model_scores=len(allscores),n_shap_recordings=len(sv),max_rate_identity_error=float(identity.max()))
 (OUT/'analysis_summary.json').write_text(json.dumps(summary_dict,indent=2));print(json.dumps(summary_dict,indent=2),flush=True)
if __name__=='__main__':main()
