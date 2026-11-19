"""Independent checks and speaker-aware analysis of the committed V1 predictions.

Never loads embeddings, reruns neural training, or accepts cross-fold proxy
training scores as same-model scores. Writes separate independent_audit outputs.
"""
from pathlib import Path
import sys,json,itertools,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from mcd.protocol import load_candidates,outer_splits,inner_split,FoldPreprocessor,QUALITY,rank_pairs
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'data/results';OUT=RES/'independent_audit';OUT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20261120)
def save(d,n):pd.DataFrame(d).to_csv(OUT/(n+'.csv'),index=False)
def ci_mean(x):
    x=np.asarray(x);return np.quantile(x[RNG.integers(0,len(x),(5000,len(x)))].mean(1),[.025,.975])
def auc_draws(y,s,i1,i0):
    pos=np.asarray(s)[np.flatnonzero(y==1)][i1];neg=np.asarray(s)[np.flatnonzero(y==0)][i0];out=[]
    for k in range(0,len(i1),200):
        dif=pos[k:k+200,:,None]-neg[k:k+200,None,:];out.extend(((dif>0)+.5*(dif==0)).mean((1,2)))
    return np.asarray(out)
def temporal_records(pred,order):
    rows=[]
    for (exp,seed,fold,speaker),g in pred.merge(order[['file','speaker','rank']],on=['file','speaker']).groupby(['experiment','seed','fold','speaker']):
        b=g.groupby('rank').prog_score.mean().sort_index().to_numpy()
        if len(b)<2:continue
        pairs=[float(b[j]>b[i])+.5*float(b[j]==b[i]) for i in range(len(b)) for j in range(i+1,len(b))]
        rows.append(dict(experiment=exp,seed=seed,fold=fold,speaker=speaker,concordance=np.mean(pairs),endpoint_hit=float(b[-1]>b[0])+.5*float(b[-1]==b[0]),n_pairs=len(pairs)))
    return pd.DataFrame(rows)
def main():
    d,names=load_candidates(ROOT);p=pd.read_csv(RES/'joint_predictions_v1.csv');j=pd.read_csv(RES/'joint_results_v1.csv');a=pd.read_csv(RES/'states_assignments_v1.csv');o=pd.read_csv(ROOT/'data/processed/recording_order.csv').rename(columns={'recording':'file'}).dropna(subset=['rank']);issues=[]
    assert len(p)==6875 and not p.duplicated(['experiment','seed','file']).any()
    assert p.groupby(['experiment','seed']).file.nunique().eq(275).all()
    checks=[];membership=[];screen_changes=[];qpred=[];qmetrics=[]
    X=d[names].to_numpy(float);Q=d[QUALITY].to_numpy(float)
    for seed in range(5):
        for fold,(tr,te) in enumerate(outer_splits(d,seed)):
            sub=p[(p.seed==seed)&(p.fold==fold)]
            assert all(set(g.file)==set(d.iloc[te].file) for _,g in sub.groupby('experiment'))
            assert not set(d.iloc[tr].speaker)&set(d.iloc[te].speaker)
            it,iv=inner_split(d,tr,seed);outer=FoldPreprocessor(names).fit(X[tr],Q[tr],d.iloc[tr].file);inner=FoldPreprocessor(names).fit(X[it],Q[it],d.iloc[it].file)
            saved=json.loads((RES/'joint_fold_features_v1'/f'V_joint_s{seed}_f{fold}.json').read_text());assert saved==outer.feature_names_
            screen_changes.append(dict(seed=seed,fold=fold,n_outer_features=len(saved),n_inner_features=len(inner.feature_names_),symmetric_difference=len(set(saved)^set(inner.feature_names_))))
            # Independently reproduce matched quality RF from the actual five input columns.
            rf=RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1).fit(Q[tr],d.label.to_numpy()[tr]);prob=rf.predict_proba(Q[te])[:,1]
            for ii,v in zip(te,prob):qpred.append(dict(file=d.file.iloc[ii],speaker=d.speaker.iloc[ii],label=int(d.label.iloc[ii]),seed=seed,fold=fold,group_prob=v))
            qmetrics.append(dict(seed=seed,fold=fold,auc=roc_auc_score(d.label.to_numpy()[te],prob),bac=balanced_accuracy_score(d.label.to_numpy()[te],prob>.5)))
            for phase,idx in [('outer_train',tr),('outer_test',te),('inner_train',it),('inner_validation',iv)]:
                for f in d.iloc[idx].file:membership.append(dict(seed=seed,fold=fold,phase=phase,file=f))
    save(qpred,'quality_oof_predictions');save(qmetrics,'quality_fold_metrics');save(screen_changes,'inner_screening_comparison');save(membership,'reconstructed_membership')
    metrics=[]
    for (exp,seed,fold),g in p.groupby(['experiment','seed','fold']):
        auc=roc_auc_score(g.label,g.group_prob);bac=balanced_accuracy_score(g.label,g.group_prob>.5)
        old=j[(j.experiment==exp)&(j.seed==seed)&(j.fold==fold)].iloc[0]
        assert abs(auc-old.auc)<1e-10 and abs(bac-old.bac)<1e-10
        sp=g.groupby('speaker').agg(label=('label','first'),prob=('group_prob','mean'))
        metrics.append(dict(experiment=exp,seed=seed,fold=fold,auc=auc,bac=bac,speaker_auc=roc_auc_score(sp.label,sp.prob),speaker_bac=balanced_accuracy_score(sp.label,sp.prob>.5)))
    save(metrics,'recomputed_fold_metrics');pd.DataFrame(metrics).groupby('experiment')[['auc','bac','speaker_auc','speaker_bac']].agg(['mean','std']).to_csv(OUT/'model_summary.csv')
    piv=j.pivot(index=['seed','fold'],columns='experiment',values='auc');contrasts=[]
    for left,right in [('V_joint','V_cls'),('V_cls','V_audio'),('V_cls','V_text'),('V_joint','V_feat')]:
        diff=piv[left]-piv[right];sem=diff.std()*np.sqrt(1/25+1/4);crit=stats.t.ppf(.975,24)
        contrasts.append(dict(left=left,right=right,mean_difference=diff.mean(),wins=int((diff>0).sum()),ties=int((diff==0).sum()),n=25,seed_min=diff.groupby(level=0).mean().min(),seed_max=diff.groupby(level=0).mean().max(),heuristic_low=diff.mean()-crit*sem,heuristic_high=diff.mean()+crit*sem))
    save(contrasts,'paired_model_contrasts')
    # Conditional bootstrap of speaker-level five-seed OOF group-probability ensembles.
    ensembles={}
    for exp,g in p.groupby('experiment'):
        ensembles[exp]=g.groupby(['speaker','file']).agg(label=('label','first'),prob=('group_prob','mean')).groupby('speaker').agg(label=('label','first'),prob=('prob','mean'))
    qp=pd.DataFrame(qpred);ensembles['quality_RF']=qp.groupby(['speaker','file']).agg(label=('label','first'),prob=('group_prob','mean')).groupby('speaker').agg(label=('label','first'),prob=('prob','mean'))
    speakers=sorted(ensembles['V_joint'].index);y=ensembles['V_joint'].loc[speakers].label.to_numpy(int);i1=RNG.integers(0,(y==1).sum(),(5000,(y==1).sum()));i0=RNG.integers(0,(y==0).sum(),(5000,(y==0).sum()));draws={};ensrows=[]
    for name,z in ensembles.items():
        s=z.loc[speakers].prob.to_numpy();b=auc_draws(y,s,i1,i0);draws[name]=b;lo,hi=np.quantile(b,[.025,.975]);ensrows.append(dict(model=name,n_speakers=len(s),auc=roc_auc_score(y,s),low=lo,high=hi,bac=balanced_accuracy_score(y,s>.5),brier=brier_score_loss(y,s)))
    save(ensrows,'speaker_ensemble_metrics');save([dict(left=l,right=r,auc_difference=next(x['auc'] for x in ensrows if x['model']==l)-next(x['auc'] for x in ensrows if x['model']==r),low=np.quantile(draws[l]-draws[r],.025),high=np.quantile(draws[l]-draws[r],.975)) for l,r in [('V_joint','V_cls'),('V_joint','quality_RF'),('V_cls','quality_RF')]],'paired_ensemble_contrasts')
    # Same-model within-speaker temporal ordering uses raw OOF scores: no pooled cross-model scale.
    per=temporal_records(p,o);save(per,'temporal_person_records');ts=[]
    for exp,g in per.groupby('experiment'):
        z=g.groupby('speaker')[['concordance','endpoint_hit']].mean();lo,hi=ci_mean(z.concordance);el,eh=ci_mean(z.endpoint_hit)
        ts.append(dict(experiment=exp,n_speakers=len(z),model_person_evaluations=len(g),concordance=z.concordance.mean(),low=lo,high=hi,endpoint=z.endpoint_hit.mean(),endpoint_low=el,endpoint_high=eh))
    save(ts,'temporal_summary')
    # Null permutations keep a speaker's epoch permutation identical across all five models.
    joint=p[p.experiment=='V_joint'].merge(o[['file','speaker','rank']],on=['file','speaker']);bucket=joint.groupby(['speaker','seed','rank']).prog_score.mean().reset_index();null=np.zeros(10000);observed=[]
    for speaker,g in bucket.groupby('speaker'):
        mat=g.pivot(index='seed',columns='rank',values='prog_score').sort_index(axis=1).to_numpy()
        if mat.shape[1]<2:continue
        def conc(m):return np.mean([np.mean((m[:,j]>m[:,i])+.5*(m[:,j]==m[:,i])) for i in range(m.shape[1]) for j in range(i+1,m.shape[1])])
        observed.append(conc(mat));values=np.array([conc(mat[:,perm]) for perm in itertools.permutations(range(mat.shape[1]))]);null+=values[RNG.integers(0,len(values),len(null))]
    null/=len(observed);obs=np.mean(observed);save([dict(n_speakers=len(observed),concordance=obs,one_sided_p=(1+(null>=obs-1e-12).sum())/10001,two_sided_p=(1+(abs(null-.5)>=abs(obs-.5)-1e-12).sum())/10001,null_q95=np.quantile(null,.95))],'temporal_permutation')
    # Verify whether alleged training scores originate in a different model's OOF prediction.
    train=a[a.split=='train'].merge(p[p.experiment=='V_joint'][['seed','file','fold','prog_score']],on=['seed','file'],suffixes=('_target','_source'))
    matched=np.isclose(train.prog_score_target,train.prog_score_source,atol=1e-12,rtol=0)&(train.fold_target!=train.fold_source)
    save([dict(n_rows=len(train),cross_model_oof_copies=int(matched.sum()),fraction=float(matched.mean()),status='NOT same-model training scores')],'state_source_integrity')
    assert matched.all()
    # Reconstruct actual pair co-occurrences in the V1 final random minibatches (not gradient logs).
    pairing=[]
    for seed in range(5):
        for fold,(tr,te) in enumerate(outer_splits(d,seed)):
            pairs,ties=rank_pairs(d,tr,pd.read_csv(ROOT/'data/processed/recording_order.csv'));epoch=int(j[(j.experiment=='V_joint')&(j.seed==seed)&(j.fold==fold)].best_epoch.iloc[0]);hits=0;covered=set()
            for e in range(epoch):
                orderidx=np.random.RandomState(e+seed*1000).permutation(len(tr));batch=np.empty(len(tr),int);batch[orderidx]=np.arange(len(tr))//32
                for k,(i,ii) in enumerate(pairs):
                    if batch[i]==batch[ii]:hits+=1;covered.add(k)
            pairing.append(dict(seed=seed,fold=fold,selected_epochs=epoch,n_pairs=len(pairs),ties_excluded=ties,reconstructed_pair_exposures=hits,unique_pairs_seen=len(covered),actual_gradient_norm_logged=False))
    save(pairing,'reconstructed_ranking_usage')
    # Model key collision is present in the source, independently of the numerical outputs.
    tree=ast.parse((ROOT/'scripts/07_train_joint_v1.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='mod_names_key');scope={'experiments':{'V_feat':({'feat'},0),'V_audio':({'feat','audio'},0),'V_text':({'feat','text'},0),'V_cls':({'feat','audio','text'},0),'V_joint':({'feat','audio','text'},.5)}};exec(compile(ast.Module(body=[fn],type_ignores=[]),'V1-key','exec'),scope);collision=scope['mod_names_key']({'feat','audio','text'})
    issues=[dict(issue='Inner feature screening includes inner validation',evidence=f"{sum(z['symmetric_difference']>0 for z in screen_changes)}/25 inner and outer feature lists differ",remedy='V2 fits independent inner and outer preprocessing'),dict(issue='V_cls and V_joint checkpoint collision',evidence='Joint modality key resolves to '+collision,remedy='V2 takes explicit experiment identity'),dict(issue='GMM cross-model score proxy',evidence=f'{matched.sum()}/{len(train)} alleged training rows are other-model OOF copies',remedy='V2 records all scores with same-model identity; legacy state claims withdrawn'),dict(issue='Ranking evidence overstated',evidence='V1 logs expected updates and booleans, not observed gradients',remedy='V2 explicit pair batches log exposures and observed gradient updates'),dict(issue='ASR provenance unresolved',evidence='275 transcript hashes but generating model unspecified',remedy='Fixed asset status; no invented model identity'),dict(issue='Neural rerun assets absent from public branch',evidence='No embeddings arrays/index or checkpoints tracked',remedy='V2 feature-only controls executed; complete multimodal rerun awaits real arrays')]
    save(issues,'remaining_issues_and_remedies')
    summary=dict(source_commit='98ebd6a',verified_predictions=len(p),verified_neural_folds=len(metrics),quality_fold_auc_mean=pd.DataFrame(qmetrics).auc.mean(),quality_fold_auc_sd=pd.DataFrame(qmetrics).auc.std(),temporal=ts,conditional_ensemble=ensrows,contrasts=contrasts,proxy_training_rows=int(matched.sum()),inner_feature_mismatches=sum(z['symmetric_difference']>0 for z in screen_changes),actual_quality_columns=QUALITY)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
