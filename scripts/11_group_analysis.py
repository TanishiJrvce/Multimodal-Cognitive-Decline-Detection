"""Post-hoc group, longitudinal, and contrast analyses. See README for estimands."""
from pathlib import Path
import argparse,json,hashlib,itertools,warnings,os
import numpy as np,pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,brier_score_loss,adjusted_rand_score
warnings.filterwarnings('ignore',category=RuntimeWarning)
BASE=Path(__file__).resolve().parent.parent  # repo root
REPO=Path(os.environ.get('MCD_V0_REPO', str(BASE)))
TABLES=BASE/'data/results';TABLES.mkdir(exist_ok=True,parents=True)
RNG=np.random.default_rng(20261119)
B=5000;P=10000
FEATURES=json.loads((REPO/'data/processed/feature_list.json').read_text())
TIMING=['speech_rate','pause_mean_s','pause_ratio']
LING=['mattr50','filler_rate','immediate_rep_rate','bigram_rep','mean_sent_len']
VOICE=['f0_mean','f0_std','f0_range','jitter','shimmer','hnr']
SPEC=[x for x in FEATURES if x not in TIMING+LING+VOICE]
Q=['log_duration','snr_db','bw99_hz','loudness_dbfs','peak','ch_corr']
def read(p):return pd.read_csv(REPO/p)
def save(df,name):df.to_csv(TABLES/(name+'.csv'),index=False)
def bh(p):
 p=np.asarray(p,float);o=np.argsort(p);z=np.minimum.accumulate((p[o]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out=np.empty(len(p));out[o]=np.minimum(z,1);return out
def ols_hc3(X,y,index=1):
 X=np.asarray(X,float);y=np.asarray(y,float);inv=np.linalg.pinv(X.T@X);beta=inv@X.T@y;r=y-X@beta;hat=np.sum((X@inv)*X,axis=1)
 meat=X.T@(((r/np.maximum(1-hat,1e-9))**2)[:,None]*X);cov=inv@meat@inv;se=np.sqrt(np.maximum(np.diag(cov),0));v=beta[index];s=se[index];return v,s,2*stats.norm.sf(abs(v/s)),v-1.96*s,v+1.96*s
def boot_diff(x1,x0):
 i=RNG.integers(0,len(x1),(B,len(x1)));j=RNG.integers(0,len(x0),(B,len(x0)));v=x1[i].mean(1)-x0[j].mean(1);return np.quantile(v,[.025,.975])
def boot_mean(x):
 x=np.asarray(x);return np.quantile(x[RNG.integers(0,len(x),(B,len(x)))].mean(1),[.025,.975])
def load():
 m=read('data/processed/manifest.csv');f=read('data/processed/features_screened.csv');q=read('data/processed/quality.csv');ling=read('data/processed/features_linguistic.csv');ac=read('data/processed/features_acoustic.csv')
 df=f.merge(q,on='file').merge(m[['file','gender','dementia type','suffix','birth','first symptoms']],on='file').merge(ling[['file','n_words','words_per_min']],on='file').merge(ac[['file','articulation_rate','pause_count','pause_total_s']],on='file')
 df['log_duration']=np.log(df.duration_s);df['female']=(df.gender=='female').astype(float);df.loc[df.gender.isna(),'female']=np.nan
 return df,m
def group_effects():
 df,m=load();sp=df.groupby('speaker')[FEATURES+Q+['label','female','n_words']].mean();rows=[]
 y=sp.label.to_numpy(int);perm=np.array([RNG.permutation(y) for _ in range(P)],dtype=float)
 for c in FEATURES:
  x=sp[c].to_numpy();x1=x[y==1];x0=x[y==0];delta=x1.mean()-x0.mean();sd=np.sqrt(((len(x1)-1)*x1.var(ddof=1)+(len(x0)-1)*x0.var(ddof=1))/(len(x)-2));g=(1-3/(4*len(x)-9))*delta/sd;ci=boot_diff(x1,x0)
  null=perm@x/(y==1).sum()-(1-perm)@x/(y==0).sum();p=(1+(abs(null)>=abs(delta)).sum())/(P+1)
  covs=['female']+Q;z=sp[['label',c]+covs].dropna();X=np.column_stack([np.ones(len(z)),z.label,StandardScaler().fit_transform(z[covs])]);out=ols_hc3(X,z[c])
  rows.append(dict(feature=c,n1=len(x1),n0=len(x0),mean1=x1.mean(),mean0=x0.mean(),difference=delta,ci_low=ci[0],ci_high=ci[1],hedges_g=g,permutation_p=p,adjusted_difference=out[0],adjusted_se=out[1],adjusted_p=out[2],adjusted_low=out[3],adjusted_high=out[4],adjusted_n=len(z)))
 r=pd.DataFrame(rows);r['q_bh_37']=bh(r.permutation_p);r['adjusted_q_bh_37']=bh(r.adjusted_p);save(r,'speaker_group_effects')
 print('GROUP strongest',r.sort_values('permutation_p')[['feature','hedges_g','permutation_p','q_bh_37','adjusted_q_bh_37']].head(10).round(4).to_dict('records'),flush=True)
 # Prespecified clinical contrasts: both groups, AD subset, one recording per speaker.
 spmeta=m.groupby('speaker')['dementia type'].first();out=[]
 for mode in ['all','Alzheimer_only']:
  subset=sp if mode=='all' else sp[(sp.label==0)|sp.index.isin(spmeta[spmeta=='Alzheimer'].index)]
  for c in TIMING+LING+VOICE:
   xx1=subset.loc[subset.label==1,c].values;xx0=subset.loc[subset.label==0,c].values;lo,hi=boot_diff(xx1,xx0)
   out.append(dict(contrast=mode,feature=c,n1=len(xx1),n0=len(xx0),difference=xx1.mean()-xx0.mean(),low=lo,high=hi))
 save(pd.DataFrame(out),'subtype_sensitivity')
 # Boundary-aware descriptive Spearman associations: speaker averages.
 cor=[]
 for c in FEATURES:
  for qc in Q+['n_words']:
   rr,pp=stats.spearmanr(sp[c],sp[qc],nan_policy='omit');cor.append(dict(feature=c,covariate=qc,rho=rr,p=pp))
 save(pd.DataFrame(cor),'feature_covariate_correlations')
def longitudinal():
 df,m=load();o=read('data/processed/recording_order.csv').rename(columns={'recording':'file','split':'original_split'});d=o[['speaker','file','rank']].merge(df.drop(columns=['speaker']),on='file')
 cols=FEATURES+Q+['articulation_rate','pause_count','pause_total_s'];bucket=d.groupby(['speaker','rank'])[cols].mean().reset_index();bucket=bucket[bucket.speaker.isin(bucket.groupby('speaker')['rank'].nunique().loc[lambda x:x>1].index)]
 changes=[]
 for s,g in bucket.groupby('speaker'):
  g=g.sort_values('rank');first,last=g.iloc[0],g.iloc[-1];r={'speaker':s,'gap_rank':last['rank']-first['rank'],'n_buckets':len(g)};r.update({c:last[c]-first[c] for c in cols});changes.append(r)
 ch=pd.DataFrame(changes);signs=RNG.choice([-1,1],(P,len(ch)));rows=[]
 for c in FEATURES+['articulation_rate']:
  x=ch[c].to_numpy();lo,hi=boot_mean(x);null=(signs*x).mean(1);p=(1+(abs(null)>=abs(x.mean())).sum())/(P+1)
  med=np.quantile(np.median(x[RNG.integers(0,len(x),(B,len(x)))],axis=1),[.025,.975])
  rows.append(dict(feature=c,n_speakers=len(x),mean_delta=x.mean(),mean_low=lo,mean_high=hi,median_delta=np.median(x),median_low=med[0],median_high=med[1],increase=int((x>0).sum()),decrease=int((x<0).sum()),ties=int((x==0).sum()),signflip_p=p,sign_p=stats.binomtest(int((x>0).sum()),int((x!=0).sum()),.5).pvalue))
 r=pd.DataFrame(rows);r['q_bh_38']=bh(r.signflip_p);save(r,'longitudinal_endpoint_effects')
 # Fixed-effects association by within-speaker demeaning; equal total weight per speaker.
 within=bucket.copy();size=within.groupby('speaker')['rank'].transform('size');weight=1/np.sqrt(size.to_numpy());keys=['rank']+FEATURES+Q
 for c in keys:within[c]=within[c]-within.groupby('speaker')[c].transform('mean')
 rows=[];grp=within.speaker.to_numpy();unique=np.unique(grp)
 for c in FEATURES:
  for adjusted in [False,True]:
   cs=['rank']+(Q[:4] if adjusted else []);X=within[cs].values*weight[:,None];yy=within[c].values*weight;inv=np.linalg.pinv(X.T@X);beta=inv@X.T@yy;res=yy-X@beta;meat=np.zeros((len(cs),len(cs)))
   for s in unique:
    z=X[grp==s].T@res[grp==s];meat+=np.outer(z,z)
   cov=inv@meat@inv*len(unique)/(len(unique)-1)*(len(yy)-1)/(len(yy)-len(cs));se=np.sqrt(np.maximum(cov[0,0],0));crit=stats.t.ppf(.975,len(unique)-1)
   rows.append(dict(feature=c,adjusted=adjusted,beta_rank=beta[0],se=se,low=beta[0]-crit*se,high=beta[0]+crit*se,p=2*stats.t.sf(abs(beta[0]/se),len(unique)-1),n_speakers=len(unique),n_buckets=len(within)))
 rr=pd.DataFrame(rows)
 for flag in [False,True]:rr.loc[rr.adjusted==flag,'q_bh_37']=bh(rr.loc[rr.adjusted==flag,'p'])
 save(rr,'within_speaker_rank_associations')
 # Domain decomposition of measured slowing: rate versus articulation versus pause fraction.
 out=ch[['speaker','gap_rank','speech_rate','articulation_rate','pause_ratio','pause_mean_s','pause_count','pause_total_s']].copy();out['speaker_id']=['S%03d'%i for i in range(len(out))];save(out.drop(columns='speaker'),'anonymized_endpoint_timing')
 print('LONGITUDINAL',r[r.feature.isin(TIMING+LING+['articulation_rate'])].round(5).to_dict('records'),flush=True)
def original_contrasts():
 j=read('data/results/joint_results.csv');p=j.pivot(index=['seed','fold'],columns='experiment',values='auc');rows=[]
 for left,right in [('V_joint','V_cls'),('V_cls','V_audio'),('V_cls','V_text'),('V_joint','V_audio')]:
  d=p[left]-p[right];seed=d.groupby(level=0).mean();corrected_se=d.std(ddof=1)*np.sqrt(1/25+1/4);crit=stats.t.ppf(.975,24)
  rows.append(dict(left=left,right=right,mean_difference=d.mean(),median_difference=d.median(),wins=int((d>0).sum()),ties=int((d==0).sum()),n_runs=len(d),seed_min=seed.min(),seed_max=seed.max(),positive_seed_means=int((seed>0).sum()),corrected_resampling_low=d.mean()-crit*corrected_se,corrected_resampling_high=d.mean()+crit*corrected_se))
 save(pd.DataFrame(rows),'original_model_contrasts')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['group','longitudinal','contrasts','all'],default='all');args=ap.parse_args()
 if args.stage in ['group','all']:group_effects()
 if args.stage in ['longitudinal','all']:longitudinal()
 if args.stage in ['contrasts','all']:original_contrasts()
