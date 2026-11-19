"""Train-only preprocessing and identities shared by corrected experiments.

No file I/O, model loading, or fitting occurs on import.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

QUALITY=['snr_db','loudness_dbfs','peak','bw99_hz','duration_s']
METADATA={'file','speaker','label','split','rank','state','row_index','group','dementia type'}
VARIANTS={'V_feat':({'feat'},0.),'V_audio':({'feat','audio'},0.),'V_text':({'feat','text'},0.),
          'V_feat_joint':({'feat'},.5),'V_cls':({'feat','audio','text'},0.),'V_joint':({'feat','audio','text'},.5)}

def load_candidates(root):
    p=Path(root)/'data/processed'
    m=pd.read_csv(p/'manifest.csv');a=pd.read_csv(p/'features_acoustic.csv');l=pd.read_csv(p/'features_linguistic.csv');q=pd.read_csv(p/'quality.csv')
    for name,d in [('manifest',m),('acoustic',a),('linguistic',l),('quality',q)]:
        if d.file.duplicated().any():raise ValueError(name+': duplicate recording IDs')
    if m.label.dtype==object:
        lookup={'dementia':1,'nodementia':0}
        if not set(m.label)<=set(lookup):raise ValueError('Unrecognised group labels')
        m['label']=m.label.map(lookup)
    if not set(m.label)<= {0,1} or m.speaker.isna().any():raise ValueError('Missing/invalid labels or speakers')
    if (m.groupby('speaker').label.nunique()!=1).any():raise ValueError('Inconsistent speaker label')
    cols=[c for c in a.columns if c!='file']+[c for c in l.columns if c!='file']
    if len(set(cols))!=len(cols) or set(cols)&METADATA:raise ValueError('Feature schema includes metadata or duplicate names')
    d=m[['file','speaker','label']].merge(q[['file']+QUALITY],on='file',validate='one_to_one').merge(a,on='file',validate='one_to_one').merge(l,on='file',validate='one_to_one')
    if len(d)!=len(m):raise ValueError('Missing extracted measurements')
    return d,cols

class FoldPreprocessor:
    def __init__(self,names,quality_threshold=.5,redundancy_threshold=.9):
        if set(names)&METADATA:raise ValueError('Outcome/metadata columns cannot be inputs')
        self.names=list(names);self.qt=quality_threshold;self.rt=redundancy_threshold
    def fit(self,X,Q,fit_ids):
        X=np.asarray(X,float);Q=np.asarray(Q,float)
        if len(X)!=len(fit_ids) or len(set(fit_ids))!=len(fit_ids):raise ValueError('Fit IDs must identify each training row')
        X=np.where(np.isfinite(X),X,np.nan);Q=np.where(np.isfinite(Q),Q,np.nan)
        self.fit_ids_=list(fit_ids)
        self.median_=np.nanmedian(X,axis=0);self.qmedian_=np.nanmedian(Q,axis=0)
        self.median_=np.nan_to_num(self.median_);self.qmedian_=np.nan_to_num(self.qmedian_)
        xx=np.where(np.isnan(X),self.median_,X);qq=np.where(np.isnan(Q),self.qmedian_,Q)
        keep=[]
        for j in range(xx.shape[1]):
            if np.ptp(xx[:,j])<1e-12:continue
            correlations=[spearmanr(xx[:,j],qq[:,k]).statistic for k in range(qq.shape[1]) if np.ptp(qq[:,k])>1e-12]
            if any(np.isfinite(r) and abs(r)>self.qt for r in correlations):continue
            keep.append(j)
        if len(keep)>1:
            ranked=rankdata(xx[:,keep],axis=0);c=np.corrcoef(ranked.T);accepted=[]
            for j,col in enumerate(keep):
                if not any(abs(c[j,i])>self.rt for i in accepted):accepted.append(j)
            keep=[keep[i] for i in accepted]
        if not keep:raise ValueError('Screening retained no features; do not silently insert a target or dummy feature')
        self.indices_=keep;self.feature_names_=[self.names[i] for i in keep]
        self.scaler_=StandardScaler().fit(xx[:,keep]);return self
    def transform(self,X,scale=True):
        xx=np.asarray(X,float);xx=np.where(np.isfinite(xx),xx,self.median_);xx=xx[:,self.indices_]
        return self.scaler_.transform(xx) if scale else xx
    def to_dict(self):
        return dict(candidates=self.names,indices=self.indices_,feature_names=self.feature_names_,fit_ids=self.fit_ids_,median=self.median_.tolist(),quality_median=self.qmedian_.tolist(),mean=self.scaler_.mean_.tolist(),scale=self.scaler_.scale_.tolist(),quality_threshold=self.qt,redundancy_threshold=self.rt)

def outer_splits(d,seed):
    return StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(d,d.label,d.speaker)

def inner_split(d,outer_train,seed):
    otr=np.asarray(outer_train);z=d.iloc[otr]
    a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(z,z.label,z.speaker))
    return otr[a],otr[b]

def rank_pairs(d,indices,order):
    z=d.iloc[np.asarray(indices)][['file','speaker','label']].copy();z['local']=np.arange(len(z))
    o=order.dropna(subset=['rank']).rename(columns={'recording':'file'})
    if 'order_basis' in o:o=o[o.order_basis.fillna('').str.strip().ne('')]
    if o.file.duplicated().any():raise ValueError('Duplicate chronology entries')
    z=z.merge(o[['file','speaker','rank']],on=['file','speaker'],validate='one_to_one');z=z[z.label==1]
    out=[];ties=0
    for _,g in z.groupby('speaker'):
        records=g.sort_values('rank').to_dict('records')
        for a in range(len(records)):
            for b in range(a+1,len(records)):
                x,y=records[a],records[b]
                if x['rank']==y['rank']:ties+=1
                else:out.append((x['local'],y['local']))
    return out,ties

def checkpoint_name(experiment,seed,fold):
    if experiment not in VARIANTS:raise ValueError(experiment)
    return f'{experiment}_s{seed}_f{fold}.pt'

def load_embeddings(root,d,variants):
    needed=set().union(*(VARIANTS[v][0] for v in variants))
    p=Path(root)/'data/processed';zeros=np.zeros((len(d),768),dtype=np.float32)
    if not (needed&{'audio','text'}):return zeros.copy(),zeros.copy()
    required=[p/'embeddings_index.csv']+[p/f'embeddings_{n}.npy' for n,k in [('wavlm','audio'),('bert','text')] if k in needed]
    absent=[str(f) for f in required if not f.exists()]
    if absent:raise FileNotFoundError('Real embeddings required; no substitutes permitted: '+', '.join(absent))
    ix=pd.read_csv(p/'embeddings_index.csv')
    if ix.file.duplicated().any() or ix.row_index.duplicated().any() or set(ix.file)!=set(d.file):raise ValueError('Embedding index not one-to-one with cohort')
    rows=d[['file']].merge(ix,on='file',validate='one_to_one').row_index.to_numpy()
    if not np.all(rows==rows.astype(int)):raise ValueError('Noninteger embedding index')
    rows=rows.astype(int);arrays=[]
    for n,k in [('wavlm','audio'),('bert','text')]:
        if k not in needed:arrays.append(zeros.copy());continue
        a=np.load(p/f'embeddings_{n}.npy',allow_pickle=False)
        if a.shape!=(len(d),768) or not np.isfinite(a).all() or set(rows)!=set(range(len(a))):raise ValueError(n+': incomplete/nonfinite/misaligned embeddings')
        arrays.append(a[rows].astype(np.float32))
    return tuple(arrays)

def input_hashes(root):
    p=Path(root)/'data/processed';out={}
    for f in sorted(p.glob('*')):
        if f.is_file() and f.suffix in {'.csv','.npy','.json'}:out[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    return out
