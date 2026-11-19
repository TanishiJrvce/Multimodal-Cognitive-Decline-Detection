"""States from actual training scores of the SAME model as each held-out score."""
from pathlib import Path
import sys,argparse,pickle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def validate_score_source(g):
    required={'model_id','scored_by_model','split','file','prog_score'}
    if not required<=set(g):raise ValueError('Actual same-model all-recording scores required; cross-fold OOF proxies are not training scores')
    if g.model_id.nunique()!=1 or not (g.model_id==g.scored_by_model).all():raise ValueError('Mixed model score scales')
    if g.file.duplicated().any() or set(g['split'])!={'train','test'}:raise ValueError('Training and held-out scores must be explicitly identified')
    if not np.isfinite(g.prog_score).all():raise ValueError('Nonfinite progression score')

def fit_states(g,seed):
    validate_score_source(g);train=g[g.split=='train'].prog_score.to_numpy()[:,None];candidates=[];models={}
    for k in range(1,7):
        m=GaussianMixture(k,random_state=seed,n_init=5).fit(train);models[k]=m;labels=m.predict(train)
        sil=silhouette_score(train,labels) if 1<len(set(labels))<len(train) else np.nan
        candidates.append(dict(k=k,bic=m.bic(train),silhouette=sil))
    best=min(candidates,key=lambda r:r['bic'])['k'];m=models[best];order={int(j):int(i) for i,j in enumerate(np.argsort(m.means_[:,0]))}
    assigned=g.copy();assigned['state']=[order[int(x)] for x in m.predict(g.prog_score.to_numpy()[:,None])];assigned['best_k']=best
    return pd.DataFrame(candidates),assigned,m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--experiment',default='V_joint',choices=['V_joint','V_feat_joint']);a=ap.parse_args();root=a.root
    files=sorted((root/'data/results/v2/scores').glob(a.experiment+'_s*.csv'))
    if len(files)!=25:raise FileNotFoundError('25 completed joint same-model score tables required; refuse legacy proxy scores')
    stats=[];assigned=[]
    for f in files:
        g=pd.read_csv(f);seed=int(g.seed.iloc[0]);fold=int(g.fold.iloc[0]);r,z,m=fit_states(g,seed);r['seed']=seed;r['fold']=fold;stats.append(r);assigned.append(z)
        with (root/'data/models/v2'/f'{g.model_id.iloc[0]}_gmm.pkl').open('wb') as handle:pickle.dump(m,handle)
    pd.concat(stats).to_csv(root/f'data/results/v2/states_selection_{a.experiment}.csv',index=False);pd.concat(assigned).to_csv(root/f'data/results/v2/states_assignments_{a.experiment}.csv',index=False)
if __name__=='__main__':main()
