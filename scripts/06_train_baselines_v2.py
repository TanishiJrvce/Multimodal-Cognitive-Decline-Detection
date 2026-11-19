"""Matched outer-fold baselines with real OOF predictions and RF explanations."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from mcd.protocol import load_candidates,FoldPreprocessor,outer_splits,QUALITY
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/results/v2'
def main():
    import shap
    OUT.mkdir(exist_ok=True);d,names=load_candidates(ROOT);X=d[names].to_numpy(float);Q=d[QUALITY].to_numpy(float);y=d.label.to_numpy();pred=[];runs=[];svs=[];screening=[]
    ac=set(pd.read_csv(ROOT/'data/processed/features_acoustic.csv').columns)-{'file'}
    for seed in range(5):
        for fold,(tr,te) in enumerate(outer_splits(d,seed)):
            prep=FoldPreprocessor(names).fit(X[tr],Q[tr],d.iloc[tr].file);trX=prep.transform(X[tr]);teX=prep.transform(X[te]);cols=prep.feature_names_
            qtr=np.where(np.isfinite(Q[tr]),Q[tr],prep.qmedian_);qte=np.where(np.isfinite(Q[te]),Q[te],prep.qmedian_)
            ia=[i for i,c in enumerate(cols) if c in ac];il=[i for i,c in enumerate(cols) if c not in ac];ir=[cols.index(c) for c in ['pause_ratio','speech_rate'] if c in cols]
            grid={'B1_rule':(LogisticRegression(C=.1,max_iter=1000,random_state=seed),trX[:,ir],teX[:,ir]),'B2_acoustic':(LogisticRegression(C=.1,max_iter=1000,random_state=seed),trX[:,ia],teX[:,ia]),'B3_linguistic':(LogisticRegression(C=.1,max_iter=1000,random_state=seed),trX[:,il],teX[:,il]),'B4_RF':(RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1),prep.transform(X[tr],False),prep.transform(X[te],False)),'B5_SVM':(SVC(C=1.,kernel='rbf',gamma='scale',probability=True,random_state=seed),trX,teX),'B6_GBM':(HistGradientBoostingClassifier(max_iter=200,random_state=seed),prep.transform(X[tr],False),prep.transform(X[te],False)),'B7_quality_RF':(RandomForestClassifier(n_estimators=200,random_state=seed,n_jobs=1),qtr,qte)}
            for name,(m,tx,vx) in grid.items():
                if tx.shape[1]==0:raise ValueError('Empty baseline feature family')
                m.fit(tx,y[tr]);prob=m.predict_proba(vx)[:,1];predicted=m.predict(vx)
                runs.append(dict(model=name,seed=seed,fold=fold,auc=roc_auc_score(y[te],prob),bac=balanced_accuracy_score(y[te],predicted),n_features=tx.shape[1]))
                for i,pp in zip(te,prob):pred.append(dict(model=name,seed=seed,fold=fold,file=d.file.iloc[i],speaker=d.speaker.iloc[i],label=int(y[i]),group_prob=pp))
                if name=='B4_RF':
                    values=shap.TreeExplainer(m).shap_values(vx)
                    values=values[:,:,1] if isinstance(values,np.ndarray) and values.ndim==3 else values[1] if isinstance(values,list) else values
                    z=pd.DataFrame(values,columns=cols);z['file']=d.file.iloc[te].to_numpy();z['speaker']=d.speaker.iloc[te].to_numpy();z['seed']=seed;z['fold']=fold;svs.append(z)
            screening.append(dict(seed=seed,fold=fold,features=cols))
            print(f'Baselines and SHAP seed={seed} fold={fold}',flush=True)
    pd.DataFrame(runs).to_csv(OUT/'baseline_runs.csv',index=False);pd.DataFrame(pred).to_csv(OUT/'baseline_oof.csv',index=False);pd.concat(svs,ignore_index=True).to_csv(OUT/'rf_shap_oof.csv',index=False);(OUT/'rf_features.json').write_text(json.dumps(screening,indent=2));pd.DataFrame(runs).groupby('model')[['auc','bac']].agg(['mean','std']).to_csv(OUT/'baseline_summary.csv')
if __name__=='__main__':main()
