"""Fully train-local epoch selection, unique checkpoints, explicit ranking batches.

V1 outputs are preserved. V2 scores contain actual same-model predictions on
ALL recordings, with explicit model identity. No fitting occurs on import.
"""
import argparse,json,sys,random,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from mcd.models import FusionNet,apply_modality_mask
from mcd.protocol import (load_candidates,FoldPreprocessor,outer_splits,inner_split,rank_pairs,checkpoint_name,load_embeddings,input_hashes,VARIANTS,QUALITY)

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def tensors(d,X,W,B,indices,prep,mods,device):
    i=np.asarray(indices);xf,xw,xb=apply_modality_mask(prep.transform(X[i]),W[i],B[i],mods)
    return [torch.as_tensor(a,dtype=torch.float32,device=device) for a in [xf,xw,xb,d.label.to_numpy()[i]]]

def train_epoch(model,opt,data,pairs,lam,rng,mod_drop_p=.2):
    xf,xw,xb,y=data;model.train();order=rng.permutation(len(y));exposures=0;gradient_steps=0;losses=[]
    for start in range(0,len(order),32):
        bi=torch.as_tensor(order[start:start+32],dtype=torch.long,device=y.device)
        drop_w=rng.random()<mod_drop_p;drop_b=rng.random()<mod_drop_p
        cw=torch.zeros_like(xw[bi]) if drop_w else xw[bi];cb=torch.zeros_like(xb[bi]) if drop_b else xb[bi]
        logit,_=model(xf[bi],cw,cb);loss=F.binary_cross_entropy_with_logits(logit,y[bi])
        if lam and pairs:
            take=rng.choice(len(pairs),size=min(16,len(pairs)),replace=False)
            pair=np.asarray(pairs,dtype=int)[take];pi=torch.as_tensor(pair.reshape(-1),dtype=torch.long,device=y.device)
            rw=torch.zeros_like(xw[pi]) if drop_w else xw[pi];rb=torch.zeros_like(xb[pi]) if drop_b else xb[pi]
            _,ps=model(xf[pi],rw,rb);ps=ps.reshape(-1,2)
            loss=loss+lam*F.softplus(-(ps[:,1]-ps[:,0])).mean();exposures+=len(pair)
        opt.zero_grad();loss.backward()
        if model.prog_head.weight.grad is not None and float(model.prog_head.weight.grad.norm())>0:gradient_steps+=1
        opt.step();losses.append(float(loss.detach()))
    return dict(pair_exposures=exposures,ranking_gradient_steps=gradient_steps,loss=float(np.mean(losses)))

@torch.no_grad()
def score(model,data):
    model.eval();logit,ps=model(*data[:3]);return torch.sigmoid(logit).cpu().numpy(),ps.cpu().numpy(),float(F.binary_cross_entropy_with_logits(logit,data[3]))

def run_fold(d,names,X,Q,W,B,order,experiment,seed,fold,otr,ote,root,out,max_epochs=100,patience=10,device='cpu'):
    mods,lam=VARIANTS[experiment];itr,iva=inner_split(d,otr,seed)
    for a,b in [(itr,iva),(otr,ote),(itr,ote),(iva,ote)]:assert not set(d.iloc[a].speaker)&set(d.iloc[b].speaker)
    inner=FoldPreprocessor(names).fit(X[itr],Q[itr],d.iloc[itr].file)
    # Selection sees neither outer-test nor inner-validation measurements in preprocessing.
    ti=tensors(d,X,W,B,itr,inner,mods,device);vi=tensors(d,X,W,B,iva,inner,mods,device)
    pairs_i,ties_i=rank_pairs(d,itr,order);init=seed*100+fold;seed_all(init)
    model=FusionNet(len(inner.indices_)).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(init);best=np.inf;best_epoch=1;wait=0;inner_logs=[]
    for epoch in range(1,max_epochs+1):
        log=train_epoch(model,opt,ti,pairs_i,lam,rng);vl=score(model,vi)[2];log.update(epoch=epoch,validation_bce=vl);inner_logs.append(log)
        if vl<best:best=vl;best_epoch=epoch;wait=0
        else:wait+=1
        if wait>=patience:break
    final=FoldPreprocessor(names).fit(X[otr],Q[otr],d.iloc[otr].file)
    td=tensors(d,X,W,B,otr,final,mods,device);pairs,ties=rank_pairs(d,otr,order)
    seed_all(init);model=FusionNet(len(final.indices_)).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    rng=np.random.default_rng(init);logs=[]
    for epoch in range(1,best_epoch+1):
        log=train_epoch(model,opt,td,pairs,lam,rng);log['epoch']=epoch;logs.append(log)
    model_id=f'{experiment}_s{seed}_f{fold}_v2'
    all_data=tensors(d,X,W,B,np.arange(len(d)),final,mods,device);prob,prog,_=score(model,all_data)
    membership=np.full(len(d),'train',dtype=object);membership[ote]='test'
    scores=pd.DataFrame(dict(file=d.file,speaker=d.speaker,label=d.label,seed=seed,fold=fold,experiment=experiment,model_id=model_id,scored_by_model=model_id,split=membership,group_prob=prob,prog_score=prog))
    target=out/'scores'/f'{model_id}.csv';scores.to_csv(target,index=False)
    checkpoint=root/'data/models/v2'/checkpoint_name(experiment,seed,fold)
    torch.save(dict(state_dict=model.state_dict(),preprocessor=final.to_dict(),inner_preprocessor=inner.to_dict(),model_id=model_id,experiment=experiment,mod_names=sorted(mods),best_epoch=best_epoch),checkpoint)
    log=dict(experiment=experiment,seed=seed,fold=fold,model_id=model_id,outer_train=d.iloc[otr].file.tolist(),outer_test=d.iloc[ote].file.tolist(),inner_train=d.iloc[itr].file.tolist(),inner_validation=d.iloc[iva].file.tolist(),inner_preprocessing=inner.to_dict(),final_preprocessing=final.to_dict(),inner_pairs=len(pairs_i),inner_ties_excluded=ties_i,final_pairs=len(pairs),final_ties_excluded=ties,inner_epochs=inner_logs,final_epochs=logs,checkpoint=str(checkpoint.relative_to(root)))
    (out/'audit'/f'{model_id}.json').write_text(json.dumps(log,indent=2))
    y=d.label.to_numpy()[ote];result=dict(experiment=experiment,seed=seed,fold=fold,auc=roc_auc_score(y,prob[ote]),bac=balanced_accuracy_score(y,prob[ote]>.5),best_epoch=best_epoch,n_inner_features=len(inner.indices_),n_features=len(final.indices_),n_rank_pairs=len(pairs),pair_exposures=sum(x['pair_exposures'] for x in logs),ranking_gradient_steps=sum(x['ranking_gradient_steps'] for x in logs))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--variants',default=','.join(VARIANTS));ap.add_argument('--seeds',default='0,1,2,3,4');ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--patience',type=int,default=10);ap.add_argument('--threads',type=int,default=1);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
    torch.set_num_threads(a.threads);root=a.root.resolve();variants=a.variants.split(',');seeds=list(map(int,a.seeds.split(',')))
    if not set(variants)<=set(VARIANTS):raise ValueError('Unrecognised experiment')
    d,names=load_candidates(root);W,B=load_embeddings(root,d,variants);X=d[names].to_numpy(float);Q=d[QUALITY].to_numpy(float);order=pd.read_csv(root/'data/processed/recording_order.csv')
    out=root/'data/results/v2';out.mkdir(exist_ok=True)
    for path in [out/'scores',out/'audit',root/'data/models/v2']:path.mkdir(parents=True,exist_ok=True)
    config=dict(protocol='V2: fully train-local inner selection; explicit ranking batches',python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,variants=variants,seeds=seeds,epochs=a.epochs,patience=a.patience,device=a.device,threads=a.threads,modality_dropout=.2,code_hashes={str(f.relative_to(root)):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in [Path(__file__).resolve(),root/'src/mcd/protocol.py',root/'src/mcd/models.py']},input_hashes=input_hashes(root),asr_provenance='Unresolved fixed upstream transcripts; no ASR regeneration',started=time.time())
    (out/'provenance.json').write_text(json.dumps(config,indent=2));results=[]
    for exp in variants:
        for seed in seeds:
            for fold,(otr,ote) in enumerate(outer_splits(d,seed)):
                z=run_fold(d,names,X,Q,W,B,order,exp,seed,fold,otr,ote,root,out,a.epochs,a.patience,a.device);results.append(z)
                pd.DataFrame(results).to_csv(out/'training_results.csv',index=False);print(json.dumps(z),flush=True)
    pd.DataFrame(results).groupby('experiment')[['auc','bac']].agg(['mean','std']).to_csv(out/'training_summary.csv')
if __name__=='__main__':main()
