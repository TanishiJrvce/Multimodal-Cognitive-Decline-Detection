import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,pytest,torch
from mcd.protocol import FoldPreprocessor,rank_pairs,checkpoint_name,load_embeddings,load_candidates,inner_split,outer_splits
ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_unique_classification_and_joint_checkpoint_names():
    assert checkpoint_name('V_cls',0,0)!=checkpoint_name('V_joint',0,0)

def test_outcome_and_metadata_rejected():
    for name in ['label','speaker','rank','state','row_index']:
        with pytest.raises(ValueError):FoldPreprocessor([name,'speech_rate'])

def test_train_only_preprocessing_is_independent_of_test_values():
    rng=np.random.default_rng(3);x=rng.normal(size=(100,6));q=rng.normal(size=(100,5));x[0,1]=np.nan
    p=FoldPreprocessor([f'f{i}' for i in range(6)]).fit(x[:70],q[:70],list(range(70)));old=p.to_dict();p.transform(x[70:]*1e6)
    assert old==p.to_dict() and np.isfinite(p.transform(x)).all()

def test_all_real_outer_and_inner_partitions_are_disjoint():
    d,_=load_candidates(ROOT)
    for seed in range(5):
        for a,b in outer_splits(d,seed):
            i,j=inner_split(d,a,seed)
            assert not set(d.iloc[a].speaker)&set(d.iloc[b].speaker)
            assert not set(d.iloc[i].speaker)&set(d.iloc[j].speaker)
            assert set(i)|set(j)==set(a)

def test_real_pair_builder_excludes_ties_and_other_partitions():
    d,_=load_candidates(ROOT);o=pd.read_csv(ROOT/'data/processed/recording_order.csv');ids=np.arange(len(d));pairs,ties=rank_pairs(d,ids,o)
    lookup=o.set_index('recording')['rank'];assert ties>0 and len(pairs)>0
    for i,j in pairs:
        assert d.iloc[i].speaker==d.iloc[j].speaker
        assert lookup[d.iloc[i].file]<lookup[d.iloc[j].file]
    sub=np.arange(50);ps,_=rank_pairs(d,sub,o);assert all(i<50 and j<50 for i,j in ps)

def test_feat_only_needs_no_embeddings_and_other_variants_fail_closed(tmp_path):
    d=pd.DataFrame({'file':['x','y']});a,b=load_embeddings(tmp_path,d,['V_feat']);assert a.shape==(2,768) and not a.any()
    with pytest.raises(FileNotFoundError):load_embeddings(tmp_path,d,['V_joint'])

def test_alignment_uses_index_not_file_order_and_rejects_duplicate_indices(tmp_path):
    p=tmp_path/'data/processed';p.mkdir(parents=True);pd.DataFrame({'file':['b','a'],'row_index':[0,1]}).to_csv(p/'embeddings_index.csv',index=False)
    a=np.stack([np.zeros(768),np.ones(768)]);np.save(p/'embeddings_wavlm.npy',a);np.save(p/'embeddings_bert.npy',a*2)
    x,y=load_embeddings(tmp_path,pd.DataFrame({'file':['a','b']}),['V_joint']);assert x[0,0]==1 and y[0,0]==2
    pd.DataFrame({'file':['b','a'],'row_index':[0,0]}).to_csv(p/'embeddings_index.csv',index=False)
    with pytest.raises(ValueError):load_embeddings(tmp_path,pd.DataFrame({'file':['a','b']}),['V_joint'])

def test_state_fit_rejects_unidentified_or_mixed_model_scores():
    m=module('states_v2','08_states_v2.py')
    with pytest.raises(ValueError):m.validate_score_source(pd.DataFrame({'file':['x'],'prog_score':[0.]}))
    z=pd.DataFrame({'model_id':['A','A'],'scored_by_model':['A','B'],'split':['train','test'],'file':['x','y'],'prog_score':[0.,1.]})
    with pytest.raises(ValueError):m.validate_score_source(z)

def test_explicit_ranking_batches_have_observed_gradients():
    from mcd.models import FusionNet
    m=module('training_v2','07_train_joint_v2.py');torch.set_num_threads(1);torch.manual_seed(4)
    net=FusionNet(6);opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    data=[torch.randn(40,6),torch.randn(40,768),torch.randn(40,768),torch.tensor([0.,1.]*20)]
    z=m.train_epoch(net,opt,data,[(0,1),(2,3)],.5,np.random.default_rng(4));assert z['pair_exposures']==4 and z['ranking_gradient_steps']==2
