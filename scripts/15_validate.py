"""Independent structural and numerical checks for the delivered secondary analyses."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import importlib.util as _ilu
_s = _ilu.spec_from_file_location("group_analysis", _P(__file__).resolve().parent / "11_group_analysis.py")
_m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
from group_analysis import *
import subprocess
def main():
 checks=[]
 def check(name,condition):
  if not bool(condition):raise AssertionError(name)
  checks.append(dict(check=name,passed=True))
 a=read('data/results/states_assignments.csv');m=read('data/processed/manifest.csv');df,_=load()
 if (REPO/'.git').exists():
  check('source branch is V0',subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='V0')
  check('source commit is pinned',subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()=='a391a57e5fe127dd44e211ffe6bd19e0d5777ac0')
  check('original source tree unchanged',subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()=='')
 else:
  check('input snapshot identifies pinned commit',(REPO/'SOURCE_COMMIT.txt').read_text().strip()=='a391a57e5fe127dd44e211ffe6bd19e0d5777ac0')
  fp=pd.read_csv(TABLES/'source_fingerprints.csv')
  check('input snapshot bytes match source fingerprints',all(hashlib.sha256((REPO/r.path).read_bytes()).hexdigest()==r.sha256 for r in fp.itertuples()))
 check('275 recordings and 141 speakers',len(m)==275 and m.speaker.nunique()==141)
 check('speaker group labels internally consistent',m.groupby('speaker').label.nunique().max()==1)
 check('no train-val speaker overlap in any saved model',a.groupby(['seed','fold','speaker']).split.nunique().max()==1)
 check('one held-out recording score per seed',a[a.split=='val'].groupby(['seed','file']).size().eq(1).all())
 check('five held-out appearances per recording',a[a.split=='val'].groupby('file').size().eq(5).all())
 check('rate identity holds on all recordings',np.max(np.abs(df.speech_rate-df.articulation_rate*(1-df.pause_ratio)))<1e-10)
 endpoint=pd.read_csv(TABLES/'longitudinal_endpoint_effects.csv');check('31 independent endpoint speakers',endpoint.n_speakers.eq(31).all())
 d=pd.read_csv(TABLES/'rate_decomposition.csv').set_index('component');check('decomposition components additive',abs(d.loc['log_r','mean_log_change']-d.loc[['log_a','log_nonpause'],'mean_log_change'].sum())<1e-10)
 sh=pd.read_csv(TABLES/'shap_family_uncertainty.csv');check('attribution family feature counts cover 37 features',sh.n_features.sum()==37)
 nn=pd.read_csv(TABLES/'supplementary_pipeline_permutations.csv');check('998 full-pipeline permutation evaluations',len(nn)==998 and nn.groupby('model').size().eq(499).all())
 pp=pd.read_csv(TABLES/'supplementary_oof_predictions.csv');check('speaker probe prediction probabilities bounded',pp.filter(regex='_seed').apply(lambda x:x.between(0,1).all()).all())
 gg=pd.read_csv(TABLES/'speaker_group_effects.csv');check('all FDR values in unit interval',gg.q_bh_37.between(0,1).all() and gg.adjusted_q_bh_37.between(0,1).all())
 X=np.column_stack([np.ones(141),df.groupby('speaker').label.first()]);Y=df.groupby('speaker').pause_mean_s.mean().values
 check('HC3 coefficient agrees with direct least-squares contrast',np.isclose(ols_hc3(X,Y)[0],np.linalg.lstsq(X,Y,rcond=None)[0][1]))
 save(pd.DataFrame(checks),'analysis_validation');print('Passed',len(checks),'checks')
if __name__=='__main__':main()
