from analysis import *
from scipy.stats import trim_mean
def run():
 df,m=load();order=read('data/processed/recording_order.csv').rename(columns={'recording':'file'});d=order[['file','speaker','rank']].merge(df.drop(columns='speaker'),on='file')
 for c,col in [('log_speech_rate','speech_rate'),('log_articulation_rate','articulation_rate')]:d[c]=np.log(d[col])
 d['log_nonpause']=np.log1p(-d.pause_ratio);cols=['log_speech_rate','log_articulation_rate','log_nonpause','log_duration']
 buck=d.groupby(['speaker','rank'])[cols].mean().reset_index();rows=[]
 for s,g in buck.groupby('speaker'):
  g=g.sort_values('rank')
  if len(g)<2:continue
  z=g.iloc[-1][cols]-g.iloc[0][cols];rows.append(z.to_dict())
 ch=pd.DataFrame(rows);out=[];signs=RNG.choice([-1,1],(10000,len(ch)))
 for c in cols:
  x=ch[c].to_numpy();null=(signs*x).mean(1);loo=np.array([np.delete(x,i).mean() for i in range(len(x))]);out.append(dict(feature=c,mean=x.mean(),median=np.median(x),trimmed_mean20=trim_mean(x,.2),min_leave_one_out_mean=loo.min(),max_leave_one_out_mean=loo.max(),p_signflip=(1+(abs(null)>=abs(x.mean())).sum())/10001,wilcoxon_p=stats.wilcoxon(x).pvalue,n_decrease=int((x<0).sum())))
 rr=pd.DataFrame(out);rr['q_bh_4']=bh(rr.p_signflip);save(rr,'relative_rate_robustness');print(rr.round(5).to_string(index=False),flush=True)
 # Duration-dependent repetition: correlation after residualising word count, as well as quality.
 a=read('data/results/states_assignments.csv');out=[]
 for (seed,fold),g in a.groupby(['seed','fold']):
  ref=np.sort(g[g.split=='train'].prog_score);g=g[g.split=='val'].copy();g['pct']=np.searchsorted(ref,g.prog_score)/len(ref);out.append(g)
 sc=pd.concat(out).groupby('file').pct.mean().rename('pct');z=df.join(sc,on='file').groupby('speaker')[['pct','bigram_rep','n_words','log_duration']+Q[1:]].mean().dropna()
 def partial(a,b,covs):
  C=np.column_stack([np.ones(len(z)),stats.rankdata(z[covs],axis=0)]);P=np.linalg.pinv(C);x=stats.rankdata(z[a]);y=stats.rankdata(z[b]);return stats.pearsonr(x-C@(P@x),y-C@(P@y)).statistic
 vals=[dict(adjustment='none',rho=stats.spearmanr(z.pct,z.bigram_rep).statistic,n=len(z)),dict(adjustment='word_count',rho=partial('pct','bigram_rep',['n_words']),n=len(z)),dict(adjustment='quality_and_word_count',rho=partial('pct','bigram_rep',Q+['n_words']),n=len(z))]
 save(pd.DataFrame(vals),'repetition_sampling_sensitivity')
 # Theoretical exact identity implies only two independent dimensions in rate/pause-rate trio.
 err=np.max(np.abs(df.speech_rate-df.articulation_rate*(1-df.pause_ratio)))
 save(pd.DataFrame([dict(check='speech_rate = articulation_rate * (1 - pause_ratio)',max_absolute_error=err,n_recordings=len(df))]),'timing_identity')
if __name__=='__main__':run()
