from analysis import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
FIG=BASE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.bbox':'tight'})
TEAL='#127f85';NAVY='#15324e';AMBER='#bc732d';GRAY='#9cabb9'
def emit(fig,name):
 fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=220);fig.savefig(FIG/(name+'.pdf'));plt.close(fig)
def main():
 r=pd.read_csv(TABLES/'speaker_group_effects.csv');fig,axs=plt.subplots(1,3,figsize=(10.5,3.5))
 for ax,(c,label,scale) in zip(axs,[('pause_mean_s','Pause duration difference (ms)',1000),('pause_ratio','Pause-share difference (percentage points)',100),('mattr50','MATTR-50 difference',1)]):
  z=r[r.feature==c].iloc[0];ax.errorbar([0,1],[z.difference*scale,z.adjusted_difference*scale],yerr=[[scale*(z.difference-z.ci_low),scale*(z.adjusted_difference-z.adjusted_low)],[scale*(z.ci_high-z.difference),scale*(z.adjusted_high-z.adjusted_difference)]],fmt='o',color=TEAL,capsize=5);ax.axhline(0,color=GRAY,ls='--');ax.set_xticks([0,1],['Unadjusted','Adjusted']);ax.set_ylabel(label);ax.set_title(c.replace('_',' '),fontsize=11)
 emit(fig,'01_speaker_effects')
 j=read('data/results/joint_results.csv');pv=j.pivot(index=['seed','fold'],columns='experiment',values='auc');fig,ax=plt.subplots(figsize=(9,3.6))
 for seed in range(5):
  v=(pv.V_joint-pv.V_cls).loc[seed].values;ax.scatter(seed+np.linspace(-.16,.16,5),v,color=TEAL,s=35);ax.plot([seed-.25,seed+.25],[v.mean()]*2,color=NAVY,lw=3)
 ax.axhline(0,color=GRAY,ls='--');ax.set_xticks(range(5),['Seed '+str(x) for x in range(5)]);ax.set_ylabel('Joint minus classification-only AUC');ax.set_title('Temporal supervision: small mean gain, heterogeneous paired changes')
 emit(fig,'02_original_paired_gains')
 d=pd.read_csv(TABLES/'rate_decomposition.csv');fig,ax=plt.subplots(figsize=(8.8,3.6));x=np.arange(3);ax.bar(x,d.mean_log_change,color=[NAVY,TEAL,GRAY],width=.58);ax.errorbar(x,d.mean_log_change,yerr=[d.mean_log_change-d.low,d.high-d.mean_log_change],fmt='none',ecolor='#333',capsize=5);ax.axhline(0,color=GRAY,ls='--');ax.set_xticks(x,['Total word rate','Pause-excluded word rate','Non-pause share']);ax.set_ylabel('Mean log(later / earlier)');ax.set_title('Within-person slowing: exact decomposition, 31 speakers');ax.text(.02,.05,'Total = pause-excluded rate + non-pause share on the log scale\nExploratory pointwise 95% speaker-bootstrap intervals',transform=ax.transAxes,fontsize=9,va='bottom')
 emit(fig,'03_rate_decomposition')
 t=pd.read_csv(TABLES/'temporal_generalization.csv');fig,ax=plt.subplots(figsize=(8.5,3.5));xx=np.arange(2);ax.bar(xx,t.concordance,color=[GRAY,TEAL],width=.55);ax.errorbar(xx,t.concordance,yerr=[t.concordance-t.low,t.high-t.concordance],fmt='none',ecolor='black',capsize=5);ax.axhline(.5,color=AMBER,ls='--',label='Chance ordering');ax.set_ylim(0,1);ax.set_xticks(xx,['Training speakers','Held-out speakers']);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Speaker-balanced pair concordance');ax.set_title('Learned temporal alignment and its portability are distinct');ax.legend(loc='lower left')
 for i,v in enumerate(t.concordance):ax.text(i,v-.08,f'{100*v:.1f}%',ha='center',color='white',fontweight='bold')
 emit(fig,'04_temporal_generalization')
 s=pd.read_csv(TABLES/'shap_family_uncertainty.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.6));names=['Timing','Language','Pitch / voice','Spectral'];axs[0].bar(names,s.share,color=[TEAL,GRAY,GRAY,NAVY]);axs[0].errorbar(range(4),s.share,yerr=[s.share-s.share_low,s.share_high-s.share],fmt='none',ecolor='black',capsize=4);axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[0].set_title('Total attribution share');axs[1].bar(names,s.per_feature_mean,color=[TEAL,GRAY,GRAY,NAVY]);axs[1].set_title('Mean attribution per feature');axs[1].set_ylabel('Mean absolute SHAP');
 for ax in axs:ax.tick_params(axis='x',rotation=15)
 emit(fig,'05_attribution_normalization')
 ss=pd.read_csv(TABLES/'oof_progression_feature_associations.csv');wanted=['log_duration','bigram_rep','speech_rate','pause_ratio','mattr50','mean_sent_len'];ss=ss.set_index('feature').loc[wanted];fig,ax=plt.subplots(figsize=(9,3.8));xx=np.arange(len(ss));ax.barh(xx-.15,ss.rho,height=.27,color=GRAY,label='Observed rank correlation');ax.barh(xx+.15,ss.partial_rho_quality.fillna(0),height=.27,color=TEAL,label='Quality-adjusted rank association');ax.set_yticks(xx,['Recording duration','Bigram repetition','Speech rate','Pause proportion','MATTR-50','Sentence length']);ax.axvline(0,color='#777');ax.set_xlim(-.55,.4);ax.invert_yaxis();ax.set_xlabel('Spearman rho / partial rank correlation');ax.legend(loc='upper left',bbox_to_anchor=(0,-.23),fontsize=8);ax.set_title('What the held-out progression coordinate tracks');fig.text(.15,-.03,'Duration has no adjusted bar because it is included in the adjustment set.',fontsize=8)
 emit(fig,'06_score_associations')
 pr=pd.read_csv(TABLES/'supplementary_probe_summary.csv');pr=pr[pr.model!='Gender1'].sort_values('auc');fig,ax=plt.subplots(figsize=(9,4.8));xx=np.arange(len(pr));ax.errorbar(pr.auc,xx,xerr=[pr.auc-pr.auc_low,pr.auc_high-pr.auc],fmt='o',color=TEAL,capsize=3);ax.set_yticks(xx,pr.model);ax.axvline(.5,color=GRAY,ls='--');ax.set_xlim(.3,.75);ax.set_xlabel('AUC of averaged OOF predictions, conditional bootstrap interval');ax.set_title('Supplementary fixed linear probes on speaker-averaged features')
 emit(fig,'07_supplementary_probes')
 bic=read('data/results/states_bic.csv').pivot(index=['seed','fold'],columns='k',values='bic');u=pd.read_csv(TABLES/'oof_state_uncertainty.csv').iloc[0];fig,axs=plt.subplots(1,2,figsize=(10,3.6));delta=bic[2]-bic[1];axs[0].bar(range(25),delta,color=[TEAL if v>0 else AMBER for v in delta]);axs[0].axhline(0,color='#777');axs[0].set_ylabel('BIC(2) - BIC(1)');axs[0].set_xlabel('Model run');axs[0].set_title('One component preferred in 24/25 runs');axs[1].bar(['Always low','Always high','Variable'],[u.always_low,u.always_high,u.variable],color=[GRAY,NAVY,TEAL]);axs[1].set_ylabel('Recordings');axs[1].set_title('State membership across five held-out assignments')
 emit(fig,'08_state_structure')
 print('Created',len(list(FIG.glob('*.png'))),'figures')
if __name__=='__main__':main()
