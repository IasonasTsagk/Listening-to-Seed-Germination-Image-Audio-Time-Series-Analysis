"""Bounded, descriptive noise/event characterization; no filtering or biology labels."""
from pathlib import Path
import json,sys,numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy import signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.audio_features import read_wav_float,basic_features
def km(x,k,iters=40):
 rng=np.random.default_rng(27); c=x[rng.choice(len(x),k,replace=False)]
 for _ in range(iters):
  lab=((x[:,None]-c[None])**2).sum(2).argmin(1); new=np.array([x[lab==i].mean(0) if np.any(lab==i) else c[i] for i in range(k)])
  if np.allclose(c,new):break
  c=new
 return lab
def main():
 tab=ROOT/'results/tables/audio';fig=ROOT/'results/figures/audio/noise_characterization';fig.mkdir(parents=True,exist_ok=True);root=Path(json.load(open(tab/'audio_phase1_parameters.json'))['audio_root'])
 h=pd.read_csv(tab/'audio_hourly_summary_july13_18.csv',parse_dates=['hour']); p=pd.read_csv(tab/'audio_candidate_peaks_july13_18.csv',parse_dates=['absolute_peak_timestamp']);
 top=h[h.hour.dt.date.astype(str).isin(['2026-07-17','2026-07-18'])].nlargest(6,'high_prominence_peaks_per_recorded_minute'); rows=[]
 for r in top.itertuples(): rows.append({'date':r.hour.date(),'start_timestamp':r.hour,'end_timestamp':r.hour+pd.Timedelta(hours=1),'recorded_duration_seconds':r.recorded_audio_seconds,'candidate_peak_rate':r.candidate_peaks_per_recorded_minute,'high_prominence_peak_rate':r.high_prominence_peaks_per_recorded_minute,'mean_rms':r.mean_rms,'reason_selected':'Top July 17–18 high-prominence hourly rate'})
 pd.DataFrame(rows).to_csv(tab/'acoustic_review_windows.csv',index=False)
 low=p[p.absolute_peak_timestamp.dt.date.astype(str)=='2026-07-13'].sample(20,random_state=27); med=p.iloc[np.argsort(np.abs(p.prominence-p.prominence.median()))[:20]]; high=p.nlargest(30,'prominence'); sample=pd.concat([low,med,high]).drop_duplicates('absolute_peak_timestamp').head(90).copy();sample.insert(0,'review_id',range(1,len(sample)+1));sample['selection_reason']=np.where(sample.prominence>=p.prominence.quantile(.9),'high prominence','baseline/median deterministic sample');sample.to_csv(tab/'manual_audio_review_manifest.csv',index=False)
 chars=[]
 for r in sample.itertuples():
  rate,s=read_wav_float(root/r.source_relative_path);x=s.mean(1); center=int(r.relative_peak_time_seconds*rate);w=int(.15*rate); seg=x[max(0,center-w):min(len(x),center+w)]; d=basic_features(seg,rate);f,px=signal.periodogram(seg,rate); bands=[px[(f>=a)&(f<b)].sum()/(px.sum()+1e-15) for a,b in [(0,500),(500,4000),(4000,24000)]];chars.append({'review_id':r.review_id,'absolute_peak_timestamp':r.absolute_peak_timestamp,'prominence':r.prominence,'event_width_seconds':r.width_seconds,**d,'low_frequency_fraction':bands[0],'mid_frequency_fraction':bands[1],'high_frequency_fraction':bands[2]})
 c=pd.DataFrame(chars); cols=['rms_energy','peak_absolute_amplitude','zero_crossing_rate_hz','spectral_centroid_hz','spectral_bandwidth_hz','spectral_rolloff_hz','spectral_flatness','dominant_frequency_hz','prominence'];x=c[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy();x=(x-x.mean(0))/(x.std(0)+1e-12);c['cluster']=km(x,3);c.to_csv(tab/'audio_event_characterization.csv',index=False)
 intervals=p.sort_values('absolute_peak_timestamp').absolute_peak_timestamp.diff().dt.total_seconds().dropna();pd.DataFrame({'inter_event_interval_seconds':intervals}).describe(percentiles=[.1,.5,.9]).to_csv(tab/'inter_event_interval_summary.csv');
 z=x@np.linalg.svd(x,full_matrices=False)[2][:2].T; plt.scatter(z[:,0],z[:,1],c=c.cluster);plt.title('Exploratory event-feature PCA');plt.savefig(fig/'event_feature_pca.png',dpi=140);plt.close();plt.hist(np.log10(intervals[intervals>0]),bins=80);plt.title('Inter-event interval distribution (log10 seconds)');plt.savefig(fig/'inter_event_interval_distribution.png',dpi=140);plt.close();
 cs=c.groupby('cluster').agg(event_count=('review_id','size'),mean_prominence=('prominence','mean'),mean_rms=('rms_energy','mean'),mean_centroid=('spectral_centroid_hz','mean'),mean_bandwidth=('spectral_bandwidth_hz','mean'),mean_dominant_frequency=('dominant_frequency_hz','mean')).reset_index();cs.to_csv(tab/'audio_cluster_summary.csv',index=False)
 plt.figure(figsize=(13,5));plt.plot(h.hour,h.candidate_peaks_per_recorded_minute,label='all candidate');plt.plot(h.hour,h.high_prominence_peaks_per_recorded_minute,label='high prominence');plt.legend();plt.title('Hourly activity, coverage normalized');plt.savefig(fig/'july13_18_hourly_activity.png',dpi=140);plt.close();
 status={'noise_characterization_complete':True,'manual_review_sample_created':True,'periodicity_analysis_complete':True,'exploratory_clustering_complete':True,'ready_for_targeted_multimodal_review':True,'cautions':['Clusters are neutral acoustic-pattern groups, not biological classes.','No event has been removed or labelled as germination.']};json.dump(status,open(tab/'audio_phase2a_status.json','w'),indent=2)
if __name__=='__main__':main()
