"""Representative-only Audio Phase 1B validation; no full-window scaling."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import signal
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.audio_config import AudioPhase1Config
from src.audio_features import read_wav_float, channel_diagnostics, rms_envelope, basic_features
from src.audio_peaks import detect_candidate_peaks

def main():
 cfg=AudioPhase1Config(); root=cfg.validate_root(); tables=ROOT/'results/tables/audio'; figs=ROOT/'results/figures/audio'; reps=figs/'representative_windows'; figs.mkdir(parents=True,exist_ok=True); reps.mkdir(parents=True,exist_ok=True)
 inv=pd.read_csv(tables/'audio_inventory.csv',parse_dates=['start_timestamp','end_timestamp','next_file_start_timestamp']); valid=inv[inv.readable & inv.timestamp_valid].copy()
 ov=valid[valid.uncovered_gap_seconds<0].copy(); ov['apparent_overlap_seconds']=-ov.uncovered_gap_seconds; ov['next_relative_path']=ov.relative_path.shift(-1); ov['next_parent_directory']=ov.parent_directory.shift(-1); ov['same_parent_directory']=ov.parent_directory.eq(ov.next_parent_directory)
 ov['overlap_bin']=pd.cut(ov.apparent_overlap_seconds,[-.01,.25,.75,1.25,1.75,2.25,2.76],include_lowest=True).astype(str); ov.to_csv(tables/'audio_overlap_diagnostics.csv',index=False)
 overlap_summary={'apparent_overlap_count':int(len(ov)),'overlap_seconds':{'min':float(ov.apparent_overlap_seconds.min()),'median':float(ov.apparent_overlap_seconds.median()),'max':float(ov.apparent_overlap_seconds.max())},'overlap_bins':ov.overlap_bin.value_counts().to_dict(),'same_parent_fraction':float(ov.same_parent_directory.mean()),'by_date':{str(k):int(v) for k,v in ov.groupby(ov.start_timestamp.dt.date).size().items()},'interpretation':'Whole-second filename timestamps and 2.75-second snippets can create apparent overlaps; no recordings were deduplicated, trimmed, merged, or discarded.'}; (tables/'audio_overlap_summary.json').write_text(json.dumps(overlap_summary,indent=2,default=str))
 dates=pd.date_range('2026-07-13','2026-07-18',freq='D'); chosen=[]
 for day in dates:
  dayrows=valid[valid.start_timestamp.dt.date==day.date()].copy(); dayrows['noon_delta']=(dayrows.start_timestamp-(day+pd.Timedelta(hours=12))).abs(); chosen.append(dayrows.sort_values('noon_delta').iloc[0])
 channel=[]; analyzed=[]
 for rec in chosen:
  path=root/rec.relative_path; rate,samples=read_wav_float(path); diag=channel_diagnostics(samples); mono=samples.mean(axis=1); times,env=rms_envelope(mono,rate,cfg.envelope_window_seconds,cfg.envelope_hop_seconds,cfg.envelope_smoothing_frames); diag.update({'timestamp':rec.start_timestamp,'filename':rec.filename,'relative_path':rec.relative_path,'sample_rate_hz':rate,'normalized_channel_difference':float(np.sqrt(np.mean((samples[:,0]-samples[:,1])**2))/(np.sqrt(np.mean(mono**2))+1e-12)),'dc_offset_mono':float(mono.mean()),'clipping_fraction':float((np.abs(mono)>=.999).mean()),'envelope_std':float(env.std())}); channel.append(diag); analyzed.append((rec,rate,samples,mono,times,env))
 pd.DataFrame(channel).to_csv(tables/'channel_diagnostics.csv',index=False)
 base=float(np.median([x[-1].std() for x in analyzed])); configs=[('low',base*.5,.1),('medium',base, .1),('high',base*2,.1),('medium_long_distance',base,.25)]; sensitivity=[]
 for rec,rate,samples,mono,times,env in analyzed:
  for ident,prom,dist in configs:
   peaks=detect_candidate_peaks(times,env,prom,dist); sensitivity.append({'parameter_set_id':ident,'timestamp':rec.start_timestamp,'filename':rec.filename,'prominence':prom,'min_distance_seconds':dist,'peak_count':len(peaks),'peaks_per_recorded_minute':len(peaks)/(len(mono)/rate/60),'median_prominence':float(np.median([p['prominence'] for p in peaks])) if peaks else np.nan})
 pd.DataFrame(sensitivity).to_csv(tables/'peak_parameter_sensitivity.csv',index=False)
 for rec,rate,samples,mono,times,env in analyzed:
  peaks=detect_candidate_peaks(times,env,base,.1); fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=False,constrained_layout=True); sec=np.arange(len(mono))/rate; axes[0].plot(sec,mono,lw=.35); axes[0].set_title(f'{rec.start_timestamp} — raw mean-mono waveform'); axes[1].plot(times,env); axes[1].scatter([p['relative_peak_time_seconds'] for p in peaks],[p['envelope_peak_value'] for p in peaks],c='crimson'); axes[1].set_title('RMS envelope and candidate peaks'); f,t,s=signal.spectrogram(mono,rate,nperseg=cfg.spectrogram_nperseg,noverlap=cfg.spectrogram_noverlap); axes[2].pcolormesh(t,f,10*np.log10(s+1e-15),shading='auto'); axes[2].set_ylim(0,rate/2); axes[2].set_title('Spectrogram (dB, no frequency filtering)'); fig.savefig(reps/f'{pd.Timestamp(rec.start_timestamp).strftime("%Y%m%d_%H%M%S")}.png',dpi=140); plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5)); pd.DataFrame(sensitivity).groupby('parameter_set_id').peaks_per_recorded_minute.median().plot.bar(ax=ax); ax.set_ylabel('Candidate peaks / recorded minute'); ax.set_title('Representative find_peaks sensitivity (exploratory)'); fig.savefig(figs/'peak_parameter_sensitivity.png',dpi=140,bbox_inches='tight'); plt.close(fig)
 first=analyzed[0]; fig,ax=plt.subplots(figsize=(12,4)); t=np.arange(first[2].shape[0])/first[1]; ax.plot(t,first[2][:,0],label='left',alpha=.75); ax.plot(t,first[2][:,1],label='right',alpha=.65); ax.legend(); ax.set_title('Representative stereo channel comparison'); fig.savefig(figs/'channel_comparison_example.png',dpi=140,bbox_inches='tight'); plt.close(fig)
 status=json.loads((tables/'audio_phase1_status.json').read_text()); status.update({'overlap_diagnostics_complete':True,'channel_validation_complete':True,'representative_analysis_complete':True,'peak_parameters_provisional':True,'ready_for_full_july13_18_processing':True,'analysis_window':'2026-07-13 through 2026-07-18 (July 18 partial)','provisional_phase1_parameters':{'prominence':base,'minimum_peak_distance_seconds':.1,'envelope_window_seconds':.025,'envelope_hop_seconds':.01,'smoothing_frames':3,'channel_strategy':'mean_mono, justified by representative diagnostics only'},'warnings':['Candidate peaks are exploratory acoustic events, not germination events.','Apparent overlaps may reflect whole-second timestamp quantization.','No recording is not acoustic silence.']}); (tables/'audio_phase1_status.json').write_text(json.dumps(status,indent=2,default=str))
if __name__=='__main__': main()
