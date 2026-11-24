"""Checkpointed July 13–18 acoustic-activity extraction; candidate peaks only."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.audio_config import AudioPhase1Config
from src.audio_features import read_wav_float,rms_envelope,basic_features
from src.audio_peaks import detect_candidate_peaks
def main():
 cfg=AudioPhase1Config(); root=cfg.validate_root(); tab=ROOT/'results/tables/audio'; log=ROOT/'results/logs/audio';log.mkdir(parents=True,exist_ok=True)
 inv=pd.read_csv(tab/'audio_inventory.csv',parse_dates=['start_timestamp','end_timestamp']); f=inv[(inv.start_timestamp>='2026-07-13')&(inv.start_timestamp<='2026-07-18 23:59:59')&inv.readable].sort_values('start_timestamp'); status=json.loads((tab/'audio_phase1_status.json').read_text()); prom=float(status['provisional_phase1_parameters']['prominence']); done=tab/'audio_file_features_july13_18.csv'; donepaths=set(pd.read_csv(done,usecols=['relative_path']).relative_path) if done.exists() else set(); features=[]; events=[]
 for i,r in enumerate(f.itertuples(index=False),1):
  if r.relative_path in donepaths: continue
  try:
   rate,s=read_wav_float(root/r.relative_path); mono=s.mean(axis=1); ts,e=rms_envelope(mono,rate,.025,.01,3); peaks=detect_candidate_peaks(ts,e,prom,.1); bf=basic_features(mono,rate); ps=[p['prominence'] for p in peaks]
   features.append({'filename':r.filename,'relative_path':r.relative_path,'start_timestamp':r.start_timestamp,'end_timestamp':r.end_timestamp,'duration_seconds':r.duration_seconds,**bf,'candidate_peak_count':len(peaks),'candidate_peaks_per_recorded_second':len(peaks)/r.duration_seconds,'maximum_prominence':max(ps) if ps else np.nan,'mean_prominence':np.mean(ps) if ps else np.nan,'median_prominence':np.median(ps) if ps else np.nan,'maximum_rms_envelope_value':float(e.max())})
   for p in peaks: events.append({'source_audio_file':r.filename,'source_relative_path':r.relative_path,'file_start_timestamp':r.start_timestamp,'relative_peak_time_seconds':p['relative_peak_time_seconds'],'absolute_peak_timestamp':r.start_timestamp+pd.Timedelta(seconds=p['relative_peak_time_seconds']),'envelope_peak_value':p['envelope_peak_value'],'prominence':p['prominence'],'width_seconds':p['width_seconds'],'near_start_boundary':p['relative_peak_time_seconds']<.05,'near_end_boundary':p['relative_peak_time_seconds']>r.duration_seconds-.05,'parameter_set_id':'provisional_phase1_medium'})
  except Exception as e: features.append({'filename':r.filename,'relative_path':r.relative_path,'start_timestamp':r.start_timestamp,'duration_seconds':r.duration_seconds,'error_message':f'{type(e).__name__}: {e}'})
  if i%500==0:
   pd.DataFrame(features).to_csv(done,mode='a',index=False,header=not done.exists()); features=[]
   ev=tab/'audio_candidate_peaks_july13_18.csv'; pd.DataFrame(events).to_csv(ev,mode='a',index=False,header=not ev.exists());events=[]
   (log/'audio_phase1c_progress.json').write_text(json.dumps({'processed_iteration':i,'total':len(f),'updated_at':time.time()}))
 pd.DataFrame(features).to_csv(done,mode='a',index=False,header=not done.exists());ev=tab/'audio_candidate_peaks_july13_18.csv';pd.DataFrame(events).to_csv(ev,mode='a',index=False,header=not ev.exists())
 feats=pd.read_csv(done,parse_dates=['start_timestamp']); peaks=pd.read_csv(ev,parse_dates=['absolute_peak_timestamp']); q75,q90=peaks.prominence.quantile([.75,.9]); peaks['moderate']=peaks.prominence>=q75;peaks['high']=peaks.prominence>=q90; peaks.to_csv(ev,index=False)
 feats['hour']=feats.start_timestamp.dt.floor('h'); hourly=feats.groupby('hour').agg(recording_count=('filename','size'),recorded_audio_seconds=('duration_seconds','sum'),mean_rms=('rms_energy','mean'),median_rms=('rms_energy','median'),maximum_rms=('rms_energy','max')).reset_index(); pc=peaks.groupby(peaks.absolute_peak_timestamp.dt.floor('h')).agg(candidate_peak_count=('prominence','size'),moderate_peak_count=('moderate','sum'),high_peak_count=('high','sum')).reset_index(names='hour'); hourly=hourly.merge(pc,on='hour',how='left').fillna(0); hourly['recorded_audio_minutes']=hourly.recorded_audio_seconds/60; hourly['candidate_peaks_per_recorded_minute']=hourly.candidate_peak_count/hourly.recorded_audio_minutes;hourly['high_prominence_peaks_per_recorded_minute']=hourly.high_peak_count/hourly.recorded_audio_minutes;hourly.to_csv(tab/'audio_hourly_summary_july13_18.csv',index=False)
 daily=hourly.assign(date=hourly.hour.dt.date).groupby('date').agg(recording_count=('recording_count','sum'),recorded_audio_seconds=('recorded_audio_seconds','sum'),candidate_peak_count=('candidate_peak_count','sum'),high_peak_count=('high_peak_count','sum'),mean_rms=('mean_rms','mean')).reset_index();daily['recorded_audio_hours']=daily.recorded_audio_seconds/3600;daily['candidate_peaks_per_recorded_minute']=daily.candidate_peak_count/(daily.recorded_audio_seconds/60);daily['high_prominence_peaks_per_recorded_minute']=daily.high_peak_count/(daily.recorded_audio_seconds/60);daily.to_csv(tab/'audio_analysis_daily_summary_july13_18.csv',index=False)
 status.update({'full_july13_18_processing_complete':True,'hourly_summary_complete':True,'daily_summary_complete':True,'image_audio_alignment_complete':False,'candidate_multimodal_windows_complete':False,'ready_for_audio_phase2':False});(tab/'audio_phase1_status.json').write_text(json.dumps(status,indent=2,default=str))
if __name__=='__main__':main()
