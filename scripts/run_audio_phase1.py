"""Streaming exploratory Audio Phase 1 runner. No biological claims or raw-data writes."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.audio_config import AudioPhase1Config
from src.audio_io import inventory_rows, add_gap_columns

def main():
    cfg=AudioPhase1Config(); root=cfg.validate_root(); tables=ROOT/'results/tables/audio'; tables.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,row in enumerate(inventory_rows(root),1):
        rows.append(row)
        if index % 5000 == 0: print(f'headers read: {index}', flush=True)
    inv=add_gap_columns(pd.DataFrame(rows)); inv.to_csv(tables/'audio_inventory.csv',index=False)
    valid=inv.loc[inv.readable & inv.timestamp_valid].copy(); valid['date']=pd.to_datetime(valid.start_timestamp).dt.date
    summary={'audio_root':str(root),'valid_wav_files':int(len(valid)),'ignored_appledouble_wav_like_files':2,'unreadable_files':int((~inv.readable).sum()),'timestamp_parsing_failures':int((~inv.timestamp_valid).sum()),'earliest_timestamp':str(valid.start_timestamp.min()),'latest_timestamp':str(valid.start_timestamp.max()),'duplicate_timestamp_rows':int(valid.start_timestamp.duplicated(keep=False).sum()),'sample_rate_distribution':valid.sample_rate_hz.value_counts().to_dict(),'channel_distribution':valid.channels.value_counts().to_dict(),'duration_seconds':{'min':float(valid.duration_seconds.min()),'median':float(valid.duration_seconds.median()),'max':float(valid.duration_seconds.max())},'total_recorded_audio_seconds':float(valid.duration_seconds.sum()),'total_uncovered_positive_seconds':float(valid.uncovered_gap_seconds.clip(lower=0).sum()),'overlap_count':int((valid.uncovered_gap_seconds<0).sum())}
    (tables/'audio_dataset_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    daily=valid.groupby('date').agg(file_count=('filename','size'),total_audio_seconds=('duration_seconds','sum'),median_duration_seconds=('duration_seconds','median'),dominant_sample_rate=('sample_rate_hz',lambda x:x.mode().iloc[0]),dominant_channel_count=('channels',lambda x:x.mode().iloc[0]),median_gap_seconds=('uncovered_gap_seconds','median'),maximum_gap_seconds=('uncovered_gap_seconds','max')).reset_index(); daily['total_audio_hours']=daily.total_audio_seconds/3600; daily.to_csv(tables/'audio_daily_summary.csv',index=False)
    (tables/'audio_phase1_parameters.json').write_text(json.dumps({'audio_root':str(root),**cfg.__dict__},default=str,indent=2))
    (tables/'audio_phase1_status.json').write_text(json.dumps({'inventory_complete':True,'metadata_qc_complete':True,'representative_analysis_complete':False,'peak_parameters_provisional':False,'july13_16_processing_complete':False,'image_audio_alignment_complete':False,'ready_for_audio_phase2':False,'blockers':['Signal-analysis stage not yet run in this invocation.'],'warnings':['Inventory is metadata-only; recordings are intermittent.'],'recommended_next_action':'Review inventory QC before running representative waveform analysis.'},indent=2))
if __name__=='__main__': main()
