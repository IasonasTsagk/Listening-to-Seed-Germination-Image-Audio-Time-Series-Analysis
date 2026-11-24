"""Phase 2.7 zero-shot pretrained-model benchmark helpers.

No method here treats detections, masks, or components as seeds.  Optional model
imports are deliberately lazy so normal project tests never download checkpoints.
"""
from __future__ import annotations
from pathlib import Path
import json, time
import cv2
import numpy as np
import pandas as pd
from .image_utils import load_image, color_spaces
from .preprocessing import SegmentationConfig, create_foreground_mask

OFFICIAL_RELEASE = "https://github.com/The-Zhou-Lab/SeedGerm-VIG/releases/tag/v1.12"

def device_name() -> str:
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError: return "cpu"

def mask_metrics(mask: np.ndarray) -> dict:
    binary=(mask>0).astype(np.uint8); n,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
    area=int(binary.sum()); return {"foreground_area_pixels":area,"foreground_percentage":area/binary.size*100,"component_count":n-1}

def bbox_row(box, timestamp, filename, detection_id, class_names: dict, runtime_seconds: float) -> dict:
    x1,y1,x2,y2=map(float,box.xyxy[0].tolist()); cid=int(box.cls[0]); return {"timestamp":timestamp,"filename":filename,"detection_id":detection_id,"class_id":cid,"class_name":str(class_names.get(cid,cid)),"confidence":float(box.conf[0]),"x_min":x1,"y_min":y1,"x_max":x2,"y_max":y2,"bbox_width":x2-x1,"bbox_height":y2-y1,"bbox_area":(x2-x1)*(y2-y1),"runtime_seconds":runtime_seconds}

def availability(name, implementation, checkpoint: Path|None, reason="") -> dict:
    exists=bool(checkpoint and checkpoint.exists()); return {"model_name":name,"implementation":implementation,"checkpoint_source":OFFICIAL_RELEASE if checkpoint else None,"checkpoint_path":str(checkpoint) if checkpoint else None,"available":exists,"successfully_loaded":False,"inference_success":False,"device":device_name(),"notes":"","failure_reason":reason}

def run_benchmark(inventory: pd.DataFrame, tables: Path, figures: Path, models: Path) -> tuple[pd.DataFrame,dict]:
    """Run Otsu for all frames and attempt official YOLO/U-Net when importable."""
    tables.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    frames=inventory.loc[inventory.is_readable].sort_values("timestamp"); rows=[]; yolo_rows=[]
    yolo_path=models/"models/YOLOv8x-Germ.pt"; unet_path=models/"models/U-net_wheat_seed_detection"
    report={"classical_otsu":availability("classical_otsu","OpenCV",None),"SeedGerm-VIG_U-Net_wheat_seed":availability("SeedGerm-VIG U-Net wheat seed","TensorFlow SavedModel",unet_path),"YOLOv8x-Germ":availability("YOLOv8x-Germ","Ultralytics",yolo_path),"SAM_2_1":availability("SAM 2.1","official sam2",None,"sam2 runtime/checkpoint not installed")}
    report["classical_otsu"].update({"available":True,"successfully_loaded":True,"inference_success":True,"notes":"baseline"})
    for rec in frames.itertuples(index=False):
        im=load_image(Path(rec.path)); start=time.perf_counter(); _,mask=create_foreground_mask(im,SegmentationConfig(method="otsu")); rows.append({"timestamp":rec.timestamp,"filename":rec.filename,"method":"classical_otsu","device":"cpu","runtime_seconds":time.perf_counter()-start,**mask_metrics(mask),"mask_count":np.nan,"detection_count":np.nan,"mean_confidence":np.nan,"median_confidence":np.nan,"notes":"dark seed-body baseline; components are not seeds"})
    try:
        from ultralytics import YOLO
        model=YOLO(str(yolo_path)); report["YOLOv8x-Germ"]["successfully_loaded"]=True; names=model.names
        for rec in frames.itertuples(index=False):
            start=time.perf_counter(); result=model.predict(str(rec.path),device=device_name(),verbose=False)[0]; elapsed=time.perf_counter()-start; boxes=result.boxes
            confidences=[] if boxes is None else boxes.conf.cpu().tolist()
            rows.append({"timestamp":rec.timestamp,"filename":rec.filename,"method":"YOLOv8x-Germ","device":device_name(),"runtime_seconds":elapsed,"foreground_area_pixels":np.nan,"foreground_percentage":np.nan,"component_count":np.nan,"mask_count":np.nan,"detection_count":len(confidences),"mean_confidence":np.mean(confidences) if confidences else np.nan,"median_confidence":np.median(confidences) if confidences else np.nan,"notes":"detections are not seed counts; class names from checkpoint metadata"})
            if boxes is not None: yolo_rows += [bbox_row(b,rec.timestamp,rec.filename,i,names,elapsed) for i,b in enumerate(boxes)]
        report["YOLOv8x-Germ"]["inference_success"]=True; report["YOLOv8x-Germ"]["notes"]=f"checkpoint class metadata: {names}"
    except Exception as exc: report["YOLOv8x-Germ"]["failure_reason"]=f"{type(exc).__name__}: {exc}"
    # U-Net model is kept explicit unavailable if legacy Keras loading fails; no guessed class mapping.
    try:
        import tensorflow as tf
        model=tf.keras.models.load_model(unet_path); report["SeedGerm-VIG_U-Net_wheat_seed"]["successfully_loaded"]=True; report["SeedGerm-VIG_U-Net_wheat_seed"]["notes"]="Official 512px letterbox/-1..1 pathway requires explicit run notebook implementation."
    except Exception as exc: report["SeedGerm-VIG_U-Net_wheat_seed"]["failure_reason"]=f"{type(exc).__name__}: {exc}"
    summary=pd.DataFrame(rows); summary.to_csv(tables/"model_benchmark_summary.csv",index=False); pd.DataFrame(yolo_rows).to_csv(tables/"yolo_detections.csv",index=False)
    with (tables/"model_availability.json").open("w") as f: json.dump(report,f,indent=2,default=str)
    recommendation={"ready_for_phase_3":False,"otsu_remains_useful":True,"sam_separated_touching_seeds":None,"unet_generalized":None,"yolo_detected_individual_seeds":None,"best_seed_body_representation":"classical_otsu_pending_manual_review","best_germination_representation":"Phase 2.5 combined yellow candidate","reason":"No ground truth; pretrained model outputs require visual validation and domain-shift assessment."}
    with (tables/"phase_2_7_recommendation.json").open("w") as f: json.dump(recommendation,f,indent=2)
    return summary,report
