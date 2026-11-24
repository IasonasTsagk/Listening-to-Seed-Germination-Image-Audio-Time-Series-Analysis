"""Envelope-based candidate acoustic peak detection; never biological labels."""
import numpy as np
from scipy.signal import find_peaks, peak_widths
def detect_candidate_peaks(times, envelope, prominence, min_distance_seconds, width_seconds=None):
    hop=float(np.median(np.diff(times))) if len(times)>1 else 1.0; kwargs={"prominence":prominence,"distance":max(1,round(min_distance_seconds/hop))}
    if width_seconds is not None: kwargs["width"]=max(1,round(width_seconds/hop))
    indexes, props=find_peaks(envelope,**kwargs); widths=peak_widths(envelope,indexes,rel_height=.5)[0]*hop if len(indexes) else []
    return [{"relative_peak_time_seconds":float(times[i]),"envelope_peak_value":float(envelope[i]),"prominence":float(props["prominences"][j]),"width_seconds":float(widths[j]) if len(widths) else np.nan} for j,i in enumerate(indexes)]
