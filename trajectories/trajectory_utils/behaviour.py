import numpy as np
from trajectories.trajectory_utils.trajectory_pack import TrajectoryPack
from trajectories.trajectory_utils.feature_extractor import WindowedFFTFeatureExtractor, MixedFreqFeatureExtractor
from trajectories.trajectory_utils.splitters import Splitter
from typing import List, Tuple
from datetime import datetime as dt

def extract_dataset(traj: TrajectoryPack,
                    splitter: Splitter,
                    min_fraction: float = 0.0,
                    window_size: int = 120,
                    mixed_features: bool = True,
                    select_axis: bool = False,
                    crop_minute: bool = False,
                    overlap: Tuple[int, int] = (0, 0),
                    window: str = "boxcar",
                    detrend: str = "constant") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mixed_features:
        fe = MixedFreqFeatureExtractor(window=window, window_size=window_size, select_axis=select_axis, det=detrend, overlap=overlap)
    else:
        fe = WindowedFFTFeatureExtractor(window=window, window_size=window_size, select_axis=select_axis, det=detrend, overlap=overlap)
    samples_per_duration = int(12.5 * window_size)
    
    features = [[], []]
    classes = [[], []]
    fractions = [[], []]
    groups = [[], []]
    for j, t in enumerate(traj.trajectories):
        overall_idxs = splitter.split_exclusive(t.exclusive_times, t.exclusive)     

        for i, (c, idxs) in enumerate(zip(t.exclusive, overall_idxs)):
            
            for start, end in idxs:
                if crop_minute:
                    start_stamp = dt.fromtimestamp(t.exclusive_times[start])
                    # ignore 2s shifts
                    if start_stamp.second > 1 and start_stamp.second < 58:
                        start = int(start + (60 - start_stamp.second + start_stamp.microsecond / 1e6) * 12.5)
                # only get splits with 2 min of valid values
                if end - start < samples_per_duration:
                    continue
                section = c[start:end]
                f, mag, _ = fe.extract_features(section)
                
                for s in range(mag.shape[0]):
                    if len(t.behaviour[i][start + s*samples_per_duration: start + (s + 1) * samples_per_duration]) > 0:
                        unique, counter = np.unique(t.behaviour[i][start + s*samples_per_duration: start + (s + 1) * samples_per_duration], 
                                                    return_counts=True)
                        cls_idx = np.argmax(counter)
                        cls = unique[cls_idx]
                        fraction = counter[cls_idx] / sum(counter)
                        if cls != -1 and fraction > min_fraction:
                            fractions[i].append(fraction)
                            features[i].append(mag[s, :])
                            classes[i].append(cls)
                            groups[i].append(j)
    
    features = [*features[0], *features[1]]
    classes = [*classes[0], *classes[1]]
    fractions = [*fractions[0], *fractions[1]]
    groups = [*groups[0], *groups[1]]
    
    return np.array(features), np.array(classes), np.array(fractions), np.array(groups), np.array(f)
        

    
    
