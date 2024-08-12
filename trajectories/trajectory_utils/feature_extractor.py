import numpy as np
from scipy.signal import stft, detrend, get_window
import scipy.fft
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union, List
import matplotlib.pyplot as plt


class FeatureExtractor(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory feature extractors

        Args:
            name (str): Name of feature extractor
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def extract_features(self,
                         trajectory_part: np.ndarray) -> Union[Tuple[np.ndarray,
                                                                     np.ndarray],
                                                               Tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     np.ndarray]]:
        """Extract features from the given trajectory part

        Args:
            trajectory_part (np.ndarray): Trajectory part to extract features for

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: Label for feature and extracted features or Label for features, extracted features and time stamps
        """
        pass
    
    @abstractmethod
    def normalize(self,
                  features: np.ndarray) -> np.ndarray:
        """Normalize the extracted features to be in between 0 and 1

        Args:
            features (np.ndarray): Features to normalize

        Returns:
            np.ndarray: Normalized features
        """

    @property
    def name(self) -> str:
        """Get the name of this feature extractor

        Returns:
            str: Name of feature extractor
        """
        return self._name


class FFTFeatureExtractor(FeatureExtractor):
    def __init__(self, fps: float) -> None:
        self._sample_spacing = 1 / fps
        super().__init__("FFTFeatureExtractor")

    def extract_features(self, trajectory_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features = np.abs(np.fft.fft(trajectory_part - np.mean(trajectory_part, axis=0), axis=0))
        return np.fft.fftfreq(features.shape[0], d=self._sample_spacing), features


class STFTFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 window: str = "flattop",
                 window_size: float = 120,
                 fps: float = 12.5,
                 select_axis: bool = False) -> None:
        super().__init__("STFTFeatureExtractor")
        self._window = window
        self._fps = fps
        self._nperseg = int(window_size * self._fps)
        self._select_axis = select_axis

    def extract_features(self,
                         trajectory_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract STFT features

        Args:
            trajectory_part (np.ndarray): Trajectory part to analyze

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, features in format (num_windows, num_features), time stamps of windows
        """
        f, t, Zxy = stft(trajectory_part,
                        self._fps,
                        nperseg=self._nperseg,
                        padded=False,
                        boundary="odd",
                        detrend="constant",
                        window=self._window,
                        axis=0)

        if self._select_axis:
            traj_to_use = np.argmax(np.std(trajectory_part, axis=0))
            mag = np.abs(Zxy[:, traj_to_use])
        else:
            mag = np.sqrt(np.sum(np.abs(Zxy) ** 2, axis=1))

        mag = np.reshape(mag, (mag.shape[1], mag.shape[0]))

        return f, mag, t
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        return features / np.sum(features, axis=1)[..., np.newaxis]
    
class WindowedFFTFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 window: str = "boxcar",
                 window_size: float = 60,
                 fps: float = 12.5,
                 det: str = "constant",
                 select_axis: bool = False,
                 include_rest: bool = False,
                 overlap: Tuple[int, int] = (0, 0)) -> None:
        super().__init__("STFTFeatureExtractor")
        self._window = window
        self._fps = fps
        self._window_size = window_size
        self._nperseg = int(window_size * self._fps)
        self._include_rest = include_rest
        self._select_axis = select_axis
        self._detrend = det
        # overlap is an extension of the window into past and future respectively (in s) 
        self._overlap = overlap
        self._noverlap = (int(self._overlap[0] * self._fps), 
                          int(self._overlap[1] * self._fps))

    def extract_features(self,
                         trajectory_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract windowed FFT features

        Args:
            trajectory_part (np.ndarray): Trajectory part to analyze

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequencies, features in format (num_windows, num_features), time stamps of windows
        """
        if self._include_rest:
            num_segments = int(np.ceil(trajectory_part.shape[0] / self._nperseg))
        else:
            num_segments = int(trajectory_part.shape[0] // self._nperseg)
            
        num_freqs = self._nperseg + sum(self._noverlap)

        w = get_window(self._window, self._nperseg + sum(self._noverlap))[:, np.newaxis]
        
        frequencies = scipy.fft.fftfreq(num_freqs, 1 / self._fps)[:num_freqs // 2]
        mag = np.zeros((num_segments, num_freqs // 2), dtype=np.float64)
        times = np.arange(num_segments) * self._window_size
        
        for i in range(num_segments):
            start = max(0, i * self._nperseg - self._noverlap[0])
            end = (i + 1) * self._nperseg + self._noverlap[1]

            # detrend / mean subtraction
            if self._detrend is not None:
                det_t = detrend(trajectory_part[start:end], axis=0, type=self._detrend)
            else:
                det_t = trajectory_part[start:end]

            # apply window
            if det_t.shape[0] != w.shape[0]:
                det_t *= get_window(self._window, det_t.shape[0])[:, np.newaxis]
            else:
                det_t *= w
                
            # compute fft
            f = scipy.fft.fft(det_t, n=num_freqs, axis=0)

            # compute magnitude
            if self._select_axis:
                traj_to_use = np.argmax(np.std(det_t, axis=0))
                f = np.abs(f[:, traj_to_use])
            else:
                f = np.sqrt(np.sum(np.abs(f) ** 2, axis=1))
                
            mag[i, :len(f) // 2] = f[:len(f) // 2]

        return frequencies, mag, times
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        return features / np.sum(features, axis=1)[..., np.newaxis]
    
class MixedFreqFeatureExtractor(WindowedFFTFeatureExtractor):
    def __init__(self, 
                 window: str = "boxcar", 
                 window_size: float = 60, 
                 fps: float = 12.5, 
                 det: str = "constant",
                 select_axis: bool = False,
                 include_rest: bool = False,
                 overlap: Tuple[int, int] = (0, 0)) -> None:
        super().__init__(window, window_size, fps, det, select_axis, include_rest, overlap=overlap)
        
    def extract_features(self, 
                         trajectory_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nf = []
        f_names = []
        # ONLY GET WINDOWS FROM HERE
        f, mag, t = super().extract_features(trajectory_part)
        
        # raw trajectory space features
        std_detrended_x = []
        std_detrended_y = []
        mean_detrended_x = []
        mean_detrended_y = []
        momentum_detrended_x = []
        momentum_detrended_y = []
        for t_ in t:
            start = int(t_ * self._fps)
            end = int(start + self._nperseg / 2)
            to_detrend = trajectory_part[start:end]
            if len(to_detrend) > 1:
                detrended_tp = detrend(to_detrend, axis=0, type="linear")
                std_detrended_x.append(np.std(detrended_tp[:, 0]))
                std_detrended_y.append(np.std(detrended_tp[:, 1]))
                mean_detrended_x.append(np.abs(np.mean(detrended_tp[:, 0])))
                mean_detrended_y.append(np.abs(np.mean(detrended_tp[:, 1])))
                # mom_sum = np.sum(detrended_tp, axis=0)
                # mom_sum[np.where(mom_sum == 0.0)] = np.finfo(float).min
                # momentum = np.sum(detrended_tp * np.arange(detrended_tp.shape[0])[:, np.newaxis], axis=0) / mom_sum
                # momentum_detrended_x.append(momentum[0])
                # momentum_detrended_y.append(momentum[1])
            else:
                std_detrended_x.append(0)
                std_detrended_y.append(0)
                mean_detrended_x.append(0)
                mean_detrended_y.append(0)
                # momentum_detrended_x.append(0)
                # momentum_detrended_y.append(0)
        

        nf.append(std_detrended_x)
        nf.append(std_detrended_y)
        # nf.append(mean_detrended_x)
        # nf.append(mean_detrended_y)
        # nf.append(momentum_detrended_x)
        # nf.append(momentum_detrended_y)

        f_names.append("std_detrended_x")
        f_names.append("std_detrended_y")
        # f_names.append("abs_mean_detrended_x")
        # f_names.append("abs_mean_detrended_y")
        # f_names.append("momentum_detrended_x")
        # f_names.append("momentum_detrended_y")
        
        # frequency based features
        
        # unnormalized fequancy features
        nf.append(np.max(mag, axis=1))
        nf.append(np.mean(mag, axis=1))
        nf.append(np.std(mag, axis=1))
        nf.append(np.argmax(mag, axis=1))
        
        f_names.append("freq_max")
        f_names.append("freq_mean")
        f_names.append("freq_std")
        f_names.append("freq_argmax")
        
        
        # normalized fequency features
        mag_norm = super().normalize(mag)
        
        nf.append(np.max(mag_norm, axis=1))
        # nf.append(np.sum(mag_norm[:] * np.arange(mag_norm.shape[1]), axis=1) / np.sum(mag_norm, axis=1))
        
        f_names.append("freq_norm_max")
        # f_names.append("freq_norm_momentum")
        
        features = np.log(mag_norm)
    
        features /= np.max(features, axis=1)[..., np.newaxis]
        
        nf.append(np.mean(features, axis=1))
        nf.append(np.std(features, axis=1))
        nf.append(np.median(features, axis=1))
        #nf.append(np.sum(features[:] * np.arange(features.shape[1]), axis=1) / np.sum(features, axis=1))
        
        f_names.append("freq_norm_log_mean")
        f_names.append("freq_norm_log_std")
        f_names.append("freq_norm_log_median")
        # f_names.append("freq_norm_log_momentum")
        
        # stereo_mass = np.sum(mag[:, np.logical_and(0.016 < f, f < 0.04)], axis=-1)
        # full_mass = np.sum(mag, axis=-1)
        # mass_relation = stereo_mass / full_mass
        # nf.append(mass_relation)
        # 
        # f_names.append("freq_mass_rel")
        
        return f_names, np.stack(nf, axis=1), t
        
        


FEATURE_EXTRACTOR_MAP: Dict[str, FeatureExtractor] = {
    "fft": FFTFeatureExtractor,
    "stft": STFTFeatureExtractor,
    "wfft": WindowedFFTFeatureExtractor
}
