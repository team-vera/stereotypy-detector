from abc import abstractmethod
from trajectories.trajectory_utils.trajectory import Trajectory
from trajectories.trajectory_utils.augmentor import Augmentor
from trajectories.trajectory_utils.splitters import Splitter
from trajectories.trajectory_utils.feature_extractor import FeatureExtractor, WindowedFFTFeatureExtractor
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from typing import Tuple

class BehaviourClassifier(Augmentor):
    def __init__(self, 
                 name: str,
                 splitter: Splitter,
                 feature_extractor: FeatureExtractor,
                 window_size: int = 60,
                 fps: float = 12.5,
                 stereo_only: bool = False) -> None:
        """Abstact base class for trajectory unmergers, which convert from exclusive to raw format

        Args:
            name (str): Name of unmerger
        """
        super().__init__(name)
        self._splitter = splitter
        self._fe = feature_extractor
        self._fps = fps
        self._window_size = window_size
        self._stereo_only = stereo_only
        self._samples_per_duration = int(self._fps * self._window_size)
        self._invalid = - np.ones(2)

    def apply_exclusive(self, t: Trajectory) -> None:

        t.init_behaviour(stereo_only=self._stereo_only)
        
        overall_idxs = self._splitter.split_exclusive(t.exclusive_times, t.exclusive)
        
        for i, (c, idxs) in enumerate(zip(t.exclusive, overall_idxs)):
            for start, end in idxs:
                if end - start < self._samples_per_duration:
                    continue

                section = c[start:end]
                f, mag, _ = self._fe.extract_features(section)

                behaviour_classes = self._classify_behaviour(mag, frequencies=f)

                for idx, b in enumerate(behaviour_classes):
                    t.behaviour[i][start + idx * self._samples_per_duration: start + (idx + 1) * self._samples_per_duration] = b

    def apply_both(self, t: Trajectory) -> None:
        raise NotImplementedError("{} does not support both trajectory types".format(self.name))

    def apply_raw(self, t: Trajectory) -> None:
        raise NotImplementedError("{} does not support raw trajectory types".format(self.name))

    @abstractmethod
    def _classify_behaviour(self, features: np.ndarray, frequencies: np.ndarray = None) -> np.ndarray:
        pass

class MultiClassBehaviourClassifier(BehaviourClassifier):
    def __init__(self, 
                 splitter: Splitter, 
                 feature_extractor: FeatureExtractor,
                 classifier_path: str, 
                 window_size: int = 60, 
                 fps: float = 12.5, 
                 stereo_only: bool = False) -> None:
        super().__init__("MultiClassBehaviourClassifier", splitter, feature_extractor, window_size, fps, stereo_only)
        with open(classifier_path, "rb") as f:
            self._clf: Pipeline = pickle.load(f)
            
    def _classify_behaviour(self, features: np.ndarray, frequencies: np.ndarray = None) -> np.ndarray:
        return self._clf.predict(features)

class HeuristicStereoClassifier(BehaviourClassifier):
    def __init__(self, 
                 splitter: Splitter,
                 window_size: int = 60, 
                 overlap: Tuple[int, int] = (0, 0),
                 fps: float = 12.5) -> None:
        super().__init__("MultiClassBehaviourClassifier", 
                         splitter, 
                         WindowedFFTFeatureExtractor(window_size=window_size,
                                                     fps=fps,
                                                     overlap=overlap), 
                         window_size, 
                         fps, 
                         True)
            
    def _classify_behaviour(self, features: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        stereo_mass = np.sum(features[:, np.logical_and(0.016 < frequencies, frequencies < 0.04)], axis=-1)
        full_mass = np.sum(features, axis=-1)
        mass_relation = stereo_mass / full_mass
        return (mass_relation > 0.3).astype(int)
        
        
