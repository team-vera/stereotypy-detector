import numpy as np
from typing import List, Union
from trajectories.trajectory_utils.mergers import Merger
from trajectories.trajectory_utils.splitters import Splitter
from trajectories.trajectory_utils.sanity_checker import SanityChecker
from trajectories.trajectory_utils.trajectory import Trajectory
from trajectories.trajectory_utils.fuser import Fuser
from trajectories.trajectory_utils.feature_extractor import FeatureExtractor
from trajectories.trajectory_utils.unmergers import UnMerger
from trajectories.trajectory_utils.augmentor import Augmentor
import logging
from threading import Lock
import time
import multiprocessing.pool
from multiprocessing import Pool


class TrajectoryPack():
    # raw: JSON format with multiple occurances of one individual possible + additional data like probabilities provided
    # exclusive: CSV format with every individual exclusively once and -1 for ivalid detections
    FORMATS = ["raw", "exclusive"]
    COLOR_CYCLE = ["C9", "C1", "C0", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
    __INSTANCE_COUNTER = 0
    __COUNTER_LOCK = Lock()

    def __init__(self,
                 traj: Union[List[str], str],
                 names: Union[List[List[str]], List[str]],
                 overlay_shape: np.ndarray,
                 format: str = "exclusive") -> None:
        """Class for properly handling trajectories with multiple individuals

        Args:
            traj (Union[List[str], str]): List of paths to trajectories or one path to a single trajectory
            names (Union[List[List[str]], List[str]]): List of names for every trajectory or one list for all trajectories
            overlay_shape (np.ndarray): Shape of the overlay to properly convert to pixel coordiantes
            format (str, optional): Format to load and process trajectories in. ["raw", "exclusive"] possible. Defaults to "exclusive".\
                "raw": JSON format with multiple occurances of one individual possible + additional data like probabilities provided.\
                    "exclusive": CSV format with every individual exclusively once and -1 for ivalid detections

        Raises:
            TypeError: Error, if trajectories are not provided in a proper way
        """
        # assert properly
        assert format in self.FORMATS, "Format has to be in {}. Got {}".format(
            self.FORMATS, format)
        assert len(overlay_shape) == 2, "Invalid length of overlay shape ({}). Lenght 2 expected".format(
            overlay_shape)

        # internal memory for trajectories of any kind
        self._trajectories: List[Trajectory] = []
        # invert for proper x-y handling!
        self._overlay_shape = overlay_shape[::-1]

        with self.__COUNTER_LOCK:
            self._logger = logging.getLogger("TrajectoryPack{}".format(self.__INSTANCE_COUNTER))
            self.__INSTANCE_COUNTER += 1

        self.load_traj(traj, names, format)

    def load_traj(self,
                  traj: Union[List[str], str],
                  names: Union[List[List[str]], List[str]],
                  format: str = "exclusive",
                  overwrite: bool = False) -> None:
        """Class for properly handling trajectories with multiple individuals

        Args:
            traj (Union[List[str], str]): List of paths to trajectories or one path to a single trajectory
            names (Union[List[List[str]], List[str]]): List of names for every trajectory or one list for all trajectories
            format (str, optional): Format to load and process trajectories in. ["raw", "exclusive"] possible. Defaults to "exclusive".\
                "raw": JSON format with multiple occurances of one individual possible + additional data like probabilities provided.\
                    "exclusive": CSV format with every individual exclusively once and -1 for ivalid detections
            overwrite (bool, optional): If True, overwrite already loaded trajectories

        Raises:
            TypeError: Error, if trajectories are not provided in a proper way
        """
        assert format in self.FORMATS, "Format has to be in {}. Got {}".format(
            self.FORMATS, format)

        if overwrite:
            self._trajectories.clear()
        # got multiple trajectories
        if isinstance(traj, list):
            # check for proper name format and extend, if necessary
            if isinstance(names[0], str):
                names = [names for _ in range(len(traj))]
            else:
                assert len(names) == len(traj), \
                    "Number of trajectories ({}) and number of names ({}) do not match.".format(len(traj), len(names))
            # load every trajectory
            for t, n in zip(traj, names):
                trajectory = Trajectory(name_list=n)
                trajectory.load_traj(t, n, format, self._overlay_shape)
                self.trajectories.append(trajectory)
                self._logger.info("Loaded {} from {}".format(trajectory, t))
        # single trajectory given
        elif isinstance(traj, str):
            assert isinstance(
                names[0], str), "A one dimensional list of names is required for a single trajectory. \
                Got {}.".format(names)
            trajectory = Trajectory(name_list=names)
            trajectory.load_traj(traj, names, format, self._overlay_shape)
            self.trajectories.append(trajectory)
            self._logger.info("Loaded {} from {}".format(trajectory, traj))
        # invalid trajectory format
        else:
            raise TypeError(
                "Argument traj is of inappropriate type {}. Must be either a list or str".format(
                    type(traj)))
            
    def save_traj(self,
                  traj_paths: List[str],
                  format: str,
                  save_behaviour: bool = True):
        assert len(self._trajectories) == len(traj_paths), \
            "Number of trajectories must equal the nubmer of trajectory paths. Got {} expected {}.".format(
                len(traj_paths),
                len(self._trajectories)
            )
        for t, traj_path in zip(self._trajectories, traj_paths):
            t.save_traj(traj_path, format, self._overlay_shape, save_behaviour)
            self._logger.info("Saved {} to {} in {} format".format(t, traj_path, format))

    def to_exclusive(self, merger: Merger, num_threads: int = 1, use_process: bool = False) -> None:
        """Convert the trajectory to exclusive format

        Args:
            merger (Merger): Merger to use for merging
            num_threads (int, optional): Maximum number of threads to use for parallel processing. \
                If < 2 no parallel processing will be used. Consider the thread safety of the merger, if parallel processing is desired.\
                    Defaults to 1.
            use_process (bool, optional): Force the usage of a process instead of thread. Defaults to False.

        Raises:
            NotImplemented: Error, if conversion to exclusive is not supported for internal format
        """
        start_time = time.time()
        if num_threads < 2 or len(self._trajectories) < 2:
            list(map(merger.merge, self._trajectories))
        else:
            if use_process:
                with Pool(min(num_threads, len(self._trajectories))) as p:
                    self._trajectories = list(p.map(merger.merge, self._trajectories))
            else:
                with multiprocessing.pool.ThreadPool(min(num_threads, len(self._trajectories))) as p:
                    list(p.map(merger.merge, self._trajectories))
        self._logger.info("{} trajectories merged by {} in {:.3f}s".format(
            len(self._trajectories), merger.name, time.time() - start_time))
        
    def to_raw(self, unmerger: UnMerger, num_threads: int = 1) -> None:
        """Convert the trajectory to raw format

        Args:
            unmerger (UnMerger): Unerger to use for conversion
            num_threads (int, optional): Maximum number of threads to use for parallel processing. \
                If < 2 no parallel processing will be used. Consider the thread safety of the unmerger, if parallel processing is desired.\
                    Defaults to 1.

        Raises:
            NotImplemented: Error, if conversion to raw is not supported for internal format
        """
        start_time = time.time()
        if num_threads < 2 or len(self._trajectories) < 2:
            list(map(unmerger.unmerge, self._trajectories))
        else:
            with multiprocessing.pool.ThreadPool(min(num_threads, len(self._trajectories))) as p:
                list(p.map(unmerger.unmerge, self._trajectories))
        self._logger.info("{} trajectories unmerged by {} in {:.3f}s".format(
            len(self._trajectories), unmerger.name, time.time() - start_time))

    def apply_to_exclusive(self, to_apply: Augmentor, num_threads: int = 1) -> None:
        """Apply operation to the internal exclusive trajectory

        Args:
            to_apply (Augmentor): Operation to ally
            num_threads (int, optional): Maximum number of threads to use for parallel processing. \
                If < 2 no parallel processing will be used. Consider the thread safety of the merger, if parallel processing is desired.\
                    Defaults to 1.
        """
        start_time = time.time()
        if num_threads < 2 or len(self._trajectories) < 2:
            list(map(to_apply.apply_exclusive, self._trajectories))
        with multiprocessing.pool.ThreadPool(min(num_threads, len(self._trajectories))) as p:
            list(p.map(to_apply.apply_exclusive, self._trajectories))
        self._logger.info("Applied {} to {} trajectories in {:.3f}s".format(
            to_apply.name, len(self._trajectories), time.time() - start_time))

    def apply_to_raw(self, to_apply: Augmentor, num_threads: int = 1) -> None:
        """Apply operation to the internal raw trajectory

        Args:
            to_apply (Augmentor): Operation to ally
            num_threads (int, optional): Maximum number of threads to use for parallel processing. \
                If < 2 no parallel processing will be used. Consider the thread safety of the merger, if parallel processing is desired.\
                    Defaults to 1.
        """
        start_time = time.time()
        if num_threads < 2 or len(self._trajectories) < 2:
            list(map(to_apply.apply_raw, self._trajectories))
        with multiprocessing.pool.ThreadPool(min(num_threads, len(self._trajectories))) as p:
            list(p.map(to_apply.apply_raw, self._trajectories))
        self._logger.info("Applied {} to {} trajectories in {:.3f}s".format(
            to_apply.name, len(self._trajectories) , time.time() - start_time))
        
    def apply_to_both(self, to_apply: Augmentor, num_threads: int = 1) -> None:
        """Apply operation to the internal trajectory

        Args:
            to_apply (Augmentor): Operation to ally
            num_threads (int, optional): Maximum number of threads to use for parallel processing. \
                If < 2 no parallel processing will be used. Consider the thread safety of the merger, if parallel processing is desired.\
                    Defaults to 1.
        """
        start_time = time.time()
        if num_threads < 2 or len(self._trajectories) < 2:
            list(map(to_apply.apply_both, self._trajectories))
        with multiprocessing.pool.ThreadPool(min(num_threads, len(self._trajectories))) as p:
            list(p.map(to_apply.apply_both, self._trajectories))
        self._logger.info("Applied {} to {} trajectories in {:.3f}s".format(
            to_apply.name, len(self._trajectories) , time.time() - start_time))

    @property
    def trajectories(self) -> List[Trajectory]:
        """Get the current internal trajectories

        Returns:
            List[Trajectory]: Trajectories in raw or exclusive format
        """
        return self._trajectories

    def crop(self, start_time: float = 0.0, end_time: float = None, relative: bool = True) -> None:
        """Crop all trajectories to the given interval

        Args:
            start_time (float, optional): Start of the interval. Defaults to 0.0.
            end_time (float, optional): End of the interval. If None, the end of the interval is the end of each trajectory. Defaults to None.
            relative (bool, optional): If True, the start and end are considered to be seconds from the start of each trajectory. \
                If False, the absolute time stamps are assumed. Defaults to True.
        """
        to_remove = []
        for i in range(len(self._trajectories)):
            if not self._trajectories[i].crop(start_time, end_time, relative):
                to_remove.append(i)

        for i, j in enumerate(sorted(to_remove)):
            self._trajectories.pop(j - i)

    def resample(self, n: int) -> None:
        """Resample the trajectory pack

        Args:
            n (int): Resample to only every n-th sample
        """
        for t in self._trajectories:
            t.resample(n)

    def fuse(self, fuser: Fuser, overwrite: bool = True) -> None:
        """Fuse multiple raw trajectories into one

        Args:
            fuser (Fuser): Fuser to use for fusing raw trajectories
            overwrite (bool, optional): If True, overwrite the trajectories to fuse. Defaults to True.
        """
        fused = fuser.fuse(self._trajectories)
        self._logger.info("Fused {} trajectories by {} into {}".format(len(self._trajectories),
                                                                       fuser.name,
                                                                       fused))
        if overwrite:
            self._trajectories.clear()
        self._trajectories.append(fused)

    def __str__(self) -> str:
        s = "{}:".format(self._logger.name)
        for i, t in enumerate(self._trajectories):
            s += "\nTrajectory {}: {}".format(i, t)
        return s
