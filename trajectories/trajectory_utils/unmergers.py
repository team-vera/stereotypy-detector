import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from trajectories.trajectory_utils.trajectory import Trajectory


class UnMerger(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory unmergers, which convert from exclusive to raw format

        Args:
            name (str): Name of unmerger
        """
        super().__init__()
        self._name = name
        self._invalid = - np.ones(2)

    @abstractmethod
    def unmerge(self, t: Trajectory) -> None:
        """Unmerge the given trajectory from exclusive to raw format

        Args:
            t (Trajectory): Trajectory to unmerge
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of this unmerger

        Returns:
            str: Name of unmerger
        """
        return self._name

    def _coords_to_obs(self, coordinates: List[np.ndarray], names: List[str]) -> List[Tuple[np.ndarray, List[object]]]:
        out = []
        for c, n in zip(coordinates, names):
            if c[0] > 0:
                out.append((c, [1.0, n, 1.0]))
        return out


class DirectUnMerger(UnMerger):
    def __init__(self) -> None:
        """Unmerger, which directly converts exclusive to raw format without modification
        """
        super().__init__("DirectUnMerger")

    def unmerge(self, t: Trajectory) -> None:
        t.raw_times = t.exclusive_times
        raw_coords = []
        for coordinates in zip(*t.exclusive):
            raw_coords.append(self._coords_to_obs(coordinates, t.names))
        t.raw = raw_coords


class ErrorUnMerger(UnMerger):
    def __init__(self) -> None:
        super().__init__("ErrorUnMerger")


UNMERGER_MAP: Dict[str, UnMerger] = {
    "direct": DirectUnMerger
}
