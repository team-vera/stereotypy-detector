import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
from trajectories.trajectory_utils.trajectory import Trajectory


class Merger(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory mergers

        Args:
            name (str): Name of merger
        """
        super().__init__()
        self._name = name
        self._invalid = - np.ones(2)

    @abstractmethod
    def merge(self, t: Trajectory) -> Trajectory:
        """Merge the given trajectory to an exclusive format

        Args:
            t (Trajectory): Trajectory to merge
            
        Returns:
            Trajectory: Merged trajectory
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of this merger

        Returns:
            str: Name of merger
        """
        return self._name


class StalinMerger(Merger):
    def __init__(self) -> None:
        """This is the Stalin merger (inspired by [Stalin sort](https://github.com/gustavo-depaula/stalin-sort)). 
        Every trajectory point with an invalid detection (i.e. two times the same individual detected) will be eliminated.
        No time stamp merging is performed.
        """
        super().__init__("StalinMerger")

    def merge(self, t: Trajectory) -> Trajectory:
        """Merge the given trajectory to an exclusive format

        Args:
            t (Trajectory): Trajectory to merge
        
        Returns:
            Trajectory: Merged trajectory
        """
        coord_list = [np.ndarray((len(t.raw_times), 2)) for _ in t.names]

        for i, c_list in enumerate(t.raw):
            for j, n in enumerate(t.names):
                c_for_name = [c for c, d in c_list if d[1] == n]
                if len(c_for_name) == 1:
                    coord_list[j][i] = c_for_name[0]
                else:
                    coord_list[j][i] = self._invalid

        t.exclusive_times = t.raw_times
        t.exclusive = coord_list
        return t


class DualGreedyMerger(Merger):
    def __init__(self) -> None:
        """The dual greedy merger will pick the detection with the highest detection probability in case of an individual being detected twice.
        """
        super().__init__("GreedyMerger")

    def merge(self, t: Trajectory) -> Trajectory:
        """Merge the given trajectory to an exclusive format

        Args:
            t (Trajectory): Trajectory to merge
            
        Returns:
            Trajectory: Merged trajectory
        """
        assert len(t.names) == 2, "This merger can only be applied to a two individual problem"

        coord_list = [np.ndarray((len(t.raw_times), 2)) for _ in t.names]
        for i, c_list in enumerate(t.raw):
            c_for_names = [[c for c in c_list if c[1][1] == n] for n in t.names]
            if len(c_for_names[0]) == 0 and len(c_for_names[1]) == 2:
                c_for_names[1].sort(key=lambda x: x[1][2], reverse=True)
                c_for_names[0].append(c_for_names[1].pop(1))
            elif len(c_for_names[1]) == 0 and len(c_for_names[0]) == 2:
                c_for_names[0].sort(key=lambda x: x[1][2], reverse=True)
                c_for_names[1].append(c_for_names[0].pop(1))

            for j, c_for_name in enumerate(c_for_names):
                c_for_name.sort(key=lambda x: x[1][0], reverse=True)
                if len(c_for_name) == 0:
                    coord_list[j][i] = self._invalid
                else:
                    coord_list[j][i] = c_for_name[0][0]

        t.exclusive_times = t.raw_times
        t.exclusive = coord_list
        
        return t


MERGER_MAP: Dict[str, Merger] = {
    "stalin": StalinMerger,
    "dual_greedy": DualGreedyMerger
}
