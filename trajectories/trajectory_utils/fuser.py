import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict
from trajectories.trajectory_utils.trajectory import Trajectory


class Fuser(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory fusers

        Args:
            name (str): Name of fuser
        """
        super().__init__()
        self._name = name
        self._invalid = - np.ones(2)

    @abstractmethod
    def fuse(self, trajectories: List[Trajectory]) -> Trajectory:
        """Fuse the given list of trajectories

        Args:
            t (Trajectory): Trajectories to fuse

        Returns:
            Trajectory: Trajectory with fused coordinates
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of this fuser

        Returns:
            str: Name of fuser
        """
        return self._name


class RawFuser(Fuser):
    def __init__(self, fps: float) -> None:
        """Fuser for raw trajectories

        Args:
            fps (float): Frames per second
        """
        self._fps = fps
        self._sampling_rate = 1 / fps
        self._half_sr = self._sampling_rate / 2
        super().__init__("RawFuser")

    def fuse(self, trajectories: List[Trajectory]) -> Trajectory:
        """Fuse the given list of raw trajectories

        Args:
            trajectories (List[Trajectory]): Trajectories to fuse

        Returns:
            Trajectory: Fused raw treajectory
        """
        assert len(trajectories) > 0, "Got an empty list of trajectories"
        assert all([tuple(trajectories[0].names) == tuple(t.names) for t in trajectories]), \
            "All trajectories must have the same subjects"
        fused = Trajectory(trajectories[0].names)
        trajectories = sorted(trajectories, key=lambda x: x.raw_times[0])
        counters = [0 for _ in trajectories]
        new_times = []
        new_raw = []
        start_time = trajectories[0].raw_times[0]
        end_time = max([t.raw_times[-1] for t in trajectories])
        current_time = start_time
        to_delete = []
        to_add = []
        while current_time <= end_time:
            for i, (c, t) in enumerate(zip(counters, trajectories)):
                if t.raw_times[c] - current_time <= self._half_sr:
                    to_add.extend(t.raw[c])
                    counters[i] += 1
                    if counters[i] >= len(t.raw):
                        to_delete.append(i)

            if len(to_add) > 0:
                new_raw.append(to_add.copy())
                new_times.append(current_time)
                to_add.clear()

            if len(to_delete) > 0:
                for i, j in enumerate(to_delete):
                    trajectories.pop(j - i)
                    counters.pop(j - i)
                to_delete.clear()

            current_time += self._sampling_rate

        fused.raw = new_raw
        fused.raw_times = np.array(new_times)

        return fused

FUSER_MAP: Dict[str, Fuser] = {
    "raw": RawFuser
}