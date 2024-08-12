import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from utilities.enums import MAX_SPEED_PB_PIXELS
from trajectories.trajectory_utils.trajectory import Trajectory
from trajectories.trajectory_utils.augmentor import Augmentor


class SanityChecker(Augmentor):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory sanity checkers

        Args:
            name (str): Name of merger
        """
        super().__init__(name)

    @abstractmethod
    def apply_exclusive(self, t: Trajectory) -> None:        
        pass
    
    def apply_raw(self, t: Trajectory) -> None:
        raise NotImplementedError("Raw augmentation is not supproted for {}".format(self.name))



class SwapSanityChecker(SanityChecker):
    def __init__(self) -> None:
        """Sanity checker, that swaps coordinates, if both are available, but do not fit.
        Algorithm: If both coordinates available, but do not pass the basic movement sanity check -> \
            check, if swapping would pass the check -> if yes, swap
        """
        super().__init__("SwapSanityChecker")

    def apply_exclusive(self, t: Trajectory) -> None:
        """Perform the swap sanity check to an exclusive two individual trajectory

        Args:
            t (Trajectory): Exclisuve trajectory to process
        """
        assert len(t.names) == 2, "Only a two individual trajectory is possible for {}".format(self._name)
        prev_time = t.exclusive_times[0]
        prev_coords_0 = t.exclusive[0][0]
        prev_coords_1 = t.exclusive[1][0]
        for i, t_now in enumerate(t.exclusive_times[1:]):
            coords_0 = t.exclusive[0][i + 1]
            coords_1 = t.exclusive[1][i + 1]
            t_delta = t_now - prev_time
            if prev_coords_0[0] < 0 or prev_coords_1[1] < 0:
                prev_coords_0 = coords_0
                prev_coords_1 = coords_1
            elif np.linalg.norm(prev_coords_0 - coords_0) / t_delta > MAX_SPEED_PB_PIXELS and \
                    np.linalg.norm(prev_coords_1 - coords_1) / t_delta > MAX_SPEED_PB_PIXELS and \
                    np.linalg.norm(prev_coords_1 - coords_0) / t_delta < MAX_SPEED_PB_PIXELS and \
                    np.linalg.norm(prev_coords_0 - coords_1) < MAX_SPEED_PB_PIXELS:
                t.exclusive[0][i + 1], t.exclusive[1][i + 1] = t.exclusive[1][i + 1], t.exclusive[0][i + 1]
                prev_coords_1 = coords_0
                prev_coords_0 = coords_1
            prev_time = t_now


SANITY_CHECKER_MAP: Dict[str, SanityChecker] = {
    "swap": SwapSanityChecker
}