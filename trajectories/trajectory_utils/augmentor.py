from abc import ABC, abstractmethod
from trajectories.trajectory_utils.trajectory import Trajectory


class Augmentor(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for augmentors

        Args:
            name (str): Name of augmentor
        """
        super().__init__()
        self._name = name

    @abstractmethod
    def apply_exclusive(self, t: Trajectory) -> None:
        """Apply this augmentor to the given exclusive trajectory

        Args:
            t (Trajectory): Trajectory to apply to
        """
        pass

    @abstractmethod
    def apply_raw(self, t: Trajectory) -> None:
        """Apply this augmentor to the given raw trajectory

        Args:
            t (Trajectory): Trajectory to ally to
        """
        pass

    @abstractmethod
    def apply_both(self, t: Trajectory) -> None:
        """Apply this augmentor to the given trajectory

        Args:
            t (Trajectory): Trajectory to ally to
        """
        pass
    

    @property
    def name(self) -> str:
        """Get the name of this augmentor

        Returns:
            str: Name of augmentor
        """
        return self._name
