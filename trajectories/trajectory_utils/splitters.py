import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class Splitter(ABC):
    def __init__(self, name: str) -> None:
        """Abstact base class for trajectory splitters

        Args:
            name (str): Name of splitter
        """
        super().__init__()
        self._name = name

    def split_exclusive(self,
                        times: np.ndarray,
                        coordinates: List[np.ndarray]) -> List[List[Tuple[int, int]]]:
        """Split the given exclusive trajectory

        Args:
            times (np.ndarray): Time stamps
            coordinates (List[np.ndarray]): Coordinates for every individual and time stamp

        Raises:
            NotImplemented: Error, if the splitting of exclusive trajectories is not supported

        Returns:
            List[List[Tuple[int, int]]]: List of splits as of tuples of start and end indices for every individual
        """
        raise NotImplementedError("Splitting of exclusive trajectories is not implemented for {}".format(self._name))

    def split_raw(self,
                  times: np.ndarray,
                  coordinates: List[List[np.ndarray]],
                  additional_data: List[List[List]],
                  names: List[str]) -> List[List[Tuple[int, int]]]:
        """Split the given raw trajectory

        Args:
            times (np.ndarray): Time stamps
            coordinates (List[List[np.ndarray]]): Raw coordinates for every time stamp
            additional_data (List[List[List]]): Additional data for every coordinate
            names (List[str]): Names of the individuals

        Raises:
            NotImplemented: Error, if the splitting of raw trajectories is not supported

        Returns:
            List[List[Tuple[int, int]]]: List of splits as of tuples of start and end indices for every individual
        """
        raise NotImplementedError("Splitting of raw trajectories is not implemented for {}".format(self._name))

    @property
    def name(self) -> str:
        """Get the name of this splitter

        Returns:
            str: Name of splitter
        """
        return self._name


class PatienceSplitter(Splitter):
    def __init__(self, 
                 coordinate_patience: int = 12,
                 time_patience: int = 1) -> None:
        """Splitter for splitting by coordinate and time gaps

        Args:
            coordinate_patience (int, optional): Patience for splitting by coordinates in number of frames. Defaults to 12.
            time_patience (int, optional): Patience for splitting by time gap in seconds. Defaults to 1.
        """
        super().__init__("PatienceTimeSplitter")
        self._coordinate_patience = coordinate_patience
        self._time_patience = time_patience

    def split_exclusive(self,
                        times: np.ndarray,
                        coordinates: List[np.ndarray]) -> List[List[Tuple[int, int]]]:
        # idxs for all individuals
        idxs_overall = []
        # iterate through idividuals
        for coords in coordinates:
            # idxs for this individual
            idxs = []
            # local coordinate patience
            c_patience = self._coordinate_patience
            # get first non negative index
            start_offset = np.argmax(coords[:, 0] > 0)
            start_idx = start_offset
            end_idx = start_offset
            # save time for first non negative index
            prev_time = times[start_idx]
            # iterate through all coordinates and time stamps
            for i, (c, t) in enumerate(zip(coords[start_idx:, 0], times[start_idx:]), start=start_offset):
                
                # last coordinate was longer ago, than time patience -> split here
                if t - prev_time > self._time_patience:
                    # see, if split already is known, if not append it
                    if (start_idx, end_idx) not in idxs and end_idx - start_idx > 0:
                        idxs.append((start_idx, end_idx))
                    # set patience as if last one was already added
                    c_patience = -1
                else:
                    # invalid coordinate -> lose patience
                    if c < 0:
                        c_patience -= 1
                    # valid coordinate -> gain full patience
                    else:
                        c_patience = self._coordinate_patience
                        end_idx = i + 1

                    # exactly no more patience -> split here
                    if c_patience == 0 and end_idx - start_idx > 0:
                        idxs.append((start_idx, end_idx))
                        start_idx = i + 1
                        end_idx = i + 1
                    # no more patience, but no new gap -> increase both start and end
                    elif c_patience < 0:
                        start_idx = i + 1
                        end_idx = i + 1
                        
                prev_time = t
            
            # see, if last step should be saved
            if end_idx - start_idx > 0:
                idxs.append((start_idx, end_idx))
            
            idxs_overall.append(idxs)
            
        return idxs_overall
                
                    
SPLITTER_MAP: Dict[str, Splitter] = {
    "patience": PatienceSplitter
}        
                    
                

