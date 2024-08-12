import numpy as np
from typing import List, Set, Tuple, Union, NamedTuple, Dict
import json
import pandas as pd
import copy
import os
from datetime import datetime
from utilities.enums import MPP


class PBStats():
    def __init__(self,
                 length: float,
                 oos: float,
                 behaviour_duration: float = None,
                 behaviour: Dict[str, float] = None) -> None:
        self.length: float = length
        self.oos: float = oos
        self.behaviour_duration: float = behaviour_duration
        self.behaviour: Dict[str, float] = behaviour
        self.__absolute: bool = False

    def to_absolute(self, seconds: float):
        if self.__absolute:
            return
        self.oos *= seconds
        if self.behaviour_duration is not None:
            self.behaviour_duration *= seconds
        if self.behaviour is not None:
            self.behaviour = {k: v * seconds for k, v in self.behaviour.items()}
        self.__absolute = True

    def __str__(self) -> str:
        ret = "Length: {:.2f}m ".format(self.length)
        if self.__absolute:
            ret += " | OOS: {:.2f}s".format(self.oos)
            if self.behaviour_duration is not None:
                ret += " | BD: {:.2f}s".format(self.behaviour_duration)
            if self.behaviour is not None:
                for k, v in self.behaviour.items():
                    ret += " | {}: {:.2f}s".format(k, v)
        else:
            ret += " | OOS: {:.4f}".format(self.oos)
            if self.behaviour_duration is not None:
                ret += " | BD: {:.4f}".format(self.behaviour_duration)
            if self.behaviour is not None:
                for k, v in self.behaviour.items():
                    ret += " | {}: {:.4f}".format(k, v)
        return ret
    
    def to_tex(self) -> Tuple[str, str]:
        header = ""
        ret = ""
        if self.__absolute:
            ret += "{:.2f}".format(self.oos)
            header += "OOS".format(self.oos)
            if self.behaviour_duration is not None:
                ret += " & {:.2f}".format(self.behaviour_duration)
                header += " & BD".format(self.behaviour_duration)
            if self.behaviour is not None:
                for k, v in self.behaviour.items():
                    ret += " & {:.2f}".format(v)
                    header += " & {}".format(k)
        else:
            ret += "{:.3f}".format(self.oos)
            header += "OOS".format(self.oos)
            if self.behaviour_duration is not None:
                ret += " & {:.3f}".format(self.behaviour_duration)
                header += " & BD".format(self.behaviour_duration)
            if self.behaviour is not None:
                for k, v in self.behaviour.items():
                    ret += " & {:.3f}".format(v)
                    header += " & {}".format(k)
        return header, ret


class TrajectoryStats(NamedTuple):
    nanuq: PBStats
    vera: PBStats
    seconds: float

    def to_absolute(self):
        self.vera.to_absolute(self.seconds)
        self.nanuq.to_absolute(self.seconds)

    def correct_for_oos(self):
        self.vera.correct_for_oos()
        self.nanuq.correct_for_oos()

    def __str__(self) -> str:
        return "Nanuq: ({}) | Vera: ({}) | Duration: {:.2f}s".format(self.nanuq, self.vera, self.seconds)
    
    def to_tex(self) -> Tuple[str, str]:
        hn, rn = self.nanuq.to_tex()
        hv, rv = self.vera.to_tex()
        header = hn + " & " + hv + " & Duration\\\\"
        ret = rn + " & " + rv + " & {:.0f}\\\\".format(self.seconds)
        return header, ret


class Trajectory():
    __FORMATS = ["raw", "exclusive"]

    def __init__(self,
                 name_list: List[str],
                 raw: List[List[Tuple[np.ndarray, List[object]]]] = None,
                 raw_times: np.ndarray = None,
                 exclusive: List[np.ndarray] = None,
                 exclusive_times: np.ndarray = None) -> None:
        # 1-D array of timestamps
        self._exclusive_times: np.ndarray = None
        self._raw_times: np.ndarray = None
        # list with one 2-D array of shape [timestamp, 2] for every subject
        self._exclusive: List[np.ndarray] = None
        # list with one List of coordiantes and additional data for each time stamp
        self._raw: List[List[Tuple[np.ndarray, List[object]]]] = None
        # list of names for each subject
        self._names: List[str] = name_list
        # formats available
        self._formats: Set[str] = set()
        # raw trajectory given
        if raw is not None or raw_times is not None:
            assert raw is not None and raw_times is not None, "Both raw coordiantes and time stamps have to be given"
            assert name_list is not None, "A list of subjects has to be given"
            self._raw = raw,
            self._raw_times = raw_times
        # exclusive trajectory given
        if exclusive is not None or exclusive_times is not None:
            assert exclusive is not None and exclusive_times is not None, "Both exclusive coordiantes and time stamps have to be given"
            assert name_list is not None, "A list of subjects has to be given"
            self._exclusive = exclusive,
            self._exclusive_times = exclusive_times

        # behaviour for every time stamp
        self._behaviour: List[np.ndarray] = None
        self._behaviour_classes: List[str] = None

    @property
    def exclusive(self) -> List[np.ndarray]:
        """Exclusive format of trajectory

        Raises:
            AttributeError: Error, if no exclusive format is available

        Returns:
            List[np.ndarray]: Trajectory in exclusive format. A list containing an 2-D array of shape [time stamps, 2] for each subject.
        """
        if self._exclusive is None:
            raise AttributeError("No exclusive format of trajectory available")
        return self._exclusive

    @exclusive.setter
    def exclusive(self, exclusive: List[np.ndarray]) -> None:
        self._formats.add("exclusive")
        self._exclusive = exclusive

    @property
    def exclusive_times(self) -> np.ndarray:
        """Time stamps for the trajectory in exclusive format

        Raises:
            AttributeError: Error, if no exclusive format is available

        Returns:
            np.ndarray: Time stampos of trajectory in exclusive format.
        """
        if self._exclusive_times is None:
            raise AttributeError("No exclusive format of trajectory available")
        return self._exclusive_times

    @exclusive_times.setter
    def exclusive_times(self, exclusive_times: np.ndarray) -> None:
        self._exclusive_times = exclusive_times

    @property
    def raw(self) -> List[List[Tuple[np.ndarray, List[object]]]]:
        """Raw format of trajectory

        Raises:
            AttributeError: Error, if no raw format is available

        Returns:
            List[List[Tuple[np.ndarray, List[object]]]]: Trajectory in raw format.\
                A list containing a list of coordiantes and additional data for each time stamp.
        """
        if self._raw is None:
            raise AttributeError("No raw format of trajectory available")
        return self._raw

    @raw.setter
    def raw(self, raw: List[List[Tuple[np.ndarray, List[object]]]]) -> None:
        self._formats.add("raw")
        self._raw = raw

    @property
    def raw_times(self) -> np.ndarray:
        """Time stamps for trajectory in raw format

        Raises:
            AttributeError: Error, if no raw format is available

        Returns:
            np.ndarray: Time stamps of trajectory in raw format.
        """
        if self._raw_times is None:
            raise AttributeError("No raw format of trajectory available")
        return self._raw_times

    @raw_times.setter
    def raw_times(self, raw_times: np.ndarray) -> None:
        self._raw_times = raw_times

    @property
    def formats(self) -> Set[str]:
        """Formats available for this trajectory

        Returns:
            Set[str]: Set of formats available for this trajectory.
        """
        return self._formats

    @property
    def names(self) -> List[str]:
        """List of all subject names

        Returns:
            List[str]: List of all subject names
        """
        return self._names

    @property
    def behaviour_classes(self) -> List[str]:
        """List of all behaviours classes

        Returns:
            List[str]: List of all behaviour classes
        """
        return self._behaviour_classes

    @property
    def behaviour(self) -> List[np.ndarray]:
        """Behaviour at every time stamp

        Raises:
            AttributeError: Error, if no behaviour is available

        Returns:
            List[np.ndarray]: Behaviour of each bear at every exclusive time stamp
        """
        if self._behaviour is None:
            raise AttributeError("No behaviours available for trajectory")
        return self._behaviour

    def save_traj(self,
                  traj_path: str,
                  format: str,
                  overlay_shape: np.ndarray = None,
                  save_behaviour: bool = True):
        """Save this trajectory in the given format

        Args:
            traj_path (str): Path ro save to
            format (str): Format to save as
            overlay_shape (np.ndarray, optional): Shape of overlay. Defaults to None.
            save_behaviour (bool, optional): Save behaviour. Defaults to True.

        Raises:
            RuntimeError: Error, if an invalid format is given
        """
        # save a raw trajectory
        if format == "raw":
            self._save_raw_traj(traj_path)
        # save a exclusive trajectory
        elif format == "exclusive":
            self._save_exclusive_traj(traj_path, overlay_shape, save_behaviour)
        # invalid format
        else:
            raise RuntimeError("Format has to be in {}. Got {}".format(self.__FORMATS, format))

    def _save_raw_traj(self,
                       traj_path: str) -> None:
        """Save this trajectory in raw format

        Args:
            traj_path (str): Path ro save raw trajectory to
        """
        data = [[t, [[list(c), d] for c, d in coords]] for t, coords in zip(self.raw_times, self.raw)]

        with open(traj_path, "w") as f:
            json.dump(data, f, indent=4)

    def _save_exclusive_traj(self,
                             traj_path: str,
                             overlay_shape: np.ndarray,
                             save_behaviour: bool = True) -> None:
        """Save this trajectory in exclusive format

        Args:
            traj_path (str): Path to save to
            overlay_shape (np.ndarray): Shape of overlay
            save_behaviour (bool, optional): Save behaviour. Defaults to True.
        """
        data = {"time": self.exclusive_times}
        for i, _ in enumerate(self.names):
            ex = self.exclusive[i] / np.array(overlay_shape)[np.newaxis, ...]
            ex[np.where(ex[:, 0] < 0)] = np.array([-1, -1])
            data["x{}".format(i + 1)] = ex[:, 0]
            data["y{}".format(i + 1)] = ex[:, 1]
            if self._behaviour is not None:
                data["b{}".format(i + 1)] = self._behaviour[i]
        pd.DataFrame(data).to_csv(traj_path, index=False)

    def load_traj(self,
                  traj_path: str,
                  name_list: List[str],
                  format: str,
                  overlay_shape: np.ndarray = None) -> None:
        """Load a single trajectory

        Args:
            traj_path (str): Path to trajectory
            name_list (List[str]): List of names for each individual
            format (str): Format of trajectory to load. Must be either 'exclusive' or 'raw'
            overlay_shape (np.ndarray): Shape of the enclosure overlay to multiply by. Must be given for exclusive format. Defaults to None. 

        Raises:
            RuntimeError: Error, if an invalid format is set internally
        """
        # add names for trajectory
        self._names = name_list
        # parse a raw trajectory
        if format == "raw":
            self._load_raw_traj(traj_path)
        # parse a exclusive trajectory
        elif format == "exclusive":
            self._load_exclusive_traj(traj_path, overlay_shape)
        # invalid format
        else:
            raise RuntimeError("Format has to be in {}. Got {}".format(self.__FORMATS, format))

    def _load_raw_traj(self,
                       traj_path: str):
        """Load a trajectory in raw format and sort by time stamp

        Args:
            traj_path (str): Path to trajectory
        """
        # read raw treajectory
        with open(traj_path, "r") as f:
            data = json.load(f)

        # time stamps and coordiantes with additional data
        times, coordinates = zip(
            *[(t, [(np.array(c), r) for c, *r in d if c is not None and np.all(np.isfinite(c)) and r[1] in self._names])
              for t, d in data])

        # have to convert here, since it is possible to have only invalid detections -> time stamp is considered invalid
        self._raw_times = np.array(times)

        # sort by time stap
        new_idx = np.array([i for _, i in sorted(zip(times, range(len(times))), key=lambda x: x[0])])

        self._raw_times = self._raw_times[new_idx]
        self._raw = [coordinates[i] for i in new_idx]
        self._formats.add("raw")

    def _load_exclusive_traj(self, traj_path: str, overlay_shape: List[int]):
        """Load a trajectory in exclusive format and sort by time stamp

        Args:
            traj_path (str): Path to trajectory

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Array of time stamps, list with an array of coordinates for each individual
        """
        t = pd.read_csv(traj_path)
        # sort by time stamp
        t.sort_values(["time"], inplace=True)
        self._exclusive_times = t["time"].to_numpy()
        self._exclusive = []

        if "b1" in t.columns:
            self._behaviour = []
            parse_beh = True
        else:
            parse_beh = False

        n_classes = 0

        # read coordiantes for every name
        for i, _ in enumerate(self._names):
            # append sorted coordinates
            self._exclusive.append(t[["x{}".format(i + 1), "y{}".format(i + 1)]].to_numpy() * overlay_shape)
            if parse_beh:
                behaviour = t["b{}".format(i + 1)].to_numpy(dtype=int)
                b_n_classes = np.unique(behaviour)
                b_n_classes = b_n_classes[np.where(b_n_classes != -1)]
                if len(b_n_classes) > n_classes:
                    n_classes = len(b_n_classes)
                self._behaviour.append(behaviour)

        if parse_beh:
            if n_classes == 3:
                self._behaviour_classes = ["resting", "stereo", "moving"]
            else:
                self._behaviour_classes = ["no stereo", "stereo"]

        self._formats.add("exclusive")

    def init_behaviour(self, stereo_only: bool):
        """Initialize behaviour memory

        Args:
            stereo_only (bool): Set stereotypy vs rest only mode
        """
        self._behaviour = [np.full(len(self._exclusive_times), -1) for _ in self._exclusive]
        if stereo_only:
            self._behaviour_classes = ["no stereo", "stereo"]
        else:
            self._behaviour_classes = ["resting", "stereo", "moving"]

    def load_behaviour(self,
                       behaviour_paths: Union[List[str], str],
                       annotation_type: str = "cont",
                       classes_path: str = None,
                       join_classes: bool = True,
                       stereo_only: bool = False):
        """Load behaviour annotations

        Args:
            behaviour_paths (Union[List[str], str]): Paths to bahviour annotations
            annotation_type (str, optional): Annotation type. Defaults to "cont".
            classes_path (str, optional): Path to classes. Defaults to None.
            join_classes (bool, optional): Join classes. Defaults to True.
            stereo_only (bool, optional): Join to stereotypy vs rest, by default False

        Raises:
            AttributeError: Error, if invalid annotation type has been given
        """
        if annotation_type == "cont":
            self._load_continuous_behaviour(behaviour_paths)
        elif annotation_type == "inst":
            self._load_instantaneous_behaviour(behaviour_paths)
        else:
            raise AttributeError("Annotation type must be cont or inst. Got {}".format(annotation_type))

        if classes_path is not None:
            self._load_bahviour_classes(classes_path)

        if join_classes:
            self.join_behaviour_classes(stereo_only)

    def _load_bahviour_classes(self, classes_path: str):
        """Load behaviour classes

        Args:
            classes_path (str): Path to classes
        """
        with open(classes_path, "r") as f:
            data = f.read()
        self._behaviour_classes = [d for d in data.replace(" ", "").split() if d != ""]

    def join_behaviour_classes(self, stereo_only: bool = False):
        """Join behaviour classes to simplified stereo, movement and resting\
            
        Args:
            stereo_only (bool, optional): Join to stereotypy vs rest, by default False
            
        Raises:
            AssertionError: Error, if not behaviour classesa have been loaded
            AssertionError: Error, if no behaviour annotations have been laoded
        """
        if self._behaviour_classes is None:
            raise AssertionError("Behaviour classes have to be loaded before joining")
        if self._behaviour is None:
            raise AssertionError("Behaviour annotations have to be loaded before joining")
        if len(self._behaviour_classes) in [2, 3]:
            return
        if not stereo_only:
            new_idxs = list(range(len(self._behaviour_classes)))
            for i, c in enumerate(self._behaviour_classes):
                if c in ["stereolaufen", "stereoschwimmen"]:
                    new_idxs[i] = 1
                elif c in ["stehen", "stehenw", "liegen", "sitzen"]:
                    new_idxs[i] = 0
                elif c in ["laufen", "schwimmen", "rennen"]:
                    new_idxs[i] = 2

            self._behaviour_classes = ["resting", "stereo", "moving"]
        else:
            new_idxs = list(range(len(self._behaviour_classes)))
            for i, c in enumerate(self._behaviour_classes):
                if c in ["stereolaufen", "stereoschwimmen"]:
                    new_idxs[i] = 1
                elif c in ["stehen", "stehenw", "liegen", "sitzen", "laufen", "schwimmen", "rennen"]:
                    new_idxs[i] = 0

            self._behaviour_classes = ["no stereo", "stereo"]
            
        for i, j in enumerate(new_idxs):
            for n, _ in enumerate(self._behaviour):
                self._behaviour[n][np.where(self._behaviour[n] == i)] = j

    def _load_continuous_behaviour(self, behaviour_paths: List[str]):
        """Load continuous behaviour annotations

        Args:
            behaviour_paths (List[str]): List of paths to annotations

        Raises:
            AssertionError: Error, if number of annotations does not equal to the number of individuals
        """
        if len(behaviour_paths) != len(self._exclusive):
            raise AssertionError("The number of behaviour files must be the same as the number of individuals. Got {} and expected {}".format(
                len(behaviour_paths), len(self._exclusive)))
        self._behaviour = []
        for b in behaviour_paths:
            df = pd.read_csv(b)
            p = os.path.basename(b)
            df["Start"] = df["Start"].apply(lambda x: datetime.strptime(
                "{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp())
            df["End"] = df["End"].apply(lambda x: datetime.strptime(
                "{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp())
            behaviour = np.full(len(self._exclusive_times), -1)
            for _, row in df.iterrows():
                if row["Start"] > self._exclusive_times[-1]:
                    continue
                start_idx = np.argmax(self._exclusive_times > row["Start"])
                if row["End"] > self._exclusive_times[-1]:
                    end_idx = len(self._exclusive_times)
                else:
                    end_idx = np.argmax(self._exclusive_times > row["End"])
                behaviour[start_idx:end_idx] = row["Activity"]

            self._behaviour.append(behaviour)

    def _load_instantaneous_behaviour(self, behaviour_path: str):
        """Load instantaneous behaviour annotations

        Args:
            behaviour_paths (List[str]): List of paths to annotations
        """
        df = pd.read_csv(behaviour_path)
        p = os.path.basename(behaviour_path)
        # time frame is 30s before and after time stamp -> set start point 30s into past
        df["Time"] = df["Time"].apply(lambda x: datetime.strptime(
            "{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp() - 30)

        vera_behaviour = np.full(len(self._exclusive_times), -1)
        nanuq_behaviour = np.full(len(self._exclusive_times), -1)

        for _, row in df.iterrows():
            if row["Time"] > self._exclusive_times[-1]:
                continue
            start_idx = np.argmax(self._exclusive_times > row["Time"])
            if row["Time"] + 60 > self._exclusive_times[-1]:
                end_idx = len(self._exclusive_times)
            else:
                end_idx = np.argmax(self._exclusive_times > row["Time"] + 60)
            vera_behaviour[start_idx:end_idx] = row["Vera"]
            nanuq_behaviour[start_idx:end_idx] = row["Nanuq"]

        self._behaviour = [nanuq_behaviour, vera_behaviour]

    def crop(self,
             start_time: float = 0.0,
             end_time: float = None,
             relative: bool = True) -> bool:
        """Crop the trajectory

        Args:
            start_time (float, optional): Start of time frame to crop. Defaults to 0.0.
            end_time (float, optional): End of time frame to crop. If not given, everything from start_time on will be cropped. Defaults to None.
            relative (bool, optional): If True, the time frame will be calculated from the first time stamp on. Defaults to True.

        Returns:
            bool: Return True, if a cropping was possible, and None, if the trajectory does not contain any more information
        """
        could_crop = False
        if self._raw is not None:
            if relative:
                local_start_time = self._raw_times[0] + start_time
                if end_time is None:
                    local_end_time = self._raw_times[-1] + 1
                else:
                    local_end_time = self._raw_times[0] + end_time
            else:
                local_start_time = start_time
                if start_time is None:
                    local_end_time = self._raw_times[-1] + 1
                else:
                    local_end_time = end_time

            if local_start_time <= self._raw_times[-1] and local_end_time >= self._raw_times[0]:
                start_idx = np.argmax(self._raw_times >= local_start_time)
                if self._raw_times[-1] <= local_end_time:
                    end_idx = len(self._raw_times)
                else:
                    end_idx = np.argmax(self._raw_times > local_end_time)

                self._raw_times = self._raw_times[start_idx:end_idx]
                self._raw = self._raw[start_idx:end_idx]

                could_crop = True
            else:
                self._raw_times = None
                self._raw = None
                self._formats.remove("raw")

        if self._exclusive is not None:
            if relative:
                local_start_time = self._exclusive_times[0] + start_time
                if end_time is None:
                    local_end_time = self._exclusive_times[-1] + 1
                else:
                    local_end_time = self._exclusive_times[0] + end_time
            else:
                local_start_time = start_time
                if start_time is None:
                    local_end_time = self._exclusive_times[-1] + 1
                else:
                    local_end_time = end_time

            if local_start_time <= self._exclusive_times[-1] and local_end_time >= self._exclusive_times[0]:
                start_idx = np.argmax(self._exclusive_times >= local_start_time)
                if self._exclusive_times[-1] <= local_end_time:
                    end_idx = len(self._exclusive_times)
                else:
                    end_idx = np.argmax(self._exclusive_times > local_end_time)

                self._exclusive_times = self._exclusive_times[start_idx:end_idx]
                self._exclusive = [e[start_idx:end_idx] for e in self._exclusive]
                self._behaviour = [b[start_idx:end_idx] for b in self._behaviour]

                could_crop = True
            else:
                self._exclusive = None
                self._exclusive_times = None
                self._behaviour = None
                self._formats.remove("exclusive")

        return could_crop

    def resample(self, n: int) -> None:
        """Resample trajectory

        Args:
            n (int): Resample to keep only every n-th sample
        """
        assert n >= 1, "n must be >= 1 for resampling"
        if self._raw is not None:
            self._raw_times = self._raw_times[::n]
            self._raw = self._raw[::n]

        if self._exclusive is not None:
            self._exclusive_times = self._exclusive_times[::n]
            self._exclusive = [e[::n] for e in self._exclusive]
            self._behaviour = [b[::n] for b in self._behaviour]

    def copy(self) -> "Trajectory":
        """Deepcopy this trajectory

        Returns:
            Trajectory: Deep copy of this trajectory
        """
        return copy.deepcopy(self)

    def calc_length(self, max_interp: float = 1, in_meter: bool = False) -> List[float]:
        """Calculate the length of all exclusive trajectories

        Args:
            max_interp (float, optional): Max time in seconds between valid coordinates, which is considered for length calculation. Defaults to 1.

        Returns:
            List[float]: List of lengths
        """
        lengths = [0 for _ in self._exclusive]
        for i, e in enumerate(self.exclusive):
            if len(e) < 2:
                continue
            # get only positive coordinates
            valid_vals = e[:, 0] > 0
            e_valid = e[valid_vals]
            e_t_valid = self.exclusive_times[valid_vals]
            # check for only one positive coordinate
            if len(e_valid) < 2:
                continue
            # calculate space distances
            diff = np.linalg.norm(e_valid[1:] - e_valid[:-1], axis=-1)
            # calculate time distances
            time_diff = e_t_valid[1:] - e_t_valid[:-1]
            # sum up all space differences, where time difference is not too large
            lengths[i] = np.sum(diff[time_diff < max_interp])
            if in_meter:
                lengths[i] *= MPP
        return lengths

    def get_stats(self, absolute: bool = False, correct_for_oos: bool = True) -> TrajectoryStats:
        lengths = self.calc_length(in_meter=True)
        seconds = self.exclusive_times[-1] - self.exclusive_times[0]
        oos_mask = [self.exclusive[i][:, 0] < 0 for i in range(len(self.exclusive))]
        oos = [np.mean(m) for m in oos_mask]
        
        if correct_for_oos:
            mask = [~m for m in oos_mask]
        else:
            mask = [np.full_like(self.exclusive[i], True) for i in range(len(self.exclusive))]
        
        if self._behaviour is not None:
            behaviour_duration = [
                1 - np.sum(np.where(self.behaviour[i][mask[i]] == -1, 1, 0)) / max(len(self.exclusive_times[mask[i]]), 1e-10)
                for i in range(len(self.exclusive))]
            behaviour = [{n: np.sum(np.where(self.behaviour[i][mask[i]] == j, 1, 0)) / max(len(self.exclusive_times[mask[i]]), 1e-10)
                          for j, n in enumerate(self.behaviour_classes)} for i in range(len(self.behaviour))]

            ts = TrajectoryStats(
                PBStats(lengths[0], oos[0], behaviour_duration[0], behaviour[0]),
                PBStats(lengths[1], oos[1], behaviour_duration[1], behaviour[1]),
                seconds)
        else:
            ts = TrajectoryStats(
                PBStats(lengths[0], oos[0]),
                PBStats(lengths[1], oos[1]),
                seconds)

        if absolute:
            ts.to_absolute()

        return ts

    def __str__(self) -> str:
        s = "Trajectory | names: {}".format(self.names)
        if self._exclusive is not None:
            s += " | exclusive: {} coordinates and {} timestamps from {} to {}".format(len(self._exclusive[0]),
                                                                                       len(self._exclusive_times),
                                                                                       self._exclusive_times[0],
                                                                                       self._exclusive_times[-1])
        if self._raw is not None:
            s += " | raw: {} coordinates and {} timestamps from {} to {}".format(len(self._raw),
                                                                                 len(self._raw_times),
                                                                                 self._raw_times[0],
                                                                                 self._raw_times[-1])
        return s
