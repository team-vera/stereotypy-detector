import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Callable
from trajectories.trajectory_utils.particle_samplers import ParticleSampler
from trajectories.trajectory_utils.particle_resamplers import ParticleResampler
from trajectories.trajectory_utils.particle_tracker import ParticleTracker
from trajectories.trajectory_utils.trajectory import Trajectory
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utilities.enums import MPP, MAX_SPEED_PB_PIXELS
import copy
import multiprocessing
import multiprocessing.pool
import itertools


class Filter(ABC):
    def __init__(self,
                 name: str,
                 fps: float,
                 max_interp: float,
                 max_fill: float,
                 min_reg: float,
                 min_reg_density: float,
                 min_inter_density: float,
                 ident_threshold: float,
                 det_threshold: float) -> None:
        """Abstact base class for trajectory mergers

        Args:
            name (str): Name of merger
            fps (float): Frames per second trajectory.
            max_interp (float): Max amount of time (in s) to interpolate.
            max_fill (float): Max amount of time (in s) to fill with invalid values. If lower than sample rate, the sampling rate will be used
            min_reg (float): Minimum time a bear has to bee seen in order to be registered
            min_reg_density (float): Minimum fraction of time a bear has to be seen in the minimum time frame given by min_reg in order to be registered
            min_inter_density (float): Minimum fraction of time a bear has to be seen in the minimum time frame given by min_inter in order to keep interpolating
            ident_threshold (float): Identification threshold
            ident_threshold (float): Detection threshold
        """
        super().__init__()
        self._name = name
        self._fps = fps
        self._sample_rate = 1 / fps
        self._max_interp = max_interp
        self._max_interp_frames = int(np.round(max_interp * self._fps)) 
        self._max_fill = max(max_fill, self._sample_rate)
        self._min_reg = min_reg
        self._min_reg_frames = int(np.round(min_reg * self._fps)) 
        assert min_reg_density >= 0 and min_reg_density <= 1, \
            "Minimal registration desity has to be in [0, 1]. Got {}".format(min_reg_density)
        self._min_reg_density = min_reg_density
        assert min_inter_density >= 0 and min_inter_density <= 1, \
            "Minimal interpolation desity has to be in [0, 1]. Got {}".format(min_reg_density)
        self._min_inter_density = min_inter_density
        self._invalid = - np.ones(2)
        self._ident_th = ident_threshold
        self._det_th = det_threshold

    @abstractmethod
    def merge(self,
              t: Trajectory) -> Trajectory:
        """Merge the given raw trajectory

        Args:
            t (Trajectory): Trajectory to merge
            
        Returns:
            Trajectory: Merged trajectory
        """
        pass

    @abstractmethod
    def apply_exclusive(self, t: Trajectory) -> None:
        """Apply filter to the given exclusive trajectory

        Args:
            t (Trajectory): Trajectory to apply filter to
        """
        pass

    @property
    def name(self) -> str:
        """Get the name of this merger

        Returns:
            str: Name of merger
        """
        return self._name

    def _get_idxs(self, times: np.ndarray) -> List[Tuple[int]]:
        """Split a trajectory by the max interpolation time

        Args:
            times (np.ndarray): Time stamps

        Returns:
            List[Tuple[int]]: Start and end indices of splits
        """
        idxs = []
        prev_t = times[0]
        start_idx = 0
        for i, t in enumerate(times[1:-1]):
            if t - prev_t > self._max_fill:
                idxs.append((start_idx, i + 1))
                start_idx = i + 1
            prev_t = t
        last_idxs = (start_idx, len(times))
        idxs.append(last_idxs)
        return idxs

    def _fill_empty_observations_raw(self, times: np.ndarray, coordinates: List
                                     [List[Tuple[np.ndarray, List[object]]]]) -> Tuple[np.ndarray,
                                                                                       List[List[Tuple[np.ndarray, List[object]]]]]:
        """Fill empty observations and time gaps in a raw trajectory

        Args:
            t (Trajectory): Trajectory to fill observations for

        Returns:
            Tuple[np.ndarray, List[List[Tuple[np.ndarray, List[object]]]]]: Interpolated time stamps, filled up coordiantes and additional data
        """
        times_interp = np.arange(times[0], times[-1] + self._sample_rate / 2, self._sample_rate)
        coordinates_interp = []
        idx_times = 0
        for t in times_interp:
            if idx_times >= len(times):
                coordinates_interp.append([])
                break
            if abs(times[idx_times] - t) > self._sample_rate:
                coordinates_interp.append([])
            else:
                coordinates_interp.append(coordinates[idx_times])
                idx_times += 1
        return times_interp, coordinates_interp

    def _fill_empty_observations_exclusive(
            self, times: np.ndarray, coordinates: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Fill empty observations and time gaps in a exclusive trajectory

        Args:
            t (Trajectory): Trajectory to fill observations for

        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: Interpolated time stamps, filled up coordiantes
        """
        times_interp = np.arange(times[0], times[-1] + self._sample_rate / 2, self._sample_rate)
        coordinates_interp = [[] for _ in range(len(coordinates))]
        empty = [-1, -1]
        idx_times = 0
        for t in times_interp:
            if idx_times >= len(times):
                [coordinates_interp[i].append(empty) for i in range(len(coordinates))]
                break
            if abs(times[idx_times] - t) > self._sample_rate:
                [coordinates_interp[i].append(empty) for i in range(len(coordinates))]
            else:
                [coordinates_interp[i].append(coordinates[i][idx_times])
                 for i in range(len(coordinates))]
                idx_times += 1

        coordinates_interp = [np.array(c) for c in coordinates_interp]

        return times_interp, coordinates_interp

    def _split_raw_by_name(self,
                           times: np.ndarray,
                           coordinates: List[List[Tuple[np.ndarray, List[object]]]],
                           names: List[str]) -> List[List[Tuple[bool, Union[List[np.ndarray], np.ndarray]]]]:
        """Split the given trajectory by invalid indices for every name

        Args:
            times (np.ndarray): Time stamps
            coordinates (List[List[Tuple[np.ndarray, List[object]]]]): Coordinates and additional data for each individual and time stamp


        Returns:
            List[List[Tuple[bool, Union[List[np.ndarray], np.ndarray]]]]: List of tuples of flag for processing (True, if it should be filtered) and the associated coordinates for each individual -> shape: [individuals, splits, [flag, coordiantes]]
        """
        out_list = [[] for _ in names]

        for j, name in enumerate(names):

            start_idx = 0
            end_idx = 1
            mode = None

            # get a list of all coordinates associated with the given name
            c_for_name = [[c for c, d in sorted(coordinates[i], key=lambda x: x[1][2], reverse=True)
                           if d[1] == name and d[0] > self._det_th and d[2] > self._ident_th]
                          for i in range(len(times))]

            c_for_name_valids = [int(len(c_) > 0) for c_ in c_for_name]

            for i, c in enumerate(c_for_name):
                if i < len(c_for_name) - 1:
                    c_valid_reg = c_for_name_valids[i:i+self._min_reg_frames]
                    c_density = sum(c_valid_reg) / (times[min(i+self._min_reg_frames, len(times) - 1)] - times[i]) / self._fps
                    c_valid_reg_inter = c_for_name_valids[i:i+self._max_interp_frames]
                    c_inter_density = sum(c_valid_reg_inter) / (times[min(i+self._max_interp_frames, len(times) - 1)] - times[i]) / self._fps
                    c_valid_interp = c_for_name_valids[i:i+self._max_interp_frames]
                    try:
                        first_valid = c_valid_interp.index(1)
                        if first_valid == 0:
                            interp = True
                        else:
                            interp = (times[i+first_valid] - times[i]) < self._max_interp and c_inter_density > self._min_inter_density
                    except ValueError:
                        interp = False
                else:
                    c_density = int(len(c) > 0)
                    interp = not c_density
                if len(c) > 0:
                    if mode is None:
                        mode = True
                    if not mode:
                        if c_density > self._min_reg_density:
                            out_list[j].append((False, np.full((end_idx - start_idx, 2), self._invalid)))
                            mode = True
                            start_idx = end_idx
                        end_idx = i + 1
                    else:
                        end_idx = i + 1
                else:
                    if mode is None:
                        mode = False
                    if mode:
                        if not interp:
                            out_list[j].append((True, c_for_name[start_idx:end_idx]))
                            start_idx = end_idx
                            mode = False
                        end_idx = i + 1
                    else:
                        end_idx = i + 1

            if mode:
                if end_idx == len(c_for_name):
                    out_list[j].append((True, c_for_name[start_idx:]))
                else:
                    out_list[j].append((True, c_for_name[start_idx:end_idx]))
                    out_list[j].append((False, np.full((len(c_for_name) - end_idx, 2), self._invalid)))
            elif mode == False:
                if end_idx == len(c_for_name):
                    out_list[j].append(
                        (False, np.full((len(c_for_name) - start_idx, 2), self._invalid)))
                else:
                    out_list[j].append((False, np.full((end_idx - start_idx, 2), self._invalid)))
                    out_list[j].append((True, c_for_name[end_idx:]))

        return out_list

    def _split_exclusive_by_name(self,
                                 times: np.ndarray,
                                 coordinates: List[np.ndarray]) -> List[List[Tuple[bool,
                                                                                   Union[List[np.ndarray],
                                                                                         np.ndarray]]]]:
        """Split the given trajectory by invalid indices for every name

        Args:
            times (np.ndarray): Time stamps
            coordinates (List[np.ndarray]): Coordinates for each individual and time stamp


        Returns:
            List[List[Tuple[bool, Union[List[np.ndarray], np.ndarray]]]]: List of tuples of flag for processing (True, if it should be filtered) and the associated coordinates for each individual -> shape: [individuals, splits, [flag, coordiantes]]
        """
        out_list = [[] for _ in coordinates]

        for j in range(len(coordinates)):

            start_idx = 0
            end_idx = 1
            mode = None
            
            coordinates_valids = [int(c_[0] > 0) for c_ in coordinates[j]]

            for i, c in enumerate(coordinates[j]):
                if i < len(coordinates[j]) - 1:
                    c_valid_reg = coordinates_valids[i:i+self._min_reg_frames]
                    c_density = sum(c_valid_reg) / (times[min(i+self._min_reg_frames, len(times) - 1)] - times[i]) / self._fps
                    c_valid_reg_inter = coordinates_valids[i:i+self._max_interp_frames]
                    c_inter_density = sum(c_valid_reg_inter) / (times[min(i+self._max_interp_frames, len(times) - 1)] - times[i]) / self._fps
                    c_valid_interp = coordinates_valids[i:i+self._max_interp_frames]
                    try:
                        first_valid = c_valid_interp.index(1)
                        if first_valid == 0:
                            interp = True
                        else:
                            interp = (times[i+first_valid] - times[i]) < self._max_interp and c_inter_density > self._min_inter_density
                    except ValueError:
                        interp = False
                else:
                    c_density = int(c[0] > 0)
                    c_inter_density = int(c[0] > 0)
                    interp = not c_density
                    
                if c[0] > 0:
                    if mode is None:
                        mode = True
                    if not mode:
                        if c_density > self._min_reg_density:
                            out_list[j].append((False, np.full((end_idx - start_idx, 2), self._invalid)))
                            mode = True
                            start_idx = end_idx
                        end_idx = i + 1
                    else:
                        end_idx = i + 1
                else:
                    if mode is None:
                        mode = False
                    if mode:
                        if not interp:
                            out_list[j].append((True, coordinates[j][start_idx:end_idx]))
                            start_idx = end_idx
                            mode = False
                        end_idx = i + 1
                    else:
                        end_idx = i + 1

            if mode:
                if end_idx == len(coordinates[j]):
                    out_list[j].append((True, coordinates[j][start_idx:]))
                else:
                    out_list[j].append((True, coordinates[j][start_idx:end_idx]))
                    out_list[j].append(
                        (False, np.full((len(coordinates[j]) - end_idx, 2), self._invalid)))
            elif mode == False:
                if end_idx == len(coordinates[j]):
                    out_list[j].append(
                        (False, np.full((len(coordinates[j]) - start_idx, 2), self._invalid)))
                else:
                    out_list[j].append((False, np.full((end_idx - start_idx, 2), self._invalid)))
                    out_list[j].append((True, coordinates[j][end_idx:]))

        return out_list


class ParticeFilter(Filter):
    def __init__(self,
                 particle_sampler: ParticleSampler,
                 particle_resampler: ParticleResampler,
                 particle_tracker: ParticleTracker,
                 fps: float = 12.5,
                 max_interp: float = 10,
                 max_fill: float = 10,
                 min_reg: float = 2,
                 min_reg_density: float = 0.5,
                 min_inter_density: float = 0.1,
                 num_particles: int = 100,
                 ident_threshold: float = 0.0,
                 det_threshold: float = 0.0,
                 debug_plot: bool = True,
                 mp_inner: bool = False,
                 mp_outer: bool = False,
                 processes_inner: int = 8,
                 processes_outer: int = 8,
                 force_threadsafe: bool = False) -> None:
        """Particle filter for processing raw observations

        Args:
            particle_sampler (ParticleSampler): Particle sampler to use for particle generation
            particle_resampler (ParticleResampler): Particle resampler for weighting an resampling particles
            particle_tracker (ParticleTracker): Particle tracker for tracking the history of a particle and producing the actual position from the particles
            fps (float, optional): Frames per second. Defaults to 12.5.
            max_interp (float, optional): Maximum time to interpolate. Defaults to 10.
            max_fill (float, optional): Maximum time to fill (should be larger or equal to max_interp). Defaults to 10.
            min_reg (float, optional): Minimal time a bear is visible to accept as the start of a trajectory. Defaults to 2.
            min_reg_density (float, optional): Minimum fraction of time a bear has to be seen in the minimum time frame given by min_reg in order to be registered.
            min_inter_density (float, optional): Minimum fraction of time a bear has to be seen in the minimum time frame given by min_interp in order to keep interpolating.
            num_particles (int, optional): Number of particles to filter with. Defaults to 100.
            ident_threshold (float, optional): Identification threshold. Defaults to 0.0.
            det_threshold (float, optional): Detection threshold. Defaults to 0.0.
            debug_plot (bool, optional): DO debug plots. Defaults to True.
            mp_inner (bool, optional): Use innner multiprocessing. Defaults to False.
            mp_outer (bool, optional): Use outer multiprocessing. Defaults to False.
            processes_inner (int, optional): Number of inner processes. Defaults to 8.
            processes_outer (int, optional): Number of outer processes. Defaults to 8.
            force_threadsafe (bool, optional): Force the filter to be threadsafe. Defaults to False.
        """
        super().__init__("ParticleFilter", fps, max_interp, max_fill,
                         min_reg, min_reg_density, min_inter_density, ident_threshold, det_threshold)
        self._particle_sampler = particle_sampler
        self._particle_resampler = particle_resampler
        self._particle_tracker = particle_tracker
        self._num_particles = num_particles
        self._debug_plot = debug_plot
        self._mp_inner = mp_inner
        self._mp_outer = mp_outer
        self._processes_inner = processes_inner
        self._processes_outer = processes_outer
        self._force_threadsafe = force_threadsafe

    def merge(self, t: Trajectory) -> Trajectory:
        """Merge a trajectory with this particle filter

        Args:
            t (Trajectory): Trajectory to merge
            
        Returns:
            Trajectory: Merged trajectory
        """
        # get indices for proper splitting
        idxs = self._get_idxs(t.raw_times)

        num_processes = min(self._processes_outer, len(idxs))

        if not self._mp_outer or num_processes < 2:
            new_times, merged_coordinates = zip(*map(self._raw_outer_filter_helper,
                                                     zip(idxs, itertools.repeat(t, len(idxs)))))
        else:
            with multiprocessing.pool.ThreadPool(num_processes) as p:
                new_times, merged_coordinates = zip(
                    *p.map(self._raw_outer_filter_helper, zip(idxs, itertools.repeat(t, len(idxs)))))

        merged_coordinates = list(zip(*merged_coordinates))

        t.exclusive_times = np.concatenate(new_times, axis=0)

        t.exclusive = [np.concatenate([m for m in merged_coordinates[i] if m is not None])
                       for i in range(len(t.names)) if len(merged_coordinates[i]) > 0]
        
        return t

    def apply_exclusive(self, t: Trajectory) -> None:
        # get indices for proper splitting
        idxs = self._get_idxs(t.exclusive_times)

        num_processes = min(self._processes_outer, len(idxs))

        if not self._mp_outer or num_processes < 2:
            new_times, merged_coordinates = zip(*map(self._exclusive_outer_filter_helper,
                                                     zip(idxs, itertools.repeat(t, len(idxs)))))
        else:
            with multiprocessing.pool.ThreadPool(num_processes) as p:
                new_times, merged_coordinates = zip(*p.map(self._exclusive_outer_filter_helper,
                                                           zip(idxs, itertools.repeat(t, len(idxs)))))

        merged_coordinates = list(zip(*merged_coordinates))

        t.exclusive_times = np.concatenate(new_times, axis=0)

        t.exclusive = [np.concatenate([m for m in merged_coordinates[i] if m is not None])
                       for i in range(len(t.names)) if len(merged_coordinates[i]) > 0]

    def _exclusive_outer_filter_helper(
            self, idx_t: Tuple[Tuple[int, int], Trajectory]) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        idx, t = idx_t

        merged_coordinates = [None for _ in t.names]
        start, end = idx

        # fill remaining missing time stamps with empty detections
        new_times, coords = self._fill_empty_observations_exclusive(
            t.exclusive_times[start:end], [e[start:end] for e in t.exclusive])

        split_coords = self._split_exclusive_by_name(new_times,
                                                     coords)

        if self._debug_plot:
            plt.ion()

        # deal with every individual separately
        for j, _ in enumerate(t.names):

            num_processes = min(self._processes_inner, len(split_coords[j]))

            if not self._mp_inner or num_processes < 2:
                mc = list(map(self._exclusive_inner_filter_helper, split_coords[j]))
            else:
                with multiprocessing.Pool(num_processes) as p:
                    mc = list(p.map(self._exclusive_inner_filter_helper, split_coords[j]))

            if len(mc) > 0:
                merged_coordinates[j] = np.concatenate(mc)

        return new_times, merged_coordinates

    def _exclusive_inner_filter_helper(
            self, p_s: Tuple[bool, Union[List[np.ndarray], np.ndarray]]) -> np.ndarray:
        process, split = p_s
        if not process:
            return split
        else:
            split = [np.array([s, ]) if s[0] > 0 else np.array([]) for s in split]
            if not self._mp_inner and not self._mp_outer:
                return self._filter(split)
            else:
                return self._filter_threadsafe(split)

    def _raw_outer_filter_helper(
            self, idx_t: Tuple[Tuple[int, int], Trajectory]) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        idx, t = idx_t

        new_times = []
        merged_coordinates = [None for _ in t.names]

        start, end = idx

        # fill remaining missing time stamps with empty detections
        new_times, coords = self._fill_empty_observations_raw(t.raw_times[start:end],
                                                              t.raw[start:end])

        split_coords = self._split_raw_by_name(new_times, coords, t.names)

        if self._debug_plot:
            plt.ion()

        # deal with every individual separately
        for j, _ in enumerate(t.names):

            num_processes = min(self._processes_inner, len(split_coords[j]))

            if not self._mp_inner or num_processes < 2:
                mc = list(map(self._raw_inner_filter_helper, split_coords[j]))
            else:
                with multiprocessing.Pool(num_processes) as p:
                    mc = list(p.map(self._raw_inner_filter_helper, split_coords[j]))

            if len(mc) > 0:
                merged_coordinates[j] = np.concatenate(mc)

        return new_times, merged_coordinates

    def _raw_inner_filter_helper(
            self, p_s: Tuple[bool, Union[List[np.ndarray], np.ndarray]]) -> np.ndarray:
        process, split = p_s
        if not process:
            return split
        else:
            if not self._mp_inner and not self._mp_outer and not self._force_threadsafe:
                return self._filter(split)
            else:
                return self._filter_threadsafe(split)

    def _filter(self, split: List[np.ndarray]):
        if self._debug_plot:
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            past_observations = [[], []]

        local_plot = self._debug_plot

        # do not process empty data

        initialized = False
        init_idx = 0
        fully_initialized = False
        state = None

        # reset particle tracker to length of split
        self._particle_tracker.reset(len(split))
        self._particle_sampler.reset()

        # initialize previous probability with uniform distribution for all particles
        weights_prev = np.full(self._num_particles, 1 / self._num_particles)

        prev_movement = None

        start_time = time.time()

        # iterate through all coordiantes
        for i, c_for_name in enumerate(split):
            # check, if is's a leading invalid coordinate
            if not fully_initialized and len(c_for_name) < 1:
                pass
            # valid coordiante and not initialized -> initialize
            elif not initialized:
                self._particle_tracker[i] = c_for_name[0]
                initialized = True
                init_idx = i
            elif not fully_initialized and len(c_for_name) > 0:
                state = np.full((self._num_particles, 2), c_for_name[0])
                self._particle_tracker[i] = state[0]
                prev_movement = (state - self._particle_tracker[init_idx]) / (i - init_idx)
                if np.linalg.norm(prev_movement[0]) > MAX_SPEED_PB_PIXELS * self._sample_rate:
                    prev_movement = 2 * self._particle_sampler.rng.random(prev_movement.shape) - 1
                    prev_movement *= np.sqrt(MAX_SPEED_PB_PIXELS)
                fully_initialized = True
                self._particle_tracker.set_index(i + 1)
            else:
                # current observation
                y = np.array(c_for_name)
                # sample particles and get their probability
                x_p = self._particle_sampler.sample(x=state,
                                                    prev_delta_x=prev_movement,
                                                    y=y)
                # resample to meet weight distribution
                new_idxs, weights = self._particle_resampler.resample(x_p, y, weights_prev)

                prev_movement = x_p - state

                # reindex new state
                state = x_p[new_idxs]
                weights_prev = weights[new_idxs]
                prev_movement = prev_movement[new_idxs]
                self._particle_sampler.reindex(new_idxs)

                self._particle_tracker.add_particles(x_p, new_idxs, weights)

                if local_plot:
                    try:
                        ax.clear()

                        trajectory = self._particle_tracker.trajectory

                        ax.plot(past_observations[0], past_observations[1], color="g")

                        if y.shape[0] > 0:
                            ax.scatter(y[:, 0], y[:, 1], color="g",
                                       marker="+", s=100, label="observation")
                            past_observations[0].append(y[0, 0])
                            past_observations[1].append(y[0, 1])

                        if hasattr(self._particle_tracker, "_memory_size"):
                            start_idx = i + 1 - self._particle_tracker._memory_size
                            start_idx = max(start_idx, init_idx + 1)
                            traj = trajectory[start_idx - 1: i + 1]
                            traj = traj[np.where(traj[:, 0] > 0)]

                            ax.plot(traj[:, 0], traj[:, 1], label="trajectory", color="r")

                            traj = trajectory[: start_idx]
                            traj = traj[np.where(traj[:, 0] > 0)]

                            ax.plot(traj[:, 0], traj[:, 1], label="memory trajectory", color="C4")
                        else:
                            traj = trajectory[: i + 1]
                            traj = traj[np.where(traj[:, 0] > 0)]

                            ax.plot(traj[:, 0], traj[:, 1], label="trajectory", color="r")

                        ax.scatter(x_p[:, 0], x_p[:, 1], label="raw particles", s=1, color="C0")

                        new_idx_list = list(new_idxs)
                        counts = {idx: new_idx_list.count(idx) for idx in set(new_idx_list)}

                        count_idxs = list(counts.keys())

                        x = np.array([x_p[i] for i in count_idxs])
                        s = [counts[i] for i in count_idxs]

                        ax.scatter(x[:, 0], x[:, 1], label="resampled particles", s=s, color="C1")

                        mean_prev_movement = np.mean(prev_movement, axis=0)

                        if trajectory[i, 0] > 0:
                            ax.scatter(
                                trajectory[i, 0] + mean_prev_movement[0],
                                trajectory[i, 1] + mean_prev_movement[1],
                                color="r", marker="+", s=100, label="predicted next state")

                        if hasattr(self._particle_sampler, "_momentum"):
                            mean_momentum = np.mean(self._particle_sampler._momentum, axis=0)
                            if trajectory[i, 0] > 0:
                                ax.scatter(
                                    trajectory[i, 0] + mean_momentum[0],
                                    trajectory[i, 1] + mean_momentum[1],
                                    color="r", marker="x", s=100, label="momentum")

                        ax.legend()
                        ticks = ax.get_yticks()
                        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
                        ax.set_yticklabels(["{:.2f}".format(x) for x in ticks * MPP])
                        ticks = ax.get_xticks()
                        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
                        ax.set_xticklabels(["{:.2f}".format(x) for x in ticks * MPP])

                        fig.canvas.draw()
                        fig.canvas.flush_events()

                        # time.sleep(1)

                    except KeyboardInterrupt:
                        self._debug_plot = False
                        local_plot = False
                        plt.close("all")
                    except Exception as ex:
                        print(ex)
                        plt.close("all")

                    if not plt.fignum_exists(fig.number):
                        local_plot = False
                
                if self._debug_plot:
                    print("FPS: {}     ".format(i / (time.time() - start_time)), end="\r")

        if self._debug_plot:
            while plt.fignum_exists(fig.number):
                plt.pause(0.1)
            plt.close("all")

        return self._particle_tracker.trajectory

    def _filter_threadsafe(self, split: List[np.ndarray]):
        # do not process empty data
        initialized = False
        init_idx = 0
        fully_initialized = False
        state = None

        # create deep copy for threading
        pt = copy.deepcopy(self._particle_tracker)
        ps = copy.deepcopy(self._particle_sampler)
        pr = copy.deepcopy(self._particle_resampler)

        # reset particle tracker to length of split
        pt.reset(len(split))
        ps.reset()

        # initialize previous probability with uniform distribution for all particles
        weights_prev = np.full(self._num_particles, 1 / self._num_particles)

        prev_movement = None

        # iterate through all coordiantes
        for i, c_for_name in enumerate(split):
            # check, if is's a leading invalid coordinate
            if not fully_initialized and len(c_for_name) < 1:
                pass
            # valid coordiante and not initialized -> initialize
            elif not initialized:
                pt[i] = c_for_name[0]
                initialized = True
                init_idx = i
            elif not fully_initialized and len(c_for_name) > 0:
                state = np.full((self._num_particles, 2), c_for_name[0])
                pt[i] = state[0]
                prev_movement = (state - pt[init_idx]) / (i - init_idx)
                if np.linalg.norm(prev_movement[0]) > MAX_SPEED_PB_PIXELS * self._sample_rate:
                    prev_movement = 2 * ps.rng.random(prev_movement.shape) - 1
                    prev_movement *= np.sqrt(MAX_SPEED_PB_PIXELS)
                fully_initialized = True
                pt.set_index(i + 1)
            else:
                # current observation
                y = np.array(c_for_name)
                # sample particles and get their probability
                x_p = ps.sample(x=state,
                                prev_delta_x=prev_movement,
                                y=y)
                # resample to meet weight distribution
                new_idxs, weights = pr.resample(x_p, y, weights_prev)

                prev_movement = x_p - state

                # reindex new state
                state = x_p[new_idxs]
                weights_prev = weights[new_idxs]
                prev_movement = prev_movement[new_idxs]
                ps.reindex(new_idxs)

                pt.add_particles(x_p, new_idxs, weights)

        return pt.trajectory


FILTER_MAP: Dict[str, Filter] = {
    "particle": ParticeFilter
}
