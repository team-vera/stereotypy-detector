from trajectories.trajectory_utils.trajectory_pack import TrajectoryPack
from trajectories.trajectory_utils.trajectory import Trajectory
from trajectories.trajectory_utils.splitters import Splitter
from trajectories.trajectory_utils.feature_extractor import FeatureExtractor, WindowedFFTFeatureExtractor, MixedFreqFeatureExtractor
from scipy.signal import stft
from utilities.enums import MPP
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.ticker as mticker
import time
from datetime import datetime
from typing import List, Union, Iterable
from collections import abc
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE


def plot(t_pack: TrajectoryPack,
         ax: plt.Axes,
         overlay: np.ndarray,
         splitter: Splitter = None,
         individuals: List[str] = None,
         nice_plot: bool = True) -> None:
    """Plot the given exclusive trajectory onto the overlay

    Args:
        t_pack (TrajectoryPack): Trajectorie pack to plot
        ax (plt.Axes): Axes to plot onto
        overlay (np.ndarray): Overaly
        splitter (Splitter, optional): Splitter to use for splitting the trajectory. Defaults to None.
        individuals (List[str], optional): names of all individuals. Defaults to None.
        nice_plot (bool, optional): Do a bit of a nicer plot (better ticks). Defaults to True.
    """
    ax.imshow(overlay)
    labeled = set()
    if splitter is not None:
        overall_idxs = [splitter.split_exclusive(t.exclusive_times, t.exclusive)
                        for t in t_pack.trajectories]
    else:
        overall_idxs = [[[(0, len(t.exclusive_times))] for _ in t.names]
                        for t in t_pack.trajectories]

    for traj, idxs_for_name in zip(t_pack.trajectories, overall_idxs):
        for i, (n, c, idxs) in enumerate(zip(traj.names, traj.exclusive, idxs_for_name)):
            if individuals is not None and n not in individuals:
                continue
            for j in idxs:
                c_split = c[j[0]:j[1]]
                if n in labeled:
                    to_label = False
                else:
                    labeled.add(n)
                    to_label = True
                color = t_pack.COLOR_CYCLE[i % len(t_pack.COLOR_CYCLE)]
                idx = c_split[:, 0] > 0
                if to_label:
                    ax.plot(c_split[idx, 0], c_split[idx, 1], label=n, linewidth=0.6, color=color)
                else:
                    ax.plot(c_split[idx, 0], c_split[idx, 1], linewidth=0.6, color=color)

    if nice_plot:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")


def plot_x_y_num_traj(t_pack: TrajectoryPack,
                      ax: plt.Axes):
    """Plot the number of available trajectory points for each time stamp as background

    Args:
        t_pack (TrajectoryPack): Trajectory pack
        ax (plt.Axes): Axes to plot onto
    """
    colors = ["C0", "C1", "C2"]
    color_labeled = [False, False, False]
    time_frames = [[t.raw_times[0], t.raw_times[-1]]
                   if "raw" in t.formats else
                   [t.exclusive_times[0], t.exclusive_times[-1]] for t in t_pack.trajectories]
    key_times = []
    [key_times.extend(t) for t in time_frames]
    key_times.sort()
    key_frames = [[key_times[i], key_times[i + 1]] for i in range(len(key_times) - 1)]
    t_count = [
        sum([1 if kf[0] >= tf[0] and kf[1] <= tf[1] else 0 for tf in time_frames])
        for kf in key_frames
    ]

    for kf, tc in zip(key_frames, t_count):
        tc = tc % (len(colors) + 1) # quick and dirty hack for plotting many synthetic trajectories
        if color_labeled[tc - 1]:
            ax.axvspan(kf[0], kf[1], color=colors[tc - 1], alpha=0.1)
        else:
            ax.axvspan(kf[0], kf[1], color=colors[tc - 1], label=str(tc), alpha=0.1)
            color_labeled[tc - 1] = True
            
def plot_behaviour(t_pack: TrajectoryPack,
                   ax: List[plt.Axes],
                   join_classes: bool = True):
    assert len(ax) == len(t_pack.trajectories[0].names), \
        "Number of axes must be the same as number of subjects. Got {} and {}.".format(
            len(ax), len(t_pack.trajectories[0].names)
        )
    
    colors = ["C0", "C1", "C2"]
    if join_classes:
        [t.join_behaviour_classes() for t in t_pack.trajectories]
    for i, a in enumerate(ax):
        spans = [[] for _ in t_pack.trajectories[0].behaviour_classes]
        for t in t_pack.trajectories:
            idxs = np.arange(len(t.exclusive_times) - 1)
            b_diff = t.behaviour[i][:-1] - t.behaviour[i][1:]
            flip_points = list(idxs[np.where(b_diff != 0)] + 1)
            flip_points.append(len(t.exclusive_times) - 1)
            start = 0
            for f in flip_points:
                b = t.behaviour[i][start]
                if b != -1:
                    spans[b].append([t.exclusive_times[start], t.exclusive_times[f]])
                start = f
        for j, sp in enumerate(spans):
            if len(sp) < 1:
                continue
            for s in sp[:-1]:
                a.axvspan( s[0], s[1], color=colors[j], alpha=0.1)
            a.axvspan(sp[-1][0], sp[-1][1], color=colors[j], label=t_pack.trajectories[0].behaviour_classes[j], alpha=0.1)
            
            
def plot_behaviour_diff(t_pack_1: TrajectoryPack,
                        t_pack_2: TrajectoryPack,
                        ax: List[plt.Axes],
                        join_classes: bool = True):
    assert len(ax) == len(t_pack_1.trajectories[0].names), \
        "Number of axes must be the same as number of subjects. Got {} and {}.".format(
            len(ax), len(t_pack_1.trajectories[0].names)
        )
    assert len(ax) == len(t_pack_2.trajectories[0].names), \
        "Number of axes must be the same as number of subjects. Got {} and {}.".format(
            len(ax), len(t_pack_2.trajectories[0].names)
        )
    
    colors = ["C0", "C1", "C2"]
    colors = ["C0", "red", "blue"]
    if join_classes:
        [t.join_behaviour_classes(stereo_only=True) for t in t_pack_1.trajectories]
        [t.join_behaviour_classes(stereo_only=True) for t in t_pack_2.trajectories]
    labels = ["correct", "FP", "FN"]
    for i, a in enumerate(ax):
        spans = [[] for _ in labels]
        for t_1, t_2 in zip(t_pack_1.trajectories, t_pack_2.trajectories):
            idxs = np.arange(len(t_1.exclusive_times) - 1)
            b_diff = np.zeros_like(t_1.behaviour[i])
            b_diff[np.logical_and(t_1.behaviour[i] == 1, t_2.behaviour[i] == 0)] = 1
            b_diff[np.logical_and(t_1.behaviour[i] == 0, t_2.behaviour[i] == 1)] = 2
            time_diff = b_diff[:-1] - b_diff[1:]
            flip_points = list(idxs[np.where(time_diff != 0)] + 1)
            flip_points.append(len(t_1.exclusive_times) - 1)
            start = 0
            for f in flip_points:
                b = b_diff[start]
                if b != -1:
                    spans[b].append([t_1.exclusive_times[start], t_1.exclusive_times[f]])
                start = f
        for j, sp in enumerate(spans[1:], start=1):
            if len(sp) < 1:
                continue
            for s in sp[:-1]:
                a.axvspan(s[0], s[1], color=colors[j], alpha=0.1)
            a.axvspan(sp[-1][0], sp[-1][1], color=colors[j], label=labels[j], alpha=0.1)
    
    
def plot_cont_gt(behavior_paths: List[str],
                 classes_path: str,
                 ax: Union[plt.Axes, List[plt.Axes]],
                 join_classes: bool = True):
    """Plot continuous ground truth into the background

    Args:
        behavior_paths (List[str]): Path to bahavior ground truth
        classes_path (str): Path to classes
        ax (Union[plt.Axes, List[plt.Axes]]): Axes or axis to plot onto
        join_classes (bool, optional): Join classes into stereo, moving and resting. Defaults to True.
    """
    if isinstance(behavior_paths, str):
        behavior_paths = [behavior_paths,]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cont_sets = {os.path.basename(p): pd.read_csv(p) for p in behavior_paths}
    for p, df in cont_sets.items():
        df["Start"] = df["Start"].apply(lambda x: datetime.strptime("{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp())
        df["End"] = df["End"].apply(lambda x: datetime.strptime("{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp())

    with open(classes_path, "r") as f:
        data = f.read()
    classes = [d for d in data.replace(" ", "").split() if d != ""]
    if join_classes:
        # new classes: stereo, moving, resting
        old_colors = colors.copy()
        for i, c in enumerate(classes):
            if c in ["stereolaufen", "stereoschwimmen"]:
                classes[i] = "stereo"
                colors[i] = old_colors[0]
            elif c in ["stehen", "stehenw", "liegen", "sitzen"]:
                classes[i] = "resting"
                colors[i] = old_colors[1]
            elif c in ["laufen", "schwimmen", "rennen"]:
                classes[i] = "moving"
                colors[i] = old_colors[2]
            
    for n in sorted(cont_sets.keys()):
        for i, row in cont_sets[n].iterrows():
            idx = int(row["Activity"])
            activity = classes[idx]
            if row["Start"] > row["End"]:
                raise RuntimeError("Issue in line {} of {}".format(i, n))

            if isinstance(ax, plt.Axes):
                ax.axvspan(row["Start"], row["End"], color=colors[idx], label=activity, alpha=0.2, linewidth=0)
            else:
                for a in ax:
                    a.axvspan(row["Start"], row["End"], color=colors[idx], label=activity, alpha=0.2, linewidth=0)

def plot_inst_gt(behavior_paths: List[str],
                 classes_path: str,
                 ax: Union[plt.Axes, List[plt.Axes]],
                 join_classes: bool = True,
                 subject: str = "Vera"):
    """Plot continuous ground truth into the background

    Args:
        behavior_paths (List[str]): Path to bahavior ground truth
        classes_path (str): Path to classes
        ax (Union[plt.Axes, List[plt.Axes]]): Axes or axis to plot onto
        join_classes (bool, optional): Join classes into stereo, moving and resting. Defaults to True.
        subject (str, optional): Subject to plot for. Vera or Nanuq
    """
    if isinstance(behavior_paths, str):
        behavior_paths = [behavior_paths,]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    cont_sets = {os.path.basename(p): pd.read_csv(p) for p in behavior_paths}
    for p, df in cont_sets.items():
        df["Time"] = df["Time"].apply(lambda x: datetime.strptime("{}{}".format(p, x), "%Y%m%d.csv%H:%M:%S").timestamp())

    with open(classes_path, "r") as f:
        data = f.read()
    classes = [d for d in data.replace(" ", "").split() if d != ""]
    if join_classes:
        # new classes: stereo, moving, resting
        old_colors = colors.copy()
        for i, c in enumerate(classes):
            if c in ["stereolaufen", "stereoschwimmen"]:
                classes[i] = "stereo"
                colors[i] = old_colors[0]
            elif c in ["stehen", "stehenw", "liegen", "sitzen"]:
                classes[i] = "resting"
                colors[i] = old_colors[1]
            elif c in ["laufen", "schwimmen", "rennen"]:
                classes[i] = "moving"
                colors[i] = old_colors[2]
            
    for n in sorted(cont_sets.keys()):
        for i, row in cont_sets[n].iterrows():
            idx = int(row[subject])
            if idx < 0:
                continue
            activity = classes[idx]

            if isinstance(ax, plt.Axes):
                ax.axvspan(row["Time"] - 30, row["Time"] + 30, color=colors[idx], label=activity, alpha=0.2, linewidth=0)
            else:
                for a in ax:
                    a.axvspan(row["Time"] - 30, row["Time"] + 30, color=colors[idx], label=activity, alpha=0.2, linewidth=0)
                    
def plot_wfft(traj: Trajectory,
              splitter: Splitter,
              axes: List[plt.Axes],
              window_size: int = 60,
              select_axis: bool = False):
    
    assert len(axes) == len(traj.names), "Number of axes must be the same as number of individuals"
    
    overall_idxs = splitter.split_exclusive(traj.exclusive_times, traj.exclusive)
    colorbars = []
    wfft_ex = WindowedFFTFeatureExtractor(select_axis=select_axis, window_size=window_size)

    for i, (a, c, idxs) in enumerate(zip(axes, traj.exclusive, overall_idxs)):
        for start, end in idxs:
            # only get splits with 2 min of valid values
            if end - start < window_size * 12.5:
                continue
            section = c[start:end]
            f, mag, t = wfft_ex.extract_features(section)
            
            mag = wfft_ex.normalize(mag)
            
            mag = np.swapaxes(mag, 0, 1)

            colorbars.append(a.pcolormesh(t + traj.exclusive_times[start], f, mag, shading='auto'))

        a.set_ylim(0, 0.5)
        a.set_ylabel("Frequency [Hz]")
        a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y %H:%M:%S")))
            
    return colorbars

def plot_wfft_tsne(traj: TrajectoryPack,
                   splitter: Splitter,
                   samples_per_duration: int = 1500,
                   select_axis: bool = False,
                   mixed_features: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    features = [[], []]
    classes = [[], []]
    
    for t in traj.trajectories:
        overall_idxs = splitter.split_exclusive(t.exclusive_times, t.exclusive)
        if not mixed_features:
            extractor = WindowedFFTFeatureExtractor(select_axis=select_axis)
        else:
            extractor = MixedFreqFeatureExtractor(select_axis=select_axis)
        

        for i, (a, c, idxs) in enumerate(zip(axes, t.exclusive, overall_idxs)):
            tsne = TSNE()
            
            for start, end in idxs:
                # only get splits with 2 min of valid values
                if end - start < samples_per_duration:
                    continue
                section = c[start:end]
                _, mag, _ = extractor.extract_features(section)
                mag /= np.max(mag, axis=-1)[..., np.newaxis]
                
                for s in range(mag.shape[0]):
                    if len(t.behaviour[i][start + s*samples_per_duration: start + (s + 1) * samples_per_duration]) > 0:
                        unique, counter = np.unique(t.behaviour[i][start + s*samples_per_duration: start + (s + 1) * samples_per_duration], 
                                                    return_counts=True)
                        features[i].append(mag[s, :])
                        classes[i].append(unique[np.argmax(counter)])
            
    for i in range(len(classes)):
        c = np.array(classes[i])
        transformed = tsne.fit_transform(features[i])
        for ci, cls in enumerate(traj.trajectories[0].behaviour_classes):
            idxs = np.where(c == ci)
            axes[i].scatter(transformed[idxs, 0], transformed[idxs, 1], label=cls)
            
    plt.legend()
    plt.show()
            
    
    
    

def plot_x_y(t_pack: TrajectoryPack,
             ax: Union[plt.Axes, Iterable[plt.Axes]],
             splitter: Splitter = None,
             individuals: List[str] = None,
             nice_plot: bool = True):
    """Plot x and y for all individuals over time

    Args:
        t_pack (TrajectoryPack): Trajectory pack
        ax (Union[plt.Axes, Iterable[plt.Axes]]): Single Axes ot multiple to plot onto
        splitter (Splitter, optional): Splitter to split trajectories with. Defaults to None.
        individuals (List[str], optional): Names of all individuals. Defaults to None.
        nice_plot (bool, optional): Plot a bit nicer. Defaults to True.
    """
    individuals = individuals if individuals is not None else t_pack.trajectories[0].names
    if isinstance(ax, abc.Iterable):
        assert len(ax) == len(individuals), "Number of axes has to be the same as the number of individuals. Got {} and expected {}".format(
            len(ax), len(individuals))
        labeled = [set() for _ in ax]
    else:
        labeled = set()

    if splitter is not None:
        overall_idxs = [
            splitter.split_exclusive(
                t.exclusive_times,
                t.exclusive) for t in t_pack.trajectories]
    else:
        overall_idxs = [[[(0, len(t.exclusive_times))] for _ in t.names]
                        for t in t_pack.trajectories]

    for traj, idxs_for_name in zip(t_pack.trajectories, overall_idxs):
        for i, (n, c, idxs) in enumerate(zip(traj.names, traj.exclusive, idxs_for_name)):
            if individuals is not None and n not in individuals:
                continue
            if isinstance(ax, abc.Iterable):
                ax_ = ax[i]
                l_ = labeled[i]
            else:
                ax_ = ax
                l_ = labeled
            for j in idxs:
                c_split = c[j[0]:j[1]]
                x_time = traj.exclusive_times[j[0]:j[1]]
                if n in l_:
                    to_label = False
                else:
                    to_label = True
                    if isinstance(ax, abc.Iterable):
                        labeled[i].add(n)
                color_x = t_pack.COLOR_CYCLE[(i * 2) % len(t_pack.COLOR_CYCLE)]
                color_y = t_pack.COLOR_CYCLE[(i * 2 + 1) % len(t_pack.COLOR_CYCLE)]
                idx = c_split[:, 0] > 0

                if to_label:
                    ax_.plot(x_time[idx], c_split[idx, 0], label="{} x".format(
                        n), linewidth=1.0, color=color_x)
                    ax_.plot(x_time[idx], c_split[idx, 1], label="{} y".format(
                        n), linewidth=1.0, color=color_y)
                else:
                    ax_.plot(x_time[idx], c_split[idx, 0], linewidth=1.0, color=color_x)
                    ax_.plot(x_time[idx], c_split[idx, 1], linewidth=1.0, color=color_y)

    if nice_plot:
        if isinstance(ax, abc.Iterable):
            for ax_ in ax:
                ax_.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
                #ax_.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y %H:%M:%S")))
                ax_.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y\n%H:%M:%S")))
                ax_.set_ylabel("position [m]")
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
            #ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y %H:%M:%S")))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y\n%H:%M:%S")))
            ax.set_ylabel("position [m]")


def plot_x_y_raw(t_pack: TrajectoryPack,
                 ax: Union[plt.Axes, Iterable[plt.Axes]],
                 individuals: List[str] = None,
                 nice_plot: bool = True):
    """Plot raw observations over time as a scatter plot

    Args:
        t_pack (TrajectoryPack): Trajectory pack
        ax (Union[plt.Axes, Iterable[plt.Axes]]): Single Axes or multiple
        individuals (List[str], optional): Names of all individuals. Defaults to None.
        nice_plot (bool, optional): Plot a bit nicer. Defaults to True.
    """
    individuals = individuals if individuals is not None else t_pack.trajectories[0].names
    if isinstance(ax, abc.Iterable):
        assert len(ax) == len(individuals), "Number of axes has to be the same as the number of individuals. Got {} and expected {}".format(
            len(ax), len(individuals))
        labeled = [set() for _ in ax]
    else:
        labeled = set()

    for traj in t_pack.trajectories:
        x = {i: [] for i in individuals}
        y = {i: [[], []] for i in individuals}
        for time_stamp, obs in zip(traj.raw_times, traj.raw):
            for n in traj.names:
                o = [x_[0] for x_ in obs if x_[1][1] == n]
                x[n].extend([time_stamp for _ in o])
                y[n][0].extend([x_[0] for x_ in o])
                y[n][1].extend([x_[1] for x_ in o])

        for i, n in enumerate(traj.names):
            if isinstance(ax, abc.Iterable):
                ax_ = ax[i]
                l_ = labeled[i]
            else:
                ax_ = ax
                l_ = labeled

            color_x = t_pack.COLOR_CYCLE[(i * 2) % len(t_pack.COLOR_CYCLE)]
            color_y = t_pack.COLOR_CYCLE[(i * 2 + 1) % len(t_pack.COLOR_CYCLE)]

            if n not in l_:
                ax_.scatter(x[n], y[n][0], label="{} x".format(n), color=color_x, s=0.5)
                ax_.scatter(x[n], y[n][1], label="{} y".format(n), color=color_y, s=0.5)
                if isinstance(ax, abc.Iterable):
                    labeled[i].add(n)
                else:
                    labeled.add(n)
            else:
                ax_.scatter(x[n], y[n][0], color=color_x, s=0.5)
                ax_.scatter(x[n], y[n][1], color=color_y, s=0.5)

    if nice_plot:
        if isinstance(ax, abc.Iterable):
            for ax_ in ax:
                ax_.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
                ax_.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y\n%H:%M:%S")))
                ax_.set_ylabel("position [m]")
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: datetime.fromtimestamp(x).strftime("%d.%m.%y\n%H:%M:%S")))
            ax.set_ylabel("position [m]")


def plot_over_time(t_pack: TrajectoryPack,
                   overlay: np.ndarray,
                   individuals: List[str] = None,
                   show_interval: int = 60,
                   step_size: int = 1,
                   fig_params: dict = None,
                   nice_plot: bool = True) -> None:
    """Plot a dynamic animation of the polar bear movement onto the overlay

    Args:
        t_pack (TrajectoryPack): Trajectory pack
        overlay (np.ndarray): Overlay
        individuals (List[str], optional): Name sof the individuals. Defaults to None.
        show_interval (int, optional): Interval to show tail. Defaults to 60.
        step_size (int, optional): Step size. Defaults to 1.
        fig_params (dict, optional): Parameter for the figure. Defaults to None.
        nice_plot (bool, optional): Nicer plot. Defaults to True.
    """
    fig_params = {} if fig_params is None else fig_params

    fig, ax = plt.subplots(1, 1, **fig_params)

    ax.imshow(overlay)

    ax.set_xlim(1000, 3300)
    ax.set_ylim(2100, 1000)

    for traj in t_pack.trajectories:

        lines = {n: None for n in traj.names}
        dots = {n: None for n in traj.names}
        text = {"text": None}

        def animate(start_end):
            start = start_end[0]
            end = start_end[1]
            start_idx = np.argmax(traj.exclusive_times > start)
            if traj.exclusive_times[-1] <= end:
                end_idx = len(traj.exclusive_times)
            else:
                end_idx = np.argmax(traj.exclusive_times > end)
            if start_idx == end_idx:
                return []
            for i, (n, c) in enumerate(zip(traj.names, traj.exclusive)):
                if individuals is not None and n not in individuals:
                    continue
                c_split = c[start_idx:end_idx]

                color = t_pack.COLOR_CYCLE[i % len(t_pack.COLOR_CYCLE)]

                idx = c_split[:, 0] > 0
                if lines[n] is None:
                    lines[n], = ax.plot(c_split[idx, 0], c_split[idx, 1], label=n,
                                        linewidth=0.6, color=color, animated=True)
                else:
                    lines[n].set_xdata(c_split[idx, 0])
                    lines[n].set_ydata(c_split[idx, 1])
                if dots[n] is None:
                    if len(idx) > 0 and idx[-1]:
                        dots[n] = ax.scatter(
                            c_split[-1, 0],
                            c_split[-1, 1],
                            marker="x", color=color, animated=True)
                    else:
                        dots[n] = None
                else:
                    if len(idx) > 0 and idx[-1]:
                        dots[n].set_offsets(c_split[-1])
                    else:
                        dots[n] = None
            if text["text"] is None:
                text["text"] = plt.text(1050, 1050, datetime.fromtimestamp(
                    traj.exclusive_times[start_idx]).ctime(), size=20)
            else:
                text["text"].set_text(
                    datetime.fromtimestamp(
                        traj.exclusive_times[start_idx]).ctime())

            lines_to_draw = [l for l in lines.values() if l is not None]
            dots_to_draw = [d for d in dots.values() if d is not None]

            return [*lines_to_draw, *dots_to_draw, text["text"]]

        a = ani.FuncAnimation(fig,
                              animate,
                              zip(np.arange(traj.exclusive_times[0],
                                            traj.exclusive_times[-1] - show_interval,
                                  step_size),
                                  np.arange(traj.exclusive_times[0] + show_interval,
                                            traj.exclusive_times[-1],
                                  step_size)),
                              interval=100,
                              blit=True)
        if nice_plot:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

        plt.show()


def plot_features_over_time(t_pack: TrajectoryPack,
                            feature_extractor: FeatureExtractor,
                            splitter: Splitter,
                            individuals: List[str] = None,
                            show_interval: int = 60,
                            step_size: int = 1,
                            min_show: float = 10,
                            fig_params: dict = None,
                            nice_plot: bool = True) -> None:
    """Animate the x and y coordiantes over time and show fft for a window over time

    Args:
        t_pack (TrajectoryPack): Trajectory pack
        feature_extractor (FeatureExtractor): Feature extractor to use
        splitter (Splitter): Splitter to split with
        individuals (List[str], optional): Names of the individuals. Defaults to None.
        show_interval (int, optional): Interval to use as a window. Defaults to 60.
        step_size (int, optional): Step size. Defaults to 1.
        min_show (float, optional): Minimal time frame to show. Defaults to 10.
        fig_params (dict, optional): Parameters for the figure. Defaults to None.
        nice_plot (bool, optional): Nicer plot. Defaults to True.
    """
    fig_params = {} if fig_params is None else fig_params
    individuals = individuals if individuals is not None else t_pack.trajectories[0].names

    for traj in t_pack.trajectories:

        splits = splitter.split_exclusive(traj.exclusive_times, traj.exclusive)

        for name_idx, splits_for_name in enumerate(splits):

            for split in splits_for_name:

                if traj.exclusive_times[split[1] - 1] - traj.exclusive_times[split[0]] < min_show:
                    continue

                fig, ax = plt.subplots(2, 1, **fig_params)

                ax[0].set_title(traj.names[name_idx])

                lines_t = {"x": None, "y": None}
                lines_f = {"x": None, "y": None}
                text = {"text": None}

                def animate(start_end):
                    start = start_end[0]
                    end = start_end[1]
                    start_idx = np.argmax(traj.exclusive_times > start)
                    if traj.exclusive_times[-1] <= end:
                        end_idx = len(traj.exclusive_times)
                    else:
                        end_idx = np.argmax(traj.exclusive_times > end)
                    if start_idx == end_idx or min_show - \
                            (traj.exclusive_times[end_idx] - traj.exclusive_times[start_idx]) < 1e-7:
                        return []

                    c_split = traj.exclusive[name_idx][start_idx:end_idx]
                    time_split = traj.exclusive_times[start_idx:end_idx]

                    x_features, features = feature_extractor.extract_features(c_split)

                    color_x = t_pack.COLOR_CYCLE[0 + 2 * name_idx]
                    color_y = t_pack.COLOR_CYCLE[1 + 2 * name_idx]

                    if lines_t["x"] is None:
                        lines_t["x"], = ax[0].plot(time_split, c_split[:, 0], label="x",
                                                   linewidth=0.6, color=color_x, animated=True)
                        lines_t["y"], = ax[0].plot(time_split, c_split[:, 1], label="y",
                                                   linewidth=0.6, color=color_y, animated=True)
                    else:
                        lines_t["x"].set_xdata(time_split)
                        lines_t["x"].set_ydata(c_split[:, 0])
                        lines_t["y"].set_xdata(time_split)
                        lines_t["y"].set_ydata(c_split[:, 1])

                    if lines_f["x"] is None:
                        lines_f["x"], = ax[1].plot(x_features, features[:, 0], label="x",
                                                   linewidth=0.6, color=color_x, animated=True)
                        lines_f["y"], = ax[1].plot(x_features, features[:, 1], label="y",
                                                   linewidth=0.6, color=color_y, animated=True)
                        ax[0].legend()
                        ax[1].legend()
                    else:
                        lines_f["x"].set_xdata(x_features)
                        lines_f["x"].set_ydata(features[:, 0])
                        lines_f["y"].set_xdata(x_features)
                        lines_f["y"].set_ydata(features[:, 1])

                    ax[0].set_xlim(np.min(time_split), np.max(time_split))
                    ax[0].set_ylim(np.min(c_split), np.max(c_split))
                    ax[1].set_xlim(0, 2)
                    ax[1].set_ylim(np.min(features), np.max(features))

                    lines_t_to_draw = [l for l in lines_t.values() if l is not None]
                    lines_f_to_draw = [l for l in lines_f.values() if l is not None]

                    return [*lines_t_to_draw, *lines_f_to_draw]

                a = ani.FuncAnimation(
                    fig, animate,
                    zip(np.arange(traj.exclusive_times[split[0]],
                        traj.exclusive_times[split[1] - 1] - show_interval,
                        step_size),
                        np.arange(traj.exclusive_times[split[0]] + show_interval,
                                  traj.exclusive_times[split[1] - 1],
                                  step_size)),
                    interval=100, blit=True, repeat=False)

                if nice_plot:
                    ax[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x * MPP)))
                    ax[0].set_xlabel("time [s]")
                    ax[0].set_ylabel("position [m]")

                plt.show()

                while plt.fignum_exists(fig.number):
                    time.sleep(0.1)
