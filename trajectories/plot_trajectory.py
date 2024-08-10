import argparse
import sys
import logging
import os
import copy

sys.path.append("..")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utilities.enums import POLAR_NAMES

from trajectories.trajectory_utils.filters import ParticeFilter
import trajectories.trajectory_utils.mergers as mergers
import trajectories.trajectory_utils.particle_resamplers as pr
import trajectories.trajectory_utils.particle_samplers as ps
import trajectories.trajectory_utils.particle_tracker as pt
import trajectories.trajectory_utils.sanity_checker as sc
import trajectories.trajectory_utils.splitters as splitters
from trajectories.trajectory_utils.trajectory_pack import TrajectoryPack
from trajectories.trajectory_utils.fuser import RawFuser
from trajectories.trajectory_utils.feature_extractor import FFTFeatureExtractor
from trajectories.trajectory_utils.plotting import (plot, plot_x_y, plot_x_y_raw, plot_features_over_time, plot_over_time, 
                                                    plot_x_y_num_traj, plot_cont_gt, plot_wfft, plot_wfft_tsne, plot_inst_gt, plot_behaviour,
                                                    plot_behaviour_diff)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--traj",
                        type=str,
                        nargs="*",
                        required=True,
                        help="Path to trajectory")
    parser.add_argument("--format",
                        type=str,
                        choices=TrajectoryPack.FORMATS,
                        default="raw",
                        help="Format of trajectory")
    parser.add_argument("--overlay",
                        type=str,
                        required=True,
                        help="Path to overlay")
    parser.add_argument("--out",
                        type=str,
                        default=None,
                        help="Directory to plot to. If not given, plot will be shown")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug plots for particle filter")
    parser.add_argument("--log",
                        type=str,
                        choices=["critical", 
                                 "error",
                                 "warning",
                                 "info",
                                 "debug"],
                        default="info",
                        help="Log level to use")
    parser.add_argument("--inst-gt",
                        type=str,
                        default=None,
                        help="Path to instantaneous ground truth")
    parser.add_argument("--cont-gt",
                        type=str,
                        default=None,
                        help="Path to cont observation ground truth directory")
    parser.add_argument("--plot-behaviour",
                        action="store_true",
                        help="Set, if behaviour ground truth read from exclusive annotation should be plottet")
    parser.add_argument("--pf",
                        action="store_true",
                        help="Plot features over time at the end")
    parser.add_argument("--wfft",
                        action="store_true",
                        help="Plot windowed FFT of signal")
    parser.add_argument("--tsne",
                        action="store_true",
                        help="Plot TSNE of stft of trajectory")
    parser.add_argument("--plot-num",
                        action="store_true",
                        help="Plot the number of trajectories as background")
    parser.add_argument("--stereo-only",
                        action="store_true",
                        help="Join classes to stereo vs rest")
    parser.add_argument("--plot-b-diff",
                        action="store_true",
                        help="Plot behaviour difference (stereo vs rest only)")

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.getLevelName(args.log.upper()))

    overlay = cv2.imread(args.overlay)[:, :, ::-1]

    t = TrajectoryPack(traj=args.traj,
                       names=POLAR_NAMES,
                       overlay_shape=overlay.shape[:2],
                       format=args.format)
    
    if args.format == "exclusive":
        print("Overall time: {:.0f}s".format(
            sum([t_.exclusive_times[-1] - t_.exclusive_times[0] for t_ in t.trajectories]))
              )

    n = 1
    
    if args.format == "raw":
        fig2, ax2 = plt.subplots(4, 1, figsize=(16, 18), sharex=True, sharey=True)
    else:
        #fig2, ax2 = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True)
        fig2, ax2 = plt.subplots(2, 1, figsize=(9, 5), sharex=True, sharey=True)
    
    if args.cont_gt is None and args.inst_gt is None and not args.plot_behaviour:
        if args.plot_num:
            for a in ax2:
                plot_x_y_num_traj(t, a)
    elif args.plot_behaviour:
        plot_behaviour(t, ax=ax2)
    else:
        if args.plot_b_diff:
            t_cpy = TrajectoryPack(traj=args.traj,
                                   names=POLAR_NAMES,
                                   overlay_shape=overlay.shape[:2],
                                   format=args.format)
        
        if args.inst_gt is not None:
            for t_, tp in zip(t.trajectories, args.traj):
                b_gt_file = os.path.basename(tp).replace("PB_", "")
                t_.load_behaviour(os.path.join(args.inst_gt, b_gt_file),
                                classes_path=os.path.join(args.inst_gt, "classes.txt"),
                                annotation_type="inst",
                                stereo_only=args.stereo_only)
        else:
            for t_, tp in zip(t.trajectories, args.traj):
                b_gt_file = os.path.basename(tp).replace("PB_", "")
                t_.load_behaviour([os.path.join(args.cont_gt, "Nanuq", b_gt_file),
                                os.path.join(args.cont_gt, "Vera", b_gt_file)],
                                classes_path=os.path.join(args.cont_gt, "classes.txt"),
                                stereo_only=args.stereo_only)
        
                
        if len(ax2) == 2:
            if args.plot_b_diff:
                plot_behaviour_diff(t_cpy, t, ax=ax2)
            else:
                plot_behaviour(t, ax=ax2)
        else:
            if args.plot_b_diff:
                plot_behaviour_diff(t_cpy, t, ax=ax2[::2])
                plot_behaviour_diff(t_cpy, t, ax=ax2[1::2])
            else:
                plot_behaviour(t, ax=ax2[::2])
                plot_behaviour(t, ax=ax2[1::2])
    
    if args.format == "raw":
        t.fuse(RawFuser(12.5 / n))
    
    start = 30
    start = 70
    # start = 170
    stop = 90
    # stop = 190
    # t.crop(start * 60, stop * 60)

    if n > 1:
        t.resample(n)
        

    num_particles = 500

    
    if args.format == "raw":
        t.to_exclusive(ParticeFilter(
            # ps.NoiseParticleSampler(3),
            # ps.NaiveParticleSampler(3),
            ps.MomentumParticleSampler(1, 0.99),
            # ps.ObservationParticleSampler(3),
            # ps.UniformParticleSampler(10),
            # pr.GaussianParticleResampler(2),
            pr.L2ParticleResampler(),
            # pr.SoftmaxParticleResampler(),
            pt.DelayParticleTracker(num_particles, 260),
            # pt.DummyMeanParticleTracker(num_particles),
            max_interp=20,
            max_fill=20,
            min_reg=2,
            num_particles=num_particles,
            debug_plot=args.debug,
            ident_threshold=0.99,
            fps=12.5 / n,
            mp_inner=True,
            mp_outer=True,
            processes_inner=4,
            processes_outer=4
        ), 4)
        
    lengths = []
    durations = []
    
    print(args.traj)
    for t_ in t.trajectories:
        ts = t_.get_stats()
        # logging.info(ts)
        print(ts.to_tex()[1])
        lengths.extend([ts.nanuq.length, ts.vera.length])
        durations.append(ts.seconds)
    
    lengths = [l for l in lengths if l != 0]
    if len(lengths) > 0:
        logging.info("Trajectory length | Mean: {:.2f}m | Std: {:.2f}m | Min: {:.2f}m | Max: {:.2f}m".format(
            np.mean(lengths), 
            np.std(lengths),
            np.min(lengths),
            np.max(lengths)))
    if len(durations) > 0:
        logging.info("Trajectory duration | Mean: {:.2f}s | Std: {:.2f}s | Min: {:.2f}s | Max: {:.2f}s".format(
            np.mean(durations), 
            np.std(durations),
            np.min(durations),
            np.max(durations)))

    fig1, ax = plt.subplots(1, 1, figsize=(16, 16))

    plot(t, ax, overlay, splitters.PatienceSplitter(coordinate_patience=1, time_patience=1))

    #ax.set_title("Particle Filter")

    ax.set_xlim(1000, 3300)
    ax.set_ylim(2100, 1000)


    plot_x_y(t, ax2[:2], splitters.PatienceSplitter(coordinate_patience=1, time_patience=1))
    if args.format == "raw":
        plot_x_y_raw(t, ax2[2:])
        ax2[2].set_title("Observations")
        ax2[0].set_title("Particle Filter")
    [a.grid(True) for a in ax2]
    
    by_label = {}
    for a in ax2:
        handles, labels = a.get_legend_handles_labels()
        for l, h in zip(labels, handles):
            if l not in by_label:
                by_label[l] = h
    fig2.legend(by_label.values(), by_label.keys())
    
    fig2.tight_layout()
    
    plt.show()
    
    if args.out is not None:
        if not os.path.isdir(args.out):
            logging.warning("Output path {} is not a directory. WIll not save figure".format(args.out))
        else:
            out_base = os.path.splitext(os.path.basename(args.traj[0]))[0]
            fig2.savefig(os.path.join(args.out, "Traj_{}.png".format(out_base)))
            fig1.savefig(os.path.join(args.out, "Map_{}.png".format(out_base)))
    
    plt.ioff()
        
    if args.wfft:
        fig, ax = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
        
        if (args.n_gt is not None or args.v_gt is not None) and args.gt_classes is None:
            raise AssertionError("Ground truth classes have to be given")
        if args.n_gt is not None:
            plot_cont_gt(args.n_gt, args.gt_classes, ax=ax[0])
        if args.v_gt is not None:
            plot_cont_gt(args.v_gt, args.gt_classes, ax=ax[2])
        
        plot_x_y(t, ax[::2], splitters.PatienceSplitter(coordinate_patience=1, time_patience=1))
        colorbars = []
        for t_ in t.trajectories:
            colorbars.extend(plot_wfft(t_, 
                    splitters.PatienceSplitter(coordinate_patience=1, time_patience=1),
                    ax[1::2],
                    select_axis=False))
        
        fig.colorbar(colorbars[0], ax=ax[3], location="bottom")

        plt.show()
        
    if args.tsne:
        if args.n_gt is None or args.v_gt is None or args.gt_classes is None:
            raise AssertionError("Vera and Nanuq ground truth as well as ground truth classes have to be given")
        for i, (n, v) in enumerate(zip(args.n_gt, args.v_gt)):
            t.trajectories[i].load_behaviour([n, v], classes_path=args.gt_classes)
        plot_wfft_tsne(t, 
                       splitters.PatienceSplitter(coordinate_patience=1, time_patience=1))
        
    
    if args.pf:
        plot_features_over_time(
            t,
            FFTFeatureExtractor(12.5 / n),
            splitters.PatienceSplitter(coordinate_patience=1, time_patience=1),
            show_interval=120,
            step_size=10,
            min_show=120,
            fig_params={"figsize": (16, 8)}
        )
        

    # plot_over_time(t, overlay, fig_params={"figsize": (16, 8)}, show_interval=10)


if __name__ == "__main__":
    main()
