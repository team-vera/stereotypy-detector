import argparse
import sys
import logging

sys.path.append("..")
import cv2
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
    parser.add_argument("--ex_out",
                        type=str,
                        default=None,
                        nargs="*",
                        help="Paths to save trajectory in exclusive format to. \
                            Depending on the processing must be either a single path or as many paths as given by --traj")
    parser.add_argument("--raw_out",
                        type=str,
                        default=None,
                        nargs="*",
                        help="Paths to save trajectory in raw format to. \
                            Depending on the processing must be either a single path or as many paths as given by --traj")
    parser.add_argument("--overlay",
                        type=str,
                        required=True,
                        help="Path to overlay")
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


    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log.upper()))
    
    if args.ex_out is not None:
        assert len(args.ex_out) == 1 or len(args.traj) == len(args.ex_out), \
            "Number of exclusive trajectory outputs must be either 1 or equal to the number of trajectories. \
                Got {} trajectories and {} outputs".format(len(args.traj), len(args.ex_out))
    if args.raw_out is not None:
        assert len(args.raw_out) == 1 or len(args.traj) == len(args.raw_out), \
            "Number of raw trajectory outputs must be either 1 or equal to the number of trajectories. \
                Got {} trajectories and {} outputs".format(len(args.traj), len(args.ex_out))
    
    overlay = cv2.imread(args.overlay)[:, :, ::-1]

    t = TrajectoryPack(traj=args.traj,
                       names=POLAR_NAMES,
                       overlay_shape=overlay.shape[:2],
                       format=args.format)

    n = 1
    
    t.fuse(RawFuser(12.5 / n))
    
    if n > 1:
        t.resample(n)
        

    num_particles = 500

    
    t.to_exclusive(ParticeFilter(
        ps.MomentumParticleSampler(1, 0.99),
        pr.L2ParticleResampler(),
        pt.DelayParticleTracker(num_particles, 260),
        max_interp=20,
        max_fill=20,
        min_reg=2,
        num_particles=num_particles,
        debug_plot=args.debug,
        ident_threshold=0.99,
        fps=12.5 / n,
        mp_inner=not args.debug,
        mp_outer=not args.debug,
        processes_inner=4,
        processes_outer=4
    ), 4 if not args.debug else 1)
    
    # t.apply_to_exclusive(ParticeFilter(
    #     ps.MomentumParticleSampler(1, 0.99),
    #     pr.SoftmaxParticleResampler(),
    #     pt.DelayParticleTracker(num_particles, 13),
    #     max_interp=1,
    #     max_fill=1,
    #     min_reg=1,
    #     num_particles=num_particles,
    #     debug_plot=args.debug,
    #     fps=12.5 / n
    #     mp_inner=True,
    #     mp_outer=True,
    #     processes_inner=4,
    #     processes_outer=4
    # ), 4)
    
    if args.ex_out is not None:
        t.save_traj(args.ex_out, "exclusive")
    if args.raw_out is not None:
        t.save_traj(args.raw_out, "raw")

if __name__ == "__main__":
    main()
