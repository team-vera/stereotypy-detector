import os
import argparse
import sys
import logging
sys.path.append("..")

import cv2

from trajectories.trajectory_utils.trajectory_pack import TrajectoryPack, Trajectory
from trajectories.trajectory_utils.behaviour_classifier import MultiClassBehaviourClassifier, HeuristicStereoClassifier
from trajectories.trajectory_utils.feature_extractor import MixedFreqFeatureExtractor
from trajectories.trajectory_utils.splitters import PatienceSplitter
from utilities.enums import POLAR_NAMES

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--traj",
                        type=str,
                        nargs="*",
                        help="Trajectories to load")
    parser.add_argument("--out",
                        type=str,
                        required=True,
                        help="Path to write annotated trajectories to. Filenames will be kept.")
    parser.add_argument("--overlay",
                        type=str,
                        default="../images/PB_Maps_Overlay.png",
                        help="Path to enclosure overlay")
    parser.add_argument("--clf",
                        type=str,
                        default=None,
                        help="Classifier pipeline to use")
    parser.add_argument("--log",
                        type=str,
                        choices=["critical",
                                 "error",
                                 "warning",
                                 "info",
                                 "debug"],
                        default="info",
                        help="Log level to use")
    parser.add_argument("--stereo-only",
                        action="store_true",
                        help="Evaluate stereotypy vs all instead of stereo vs movement vs resting")
    parser.add_argument("--stupid",
                        action="store_true",
                        help="Use stupid heuristic classifier")

    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log.upper()))
    
    overlay = cv2.imread(args.overlay)
    
    traj = TrajectoryPack(args.traj,
                          POLAR_NAMES,
                          overlay.shape[:2])
    
    if not args.stupid:
        
        fe = MixedFreqFeatureExtractor(
            overlap=(0, 0)
        )
        
        bc = MultiClassBehaviourClassifier(
            PatienceSplitter(coordinate_patience=1, time_patience=1),
            fe,
            args.clf,
            stereo_only=args.stereo_only)
    else:
        bc = HeuristicStereoClassifier(PatienceSplitter(coordinate_patience=1, time_patience=1))
    
    for t in args.traj:
        traj = Trajectory(POLAR_NAMES)
        traj.load_traj(t, POLAR_NAMES, "exclusive", overlay_shape=overlay.shape[:2])
        
        bc.apply_exclusive(traj)

        traj.save_traj(os.path.join(args.out, os.path.basename(t)), 
                       "exclusive", 
                       overlay_shape=overlay.shape[:2],
                       save_behaviour=True)
        

if __name__ == "__main__":
    main()