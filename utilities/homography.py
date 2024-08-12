import cv2
from typing import List
import numpy as np
import os
import pickle
import logging
from typing import Callable


class HomographyTransformer():
    def __init__(self, mapping_path: str, keypoint: List[float] = [0, 0.5]):
        # mapping: camera-plane position in enclosure -> homography matirx
        self._transform_mapping = []

        # list of homography matrices in correspondance to the transformation mapping
        self._transformations = []

        # realtive bbox internal position to use for mapping (x ^ ; y >)
        self._keypoint = keypoint

        self._logger = logging.getLogger("HomographyTransformer")

        self.load_transforms(mapping_path)

        self.SPECIAL_TRANSFORMS = {
            # img: map on line from 1718, 683 to 3434, 676
            # enc: map on line from 1275, 1520 to 1622, 1416
            (0, 10): self._get_line_mapping(
                np.array([3434.0, 676.0]),
                np.array([1718.0, 683.0]),
                np.array([1622, 1416]),
                np.array([1275, 1520])),
            # map on line from 1876, 550 to 2243, 533
            # map on line from 2755, 1318 to 2849, 1317
            (2, 14): self._get_line_mapping(
                np.array([2243, 533]),
                np.array([1876, 550]),
                np.array([2849, 1317]),
                np.array([2755, 1318])),
        }

    def transform(self, bboxes: List, enclosure_part: int = 0, failsafe: bool = True) -> List[List[float]]:
        """Transform a list of bounding boxes to 2D-enclosure positions.
        Transformation is done with a position dependent homography transform.

        Args:
            bboxes (List): List of bounding boxes in fromat [[x, y, h, w, ...], ...]
            enclosure_part (int, optional): Index of the enclosure part the boudning boxes are from. Defaults to 0.
            failsafe (bool, optional): If set to true, no error will be thrown from indexes out of range.

        Raises:
            RuntimeError: Error, if no mapping can be found dure to invalid keypoint indices

        Returns:
            List[List[float]]: List of 2D-enclosure positions in format [[x, y], ...]. If in background, None instead of [x, y] for this point.
        """
        enc_positions = []
        for bbox in bboxes:
            # get keypoint for bounding box as homography coordinates p = [x, y, 1]
            keypoint = np.array([int(bbox[0] + self._keypoint[0] * bbox[2]),
                                 int(bbox[1] + self._keypoint[1] * bbox[3]),
                                 1])

            # check for error-prone keypoints
            if not failsafe:
                # check, if keypoint valid
                if 0 > keypoint[0] or keypoint[0] >= self._transform_mapping[enclosure_part].shape[0] or \
                        0 > keypoint[1] or keypoint[1] >= self._transform_mapping[enclosure_part].shape[1]:
                    raise IndexError(
                        "Keypoint {} out of range for transformation mapping with shape {}".format(
                            keypoint, self._transform_mapping[enclosure_part].shape))
            else:
                keypoint[0] = np.maximum(0, np.minimum(
                    keypoint[0], self._transform_mapping[enclosure_part].shape[1] - 1))
                keypoint[1] = np.maximum(0, np.minimum(
                    keypoint[1], self._transform_mapping[enclosure_part].shape[0] - 1))

            h_idx = self._transform_mapping[enclosure_part][keypoint[1], keypoint[0]]

            if h_idx == 0:
                enc_positions.append(None)
                continue
            
            # special line mappings
            if (enclosure_part, h_idx) in self.SPECIAL_TRANSFORMS:
                pos = self.SPECIAL_TRANSFORMS[(enclosure_part, h_idx)](keypoint[:2])
            else:
                # get 3x3 homography matrix
                h = self._transformations[enclosure_part][h_idx - 1]

                # transform with s * p` = h @ p -> p`[2] has to be 1 -> p` / p`[2] for [x`, y`, 1]
                pos = h @ keypoint
                pos = pos / pos[2]

            enc_positions.append(pos[:2])

        return enc_positions

    def load_transforms(self, path: str) -> None:
        """Load the transofmation information form a dedicated directory.

        Format:
            mapping_<n>.png: Mapping of the <n>-th part of the enclosure to a homography matrix
            transformations_<n>.pt: Pickled list of transformation matrices corresponding to the <n>-th part of the enclosure

        Args:
            path (str): Path to the directory containing the transofmrations
        """
        self._logger.debug("Will attempt to load transformations from {}".format(path))
        counter = 1
        while 1:
            mapping_path = os.path.join(path, "mapping_{}.png".format(counter))
            trans_path = os.path.join(path, "transformations_{}.pt".format(counter))
            if not os.path.exists(mapping_path):
                self._logger.info("No more transformations found ({} overall)".format(counter - 1))
                break
            else:
                self._logger.debug("Found tranformation mapping {} at {}".format(counter, mapping_path))
                m = cv2.imread(mapping_path, flags=cv2.IMREAD_GRAYSCALE)
                with open(trans_path, "rb") as f:
                    t = pickle.load(f)

                if m.max() > len(t):
                    raise AssertionError("Maximum value in the mapping ({}) is {}, while only {} transformations are available in {}".format(
                        mapping_path, m.max(), len(t), trans_path))

                self._logger.debug("{} homography matrices found for mapping {}".format(len(t), counter))

                self._transform_mapping.append(m)
                self._transformations.append(t)

            counter += 1

    def _get_line_mapping(self, end_i_line, start_i_line, end_e_line, start_e_line) -> Callable:
        # image line to map onto
        end_minus_start = end_i_line - start_i_line
        end_minus_start_norm = end_minus_start / np.linalg.norm(end_minus_start)

        # enclosure line to map onto
        e_line = end_e_line - start_e_line

        def _map_line(keypoint: np.ndarray) -> np.ndarray:
            # precalculate keypoint stuff
            key_minus_start = keypoint - start_i_line
            key_minus_start_norm = key_minus_start / np.maximum(np.linalg.norm(key_minus_start), 1e-12)

            # calculate angle between end and keypoint over start
            cos_angle = end_minus_start_norm @ key_minus_start_norm

            # project onto image line (relative to start)
            proj_i_keypoint_l2 = np.linalg.norm(key_minus_start) * cos_angle
            
            # project onto enclosure line
            proj_e_keypoint = start_e_line + e_line * proj_i_keypoint_l2 / np.linalg.norm(end_minus_start)
            
            return proj_e_keypoint

        return _map_line
