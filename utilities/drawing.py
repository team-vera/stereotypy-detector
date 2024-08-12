import numpy as np
import cv2
from .enums import *
from typing import List


def draw_bboxes(img: np.ndarray, bboxes: dict, color_coding: str = "RGB") -> np.ndarray:
    """Draw a bounding boxes on the given image and return it

    Parameters
    ----------
    img : np.ndarray
        Image to draw onto
    bboxes : dict
        Bounding boxes as a dict in format {<subject_1>: <bbox_>, <subject_2>: <bbox_>, ...}
    color_coding : str, optional
        Color coding of the image, by default 'RGB'

    Returns
    -------
    np.ndarray
        The given image with bounding boxes drawn
    """
    if bboxes == None:
        return img

    for subject in bboxes:
        bbox = bboxes[subject]
        # left upper point
        pt1 = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)]
        # right lower point
        # pt2 = [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]

        # new bbox for opencv
        new_bbox = [*pt1, *bbox[2:]]

        color_mapping = POLAR_COLOR_MAPPING[color_coding]

        img = cv2.rectangle(img, new_bbox, color_mapping[subject], thickness=3)

    return img


def draw_bboxes_raw(img: np.ndarray, bboxes: List[List], color_coding: str = "RGB") -> np.ndarray:
    """Draw a bounding boxes on the given image and return it

    Parameters
    ----------
    img : np.ndarray
        Image to draw onto
    bboxes : List[List]
        Bounding boxes in list format: [[x, y, w, h, confidence, class], ...] 
    color_coding : str, optional
        Color coding of the image, by default 'RGB'

    Returns
    -------
    np.ndarray
        The given image with bounding boxes drawn
    """
    if bboxes == None:
        return img

    for bbox in bboxes:
        # left upper point
        pt1 = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)]
        # right lower point
        pt2 = [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]


        if color_coding == "RGB":
            color = [255, 125, 0]
        else:
            color = [0, 125, 255]
            
        if bbox[5] in POLAR_BGR_COLOR_MAPPING:
            color_mapping = POLAR_COLOR_MAPPING[color_coding]
            
            color = color_mapping[bbox[5]]

        cv2.rectangle(img, pt1=pt1, pt2=pt2, color=color,
                      thickness=1, lineType=cv2.LINE_AA)
        bg_pt2 = [pt1[0] + 71, pt1[1] - 20]
        cv2.rectangle(img, pt1=pt1, pt2=bg_pt2, color=color, thickness=-1)

        txt_pt1 = [pt1[0] + 1, pt1[1] - 11]
        txt_pt2 = [pt1[0] + 1, pt1[1] - 1]
        cv2.putText(img,
                    "{} {:.1f}%".format(bbox[5], bbox[6] * 100),
                    txt_pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    [255, 255, 255],
                    thickness=1)
        cv2.putText(img,
                    "BBox: {:.1f}%".format(bbox[4] * 100),
                    txt_pt2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    [255, 255, 255],
                    thickness=1)

    return img
