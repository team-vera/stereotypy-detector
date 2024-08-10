import numpy as np
from typing import List


def crop_bboxes(img: np.ndarray, bboxes: dict) -> dict:
    """Crop bounding boxes out of image. Bounding boxes in dict format

    Args:
        img (np.ndarray): Image to crop out of
        bboxes (dict): Bounding box in format {<subject_0>: <bbox_0>, <subject_1> <bbox_1>, ...}

    Returns:
        dict: Cropped images in format {<subject_0>: <img_0>, <subject_1> <img_1>, ...}
    """
    if bboxes == None:
        return img

    cropped_images = {}

    for subject in bboxes:
        bbox = bboxes[subject]
        # left upper point
        pt1 = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)]
        # right lower point
        pt2 = [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]

        # new bbox for opencv
        cropped_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        cropped_images[subject] = cropped_img

    return cropped_images


def crop_bboxes_raw(img: np.ndarray, bboxes: List[List], tensor=False) -> List[np.ndarray]:
    """Crop bounding boxes out of image. Bounding boxes in list format

    Args:
        img (np.ndarray): Image to crop out of
        bboxes (List[List]): Bounding boxes in format [[x, y, h, w, ...], [x, y, h, w, ...], ...]
        tensor (bool, optional): If true, channel first format is assumed

    Returns:
        List[np.ndarray]: List of cropped images
    """
    if bboxes == None:
        return img

    cropped_images = []

    for bbox in bboxes:
        # left upper point
        pt1 = [int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)]
        # right lower point
        pt2 = [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]

        # new bbox for opencv
        if tensor:
            cropped_img = img[:, pt1[1]:pt2[1], pt1[0]:pt2[0]]
        else:
            cropped_img = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

        cropped_images.append(cropped_img)


    return cropped_images


def get_iou(bbox_1: List, bbox_2: List) -> float:
    """Get the Intersection over Union (IoU) of two bounding boxes

    Args:
        bbox_1 (List): First bounding box in format [x, y, h, w, ...]
        bbox_2 (List): Second bounding box in format [x, y, h, w, ...]

    Returns:
        float: Intersection over union.
    """
    x_dist = abs(bbox_1[0] - bbox_2[0])
    y_dist = abs(bbox_1[1] - bbox_2[1])

    if x_dist >= (bbox_1[2] + bbox_2[2]) / 2 or y_dist >= (bbox_1[3] + bbox_2[3]) / 2:
        return 0.0

    inter_h = min(bbox_1[0] + bbox_1[2] / 2, bbox_2[0] + bbox_2[2] / 2) - \
        max(bbox_1[0] - bbox_1[2] / 2, bbox_2[0] - bbox_2[2] / 2)
    inter_w = min(bbox_1[1] + bbox_1[3] / 2, bbox_2[1] + bbox_2[3] / 2) - \
        max(bbox_1[1] - bbox_1[3] / 2, bbox_2[1] - bbox_2[3] / 2)

    intersection = inter_h * inter_w
    union = bbox_1[2] * bbox_1[3] + bbox_2[2] * bbox_2[3] - intersection

    return intersection / union


def get_diff_keypoints(bbox_1: List, bbox_2: List, keypoint: List[float] = [0, 0.5]) -> float:
    """Get the difference vector of two bounding boxes with respect to a specified point. Vector will be bbox_1 - bbox_2.

    Args:
        bbox_1 (List): First bounding box in format [x, y, h, w, ...]
        bbox_2 (List): Second bounding box in format [x, y, h, w, ...]
        keypoint (List[float], optional): Relative keypoint inside the bounding box 

    Returns:
        float: Intersection over union.
    """
    keypoint_1 = np.array([bbox_1[0] + keypoint[0] * bbox_1[2],
                           bbox_1[1] + keypoint[1] * bbox_1[3]])
    keypoint_2 = np.array([bbox_2[0] + keypoint[0] * bbox_2[2],
                           bbox_2[1] + keypoint[1] * bbox_2[3]])
    return keypoint_1 - keypoint_2
