import numpy as np
from typing import List, Union
from detection.detection_utils.yolo_wrapper import Yolo
import torch
from utilities.bbox_utils import crop_bboxes_raw
from identification.identification_utils.data_loading import get_resize_transform
from utilities.enums import POLAR_NAMES

class Detector():
    def __init__(self,
                 det_weights: str,
                 device: str,
                 ident_weights: str = None,
                 det_conf: float = 0.5,
                 ident_conf: float = 0.6,
                 det_one_class: bool = True,
                 ident_batch: int = 16):
        """Detector for combined detecting and identifying

        Args:
            det_weights (str): Weights for detection network
            device (str): Device to operate on
            ident_weights (str, optional): Weights for identification network. If None, no separate identification is performed. Defaults to None.
            det_conf (float, optional): Detection confidence threshold. Defaults to 0.5.
            ident_conf (float, optional): Identification confidence threshold. Defaults to 0.6.
            det_one_class (bool, optional): Indicator, if detection network detects one class (True) or multiple (False). Defaults to True.
            ident_batch (int, optional): Max batch size for identification network. Defaults to 16.
        """
        self.od = Yolo(det_weights,
                       device,
                       det_conf)

        self._ident_batch = ident_batch

        if ident_weights is not None:
            self.ident = torch.load(ident_weights, map_location=device)
            self.ident.eval()
            self.ident_transform = get_resize_transform()
            self.ident_tensor_transform = get_resize_transform(tensor=True)
        else:
            self.ident = None

        self._device = device
        self._ident_conf = ident_conf
        self._det_one_class = det_one_class

    def detect(self, img: Union[np.ndarray, List[np.ndarray]]) -> List[List[List]]:
        """Apply detection to a batch of images

        Args:
            img (np.ndarray): Batch of images with <b, h, w, c> format in RGB

        Returns:
            List[List[List]]: List of bounding boxes for every sample in format [[[x, y, h, w, det_conf, class, ident_conf], ...], ...]
        """
        assert len(img) > 0, "Must get at least one image"
        if not isinstance(img, list):
            assert len(img.shape) == 4, "Input has to be given in batch format. Got {}".format(img.shape)
        else:
            assert len(img[0].shape) == 3, "Input samples gave to have three dimensions. Got {} ({})".format(
                len(img[0].shape),
                img[0].shape
            )

        # detect polar bears
        bboxes = self.od.detect(img)

        # do a separate identification
        if self.ident is not None:
            cropped_images = []
            indices = []
            # collect all bounding boxes and remember where they belong
            for i, b in enumerate(bboxes):
                cis = crop_bboxes_raw(img[i], b, tensor=isinstance(img[i], torch.Tensor))
                cropped_images.extend(cis)
                indices.extend([(i, j) for j in range(len(cis))])
                
            # split bounding boxes into batches of desired size
            cropped_images = [cropped_images[i:i+self._ident_batch] for i in range(0, len(cropped_images), self._ident_batch)]
            indices = [indices[i:i+self._ident_batch] for i in range(0, len(indices), self._ident_batch)]

            # Process every batch and assign result to correct bbox
            for c, ind in zip(cropped_images, indices):
                if len(c) > 0:
                    ident_pred = self.identify(c)
                    
                    for ((i, j), pred) in zip(ind, ident_pred):

                        pred_thresh = np.where(pred > self._ident_conf, 1, 0)

                        if 1 not in pred_thresh:
                            class_name = "Unknown"
                        else:
                            class_name = POLAR_NAMES[np.argmax(pred)]

                        bboxes[i][j] = [*bboxes[i][j][:5], class_name, float(np.max(pred))]
        # don't do separate identification
        else:
            # convert classes internally to names
            for i in range(len(bboxes)):
                for j in range(len(bboxes[i])):
                    # get class string
                    if not self._det_one_class:
                        if bboxes[i][j][4] > self._ident_conf:
                            class_name = POLAR_NAMES[bboxes[i][j][5]]
                        else:
                            class_name = "Unknown"
                    else:
                        class_name = "PolarBear"

                    bboxes[i][j] = [*bboxes[i][j][:5],
                                    class_name,
                                    bboxes[i][j][4]]

        return bboxes

    def identify(self, img: List[np.ndarray]) -> np.ndarray:
        """Identify the subjects in a batch of images

        Args:
            img (List[np.ndarray]): List of images with <h, w, c> format in RGB

        Returns:
            np.ndarray: Results in format <b, num_classes>
        """
        if isinstance(img[0], torch.Tensor):
            cropped_images = torch.stack(
                [self.ident_tensor_transform(c_i) for c_i in img]
            ).to(self._device)
        else:
            cropped_images = torch.stack(
                [self.ident_transform(c_i.copy()) for c_i in img]
            ).to(self._device)

        return self.ident(cropped_images).cpu().detach().numpy()
