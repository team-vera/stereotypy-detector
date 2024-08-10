import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as functional
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from typing import List, Union

try:
    from utils.general import scale_coords, non_max_suppression
    from utils.augmentations import letterbox
except ModuleNotFoundError as e:
    print("Could not find Yolov5 module. Please make sure to use \
        `sys.path.append(<path_to_yolov5>)` \
            before importing {}.\nOriginal Error: {}".format(__name__, e))
    exit()

class _PaddedResize():
    def __init__(self, size: List[int] = [384, 640]) -> None:
        self._size = torch.Tensor(size).to(dtype=int)
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # scale image the least amount possible
        img_shape = torch.Tensor(list(img.shape[-2:]))
        scale_factor = (torch.min(self._size / img_shape))
        des_size = img_shape * scale_factor
        des_size = des_size.to(dtype=int)
        img = functional.resize(img, des_size.tolist(), interpolation=InterpolationMode.BILINEAR)
        pad_left = torch.div((self._size - des_size), 2, rounding_mode='floor')
        pad_right = (self._size - (des_size + pad_left)).to(dtype=int).tolist()
        pad_left = pad_left.to(dtype=int).tolist()
        # pad with 114
        img = functional.pad(img, [*pad_left[::-1], *pad_right[::-1]], fill=0.4453125)
        if img.shape[-2] != self._size[0] or img.shape[-1] != self._size[1]:
            img = functional.resize(img, self._size.tolist(), interpolation=InterpolationMode.BILINEAR)
        return img

class Yolo:
    # currently too inefficient, since casting to tensor is damn slow
    _array_transform = transforms.Compose([
        transforms.ToTensor(),
        _PaddedResize([384, 640])
    ])
    _tensor_transform = transforms.Compose([
        _PaddedResize([384, 640])
    ])

    def __init__(self, 
                 weights: str,
                 device: str,
                 conf: float = 0.5):
        """Wrapper for Yolov5

        Args:
            weights (str): Weights to load
            device (str): Device to use for detection
            conf (float, optional): Confidence threshold for detection. Defaults to 0.5.
        """
        self.device = device
        self.conf = conf

        ckpt = torch.load(weights, map_location=device)  # load model
        self.model = ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()

        self.model.inplace = True  # backward compatibility 1.7

        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names  # get class names

        if device != "cpu":
            self.model(torch.zeros(1, 3, 384, 640).to(
                device).type_as(next(self.model.parameters())))  # run once

    def detect(self, img: Union[np.ndarray, List[np.ndarray]]) -> List[List]:
        """Detect on a single image

        Args:
            img (np.ndarray | List[np.ndarray]): Batch as list of images or tensor expected with 4 dimensins and format <b, h, w, c>

        Returns:
            List[List]: Bounding boxes in format [[[x, y, h, w, conf, class], ...], ...] -> Batch[Detections[Boundingboxes]]
        """
        if isinstance(img, list):
            assert len(img[0].shape) == 3, "Expected 3 dimensions for every sample in the batch, but got {} ({})".format(
                len(img[0].shape), img[0].shape)
            if isinstance(img[0], torch.Tensor):
                old_shape = img[0].shape[1:]
            else:
                old_shape = img[0].shape[:2]
        else:
            assert len(img.shape) == 4, "Expected 4 dimensions, but got {} ({})".format(
                len(img.shape), img.shape)
            if isinstance(img, torch.Tensor):
                old_shape = img.shape[2:]
            else:
                old_shape = img.shape[1:3]

        # transform image to tensor (if necessary) and apply padded resize
        if isinstance(img, np.ndarray):
            img = torch.stack([transforms.ToTensor()(letterbox(i, [384, 640])[0]) for i in img]).to(self.device)
        elif isinstance(img, torch.Tensor):
            img = self._tensor_transform(img).to(self.device)
        elif isinstance(img, list):
            if isinstance(img[0], torch.Tensor):
                img = torch.stack([self._tensor_transform(i) for i in img]).to(self.device)
            else:
                img = torch.stack([transforms.ToTensor()(letterbox(i, [384, 640])[0]) for i in img]).to(self.device)

        # forward through network
        pred = self.model(img)[0]

        img_shape = img.shape

        pred = non_max_suppression(pred.cpu().detach(), self.conf)
    
        out_list = []

        # go throuch every batch element
        for det in pred:
            det_list = []
            # scale coordinates to correct values
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], old_shape).round()

            # iterate through bboxes and transform to x, y, h, w
            for *xyxy, conf, cls in reversed(det):
                if conf > self.conf:
                    xyhw = [
                        float((xyxy[2] + xyxy[0]) / 2),
                        float((xyxy[3] + xyxy[1]) / 2),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1]),
                    ]

                    det_list.append([*xyhw, float(conf), int(cls)])

            out_list.append(det_list)

        return out_list
