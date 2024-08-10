import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torchvision.transforms.functional import InterpolationMode
import torch
from typing import List


def get_transform() -> transforms.Compose:
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return t

class PaddedResize():
    def __init__(self, size: List[int] = [384, 640]) -> None:
        self._size = torch.Tensor(size).to(dtype=int)
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_shape = torch.Tensor(list(img.shape[-2:]))
        scale_factor = torch.min(self._size / img_shape)
        des_size = img_shape * scale_factor
        des_size = des_size.to(dtype=int)
        img = functional.resize(img, des_size.tolist(), interpolation=InterpolationMode.BILINEAR)
        pad_left = torch.div((self._size - des_size), 2, rounding_mode='floor')
        pad_right = (self._size - (des_size + pad_left)).to(dtype=int).tolist()
        pad_left = pad_left.to(dtype=int).tolist()
        img = functional.pad(img, [*pad_left[::-1], *pad_right[::-1]])
        if img.shape[-2] != self._size[0] or img.shape[-1] != self._size[1]:
            img = functional.resize(img, self._size.tolist(), interpolation=InterpolationMode.BILINEAR)
        return img


def get_resize_transform(input_size=224, augment: bool = False, normalize: bool = True, tensor=False) -> transforms.Compose:
    """Get transform for resizing and augmenting, if desired

    Args:
        input_size (int, optional): Input size to resize to. Defaults to 224.
        augment (bool, optional): Do random affine transforms. Defaults to False.
        normalize (bool, optional): Apply normalization. Defaults to True.
        tensor (bool, optional): Set to true, if tensors will be transformed

    Returns:
        transforms.Compose: Transformation object
    """
    t_list = []
    if augment:
        t_list.append(transforms.RandomAffine(degrees=15,
                                              translate=(0.1, 0.1),
                                              scale=(0.5, 2),
                                              shear=15))
    if not tensor:
        t_list.append(transforms.ToTensor())
    t_list.append(PaddedResize([input_size, input_size]))
    if normalize:
        t_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    t = transforms.Compose(t_list)

    return t


def padded_batch_stack(batch):
    max_width = max([x[0].shape[1] for x in batch])
    max_height = max([x[0].shape[2] for x in batch])
    x_tensor = torch.zeros(
        [len(batch), batch[0][0].shape[0], max_width, max_height], dtype=torch.float32)
    y_true = torch.zeros(len(batch), dtype=torch.int64)
    for i, x in enumerate(batch):
        x_tensor[i, :, :x[0].shape[1], :x[0].shape[2]] = x[0]
        y_true[i] = x[1]
    return x_tensor, y_true
