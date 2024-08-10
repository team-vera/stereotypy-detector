import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


PRETRAINED_MODELS = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152
}


class ClsNet(nn.Module):
    def __init__(self, depth: str):
        super().__init__()

        assert depth in PRETRAINED_MODELS, "ClsNet depth not supported"

        self.resnet = PRETRAINED_MODELS[depth](pretrained=True)

        self.fc_out_1 = nn.Linear(1000, 32)
        self.fc_out_2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.resnet(x))
        x = F.relu(self.fc_out_1(x))
        x = torch.softmax(self.fc_out_2(x), dim=-1)
        return x
