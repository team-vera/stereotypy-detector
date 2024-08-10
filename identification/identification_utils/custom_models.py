import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

PRETRAINED_MODELS = {
    "res18": torchvision.models.resnet18,
    "res34": torchvision.models.resnet34,
    "res50": torchvision.models.resnet50,
    "res101": torchvision.models.resnet101,
    "res152": torchvision.models.resnet152,
    "vgg16": torchvision.models.vgg16,
    "mobilev2": torchvision.models.mobilenet_v2,
    "resnext50": torchvision.models.resnext50_32x4d,
    "dense121": torchvision.models.densenet121
}


class IdentNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input and conv layers
        self.conv_1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv_2 = nn.Conv2d(8, 16, 3, padding=1)

        # Downsample to half size (conv)
        self.conv_down_1 = nn.Conv2d(16, 16, 3, stride=2)

        # Further conv layer
        self.conv_3 = nn.Conv2d(16, 32, 3, padding=1)

        # Downsample to half size (max_pool)
        self.max_down_1 = nn.MaxPool2d(2)

        # Further conv layers
        self.conv_4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv_5 = nn.Conv2d(32, 64, 3, padding=1)

        # Mapping to classes with fully connected
        self.output = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_down_1(x))
        x = F.relu(self.conv_3(x))
        x = self.max_down_1(x)
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        x = x.mean(dim=(2, 3))
        x = F.softmax(self.output(x), dim=-1)
        return x


class ClsNet(nn.Module):
    def __init__(self, depth: str):
        super().__init__()

        assert depth in PRETRAINED_MODELS, "Pretrained model not supported"

        self.resnet = PRETRAINED_MODELS[depth](
            pretrained=True,
            num_classes=1000)

        self.fc_out_1 = nn.Linear(1000, 32)
        self.fc_out_2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.resnet(x))
        x = F.relu(self.fc_out_1(x))
        x = F.softmax(self.fc_out_2(x), dim=-1)
        return x
