from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(BaseModel):
    def __init__(self, num_classes: int = 10, base_filters: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self._base_filters = base_filters
        self._dropout_p = dropout

        self.in_planes = base_filters
        self.conv1 = _conv3x3(3, base_filters)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.layer1 = self._make_layer(base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters * 8, 2, stride=2)
        self.fc = nn.Linear(base_filters * 8 * BasicBlock.expansion, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def expand_head(self, num_new_classes: int) -> None:
        old_num = self.fc.in_features
        old_out = self.fc.out_features
        new_out = old_out + num_new_classes

        device = self.fc.weight.device
        dtype = self.fc.weight.dtype

        old_weight = self.fc.weight.data
        old_bias = self.fc.bias.data if self.fc.bias is not None else None

        new_fc = nn.Linear(old_num, new_out).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_fc.weight.data[:old_out] = old_weight
            if old_bias is not None:
                new_fc.bias.data[:old_out] = old_bias

            nn.init.normal_(new_fc.weight.data[old_out:], mean=0.0, std=0.001)
            if new_fc.bias is not None:
                new_fc.bias.data[old_out:].zero_()

        self.fc = new_fc

    @property
    def num_classes(self) -> int:
        return self.fc.out_features
