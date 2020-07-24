from __future__ import absolute_import
import math
import torch
import torch.nn as nn

__all__ = ['ResNet_v2']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], cfg[-1], kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class ResNet_v2(nn.Module):
    def __init__(self, depth=20, cfg=None, num_classes=10):
        super(ResNet_v2, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            cfg = [[16], [16, 16, 64]*n, [32, 32, 128]*n, [64, 64, 256]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0: 3*n+1])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n: 6*n+1], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n: 9*n+1], stride=2)

        self.bn = nn.BatchNorm2d(cfg[-1])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or cfg[0] != cfg[-1]:
            downsample = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[-1],
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3]+ [cfg[-1]], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i:3*(i+1)]+ [cfg[-1]] ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x