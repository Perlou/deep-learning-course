import torch
import torch.nn as nn

print("=" * 60)
print("第6节: VGGNet 架构")
print("=" * 60)

# VGG 配置
cfgs = {
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}


def make_layers(cfg, batch_norm=False):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(2, 2)]
        else:
            conv = nn.Conv2d(in_ch, v, 3, padding=1)
            layers += (
                [conv, nn.BatchNorm2d(v), nn.ReLU(True)]
                if batch_norm
                else [conv, nn.ReLU(True)]
            )
            in_ch = v
    return nn.Sequential(*layers)
