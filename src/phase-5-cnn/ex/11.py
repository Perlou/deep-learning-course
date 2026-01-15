import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.features[name] = output.detach()

            return hook

        for name, module in self.model.named_modules():
            if name in self.layers:
                module.register_forward_hook(hook_fn(name))

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 指定要提取的层
layers = ["layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
extractor = FeatureExtractor(model, layers)

# 创建随机输入测试
x = torch.randn(1, 3, 224, 224)
features = extractor(x)

print("各层特征图形状:")
for name, feat in features.items():
    print(f"  {name}: {feat.shape}")


def visualize_features(features, layer_name, num_channels=16):
    """可视化特征图"""
    if layer_name not in features:
        print(f"层 {layer_name} 不存在")
        return

    feat = features[layer_name][0]  # 取第一个样本
    num_channels = min(num_channels, feat.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.imshow(feat[i].cpu().numpy(), cmap="viridis")
            ax.set_title(f"Ch {i}")
        ax.axis("off")
    plt.suptitle(f"Feature Maps: {layer_name}")
    plt.tight_layout()
    return fig


# 加载真实图片
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img = Image.open("./avatar.jpg")
x = transform(img).unsqueeze(0)

# 提取特征
features = extractor(x)

# 可视化
fig = visualize_features(features, "layer1.0.conv1")
plt.savefig("features.png")
