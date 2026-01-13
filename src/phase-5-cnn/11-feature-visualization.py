"""
11-feature-visualization.py - 特征图可视化

本节学习: 如何可视化 CNN 各层的特征图
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import numpy as np

print("=" * 60)
print("第11节: 特征图可视化")
print("=" * 60)

print("""
📌 为什么要可视化特征图?
1. 理解 CNN 在"看什么"
2. 调试和改进模型
3. 解释模型决策

可视化方法:
- 中间层特征图
- 卷积核权重
- 激活最大化
- Grad-CAM (下一节)
""")

# 特征提取器
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

# 创建模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 指定要提取的层
layers = ['layer1.0.conv1', 'layer2.0.conv1', 'layer3.0.conv1', 'layer4.0.conv1']
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
            ax.imshow(feat[i].cpu().numpy(), cmap='viridis')
            ax.set_title(f'Ch {i}')
        ax.axis('off')
    plt.suptitle(f'Feature Maps: {layer_name}')
    plt.tight_layout()
    return fig

# 可视化卷积核
def visualize_conv_weights(model, layer_name='conv1'):
    """可视化第一层卷积核"""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            weights = module.weight.data.cpu()
            print(f"卷积核形状: {weights.shape}")
            
            # 只展示前 16 个核
            n = min(16, weights.shape[0])
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i, ax in enumerate(axes.flat):
                if i < n:
                    w = weights[i]
                    if w.shape[0] == 3:  # RGB
                        w = w.permute(1, 2, 0)
                        w = (w - w.min()) / (w.max() - w.min())
                        ax.imshow(w.numpy())
                    else:
                        ax.imshow(w[0].numpy(), cmap='gray')
                ax.axis('off')
            plt.suptitle(f'Conv Kernels: {layer_name}')
            return fig
    return None

print("""
📌 使用示例:

# 加载真实图片
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = Image.open('your_image.jpg')
x = transform(img).unsqueeze(0)

# 提取特征
features = extractor(x)

# 可视化
fig = visualize_features(features, 'layer1.0.conv1')
plt.savefig('features.png')
""")

print("""
📝 要点总结:
1. 使用 forward hook 提取中间层特征
2. 浅层: 边缘、颜色等低级特征
3. 深层: 物体部件、语义等高级特征
4. 可视化帮助理解和调试模型
""")
