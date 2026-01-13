"""
12-grad-cam.py - Grad-CAM 可视化

本节学习: 使用 Grad-CAM 理解 CNN 的决策依据
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

print("=" * 60)
print("第12节: Grad-CAM 热力图可视化")
print("=" * 60)

print("""
📌 Grad-CAM (Gradient-weighted Class Activation Mapping)

核心思想: 使用梯度信息确定每个特征图通道的重要性

步骤:
1. 前向传播，获取目标层特征图 A
2. 计算目标类别分数对特征图的梯度
3. 全局平均池化梯度，得到通道权重 α
4. 加权求和: L = ReLU(Σ αᵢ × Aᵢ)
5. 上采样到原图尺寸

优点:
- 无需修改模型结构
- 可解释模型关注区域
- 支持任意 CNN 架构
""")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # 计算通道权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 上采样到原图尺寸
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy(), target_class

def apply_colormap(cam, img_array):
    """将 CAM 叠加到原图上"""
    heatmap = plt.cm.jet(cam)[:, :, :3]
    result = heatmap * 0.4 + img_array * 0.6
    return np.clip(result, 0, 1)

# 使用示例
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 目标层 (ResNet 的最后一个卷积层)
target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)

# 测试
x = torch.randn(1, 3, 224, 224)
cam, pred_class = gradcam.generate(x)
print(f"预测类别: {pred_class}")
print(f"CAM 形状: {cam.shape}")

print("""
📌 完整使用示例:

# 1. 加载图片
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = Image.open('dog.jpg')
input_tensor = transform(img).unsqueeze(0)

# 2. 生成 Grad-CAM
cam, pred_class = gradcam.generate(input_tensor)

# 3. 可视化
img_array = np.array(img.resize((224, 224))) / 255.0
result = apply_colormap(cam, img_array)

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(img_array); plt.title('Original')
plt.subplot(132); plt.imshow(cam, cmap='jet'); plt.title('Grad-CAM')
plt.subplot(133); plt.imshow(result); plt.title('Overlay')
plt.savefig('gradcam_result.png')
""")

print("""
📝 要点总结:
1. Grad-CAM 通过梯度确定重要区域
2. 目标层通常选择最后一个卷积层
3. 热力图显示模型"看"的位置
4. 可用于模型调试和可解释性分析
""")
