import os
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)


# 下载示例图片（如果不存在）
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.jpg")
if not os.path.exists(IMAGE_PATH):
    print("Downloading sample image...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/640px-Image_created_with_a_mobile_phone.png"
    urllib.request.urlretrieve(url, IMAGE_PATH)
    print(f"Image saved to {IMAGE_PATH}")


# VOC 调色板
def get_voc_palette(num_classes=21):
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = [r, g, b]
    return palette


# 加载模型
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()

# 预处理
image = Image.open(IMAGE_PATH)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
input_tensor = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(input_tensor)["out"]
pred = output.argmax(1).squeeze().numpy()


# 彩色可视化
palette = get_voc_palette()
colored = palette[pred]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(colored)
plt.title("Segmentation")
plt.savefig("segmentation_result.png")
