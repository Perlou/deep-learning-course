import torch
import os
import urllib.request
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


def main():
    # 下载示例图片（如果不存在）
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.jpg")
    if not os.path.exists(IMAGE_PATH):
        print("Downloading sample image...")
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/640px-Image_created_with_a_mobile_phone.png"
        urllib.request.urlretrieve(url, IMAGE_PATH)
        print(f"Image saved to {IMAGE_PATH}")
    print(IMAGE_PATH)
    replace_background(
        IMAGE_PATH,
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/640px-Image_created_with_a_mobile_phone.png",
        "",
    )


def replace_background(image_path, bg_path, output_path):
    # 加载模型
    model = deeplabv3_resnet50(pretrained=True).eval()

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    background = Image.open(bg_path).convert("RGB")
    background = background.resize(image.size)

    # 预处理
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))["out"]
    mask = output.argmax(1).squeeze().numpy()

    person_mask = (mask == 15).astype(np.float32)

    # 平滑边缘 (可选)
    from scipy.ndimage import gaussian_filter

    person_mask = gaussian_filter(person_mask, sigma=2)

    # 合成
    image_np = np.array(image) / 255.0
    bg_np = np.array(background) / 255.0

    mask_3d = np.stack([person_mask] * 3, axis=-1)
    result = image_np * mask_3d + bg_np * (1 - mask_3d)

    result = (result * 255).astype(np.uint8)
    Image.fromarray(result).save(output_path)


if __name__ == "__main__":
    main()
