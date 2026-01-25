"""
推理脚本
========

对新图像进行分割推理。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from config import DEVICE, MODELS_DIR, RESULTS_DIR, MODEL_CONFIG, DATASET_CONFIG
from model import create_model
from utils import postprocess_mask, normalize_image


def load_model(model_path=None, device=None):
    """加载训练好的模型"""
    device = device or DEVICE

    if model_path is None:
        model_path = MODELS_DIR / "best_model.pth"
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"⚠ 模型不存在: {model_path}")
        print("请先训练模型: python train.py")
        return None

    # 创建模型
    model = create_model(
        model_name=MODEL_CONFIG["name"],
        n_channels=MODEL_CONFIG["in_channels"],
        n_classes=MODEL_CONFIG["out_channels"],
        bilinear=MODEL_CONFIG["bilinear"],
        features=MODEL_CONFIG["features"],
    )

    # 加载权重
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✓ 模型加载成功: {model_path.name}")

    return model


def preprocess_image(image_path, image_size=(256, 256)):
    """
    预处理输入图像

    Args:
        image_path: 图像路径
        image_size: 目标大小

    Returns:
        tensor: 预处理后的图像张量 [1, 1, H, W]
        original: 原始图像 (用于可视化)
    """
    # 加载图像
    image = Image.open(image_path).convert("L")
    original_size = image.size

    # 调整大小
    image_resized = image.resize(image_size, Image.Resampling.BILINEAR)

    # 转换为 tensor
    tensor = TF.to_tensor(image_resized).unsqueeze(0)

    return tensor, np.array(image), original_size


@torch.no_grad()
def predict(
    image_path,
    model=None,
    model_path=None,
    device=None,
    threshold=0.5,
    postprocess=True,
):
    """
    对单张图像进行分割预测

    Args:
        image_path: 输入图像路径
        model: 已加载的模型 (可选)
        model_path: 模型权重路径 (可选)
        device: 设备
        threshold: 二值化阈值
        postprocess: 是否应用后处理

    Returns:
        mask: 预测掩码 (numpy array)
        prob: 概率图 (numpy array)
    """
    device = device or DEVICE

    # 加载模型
    if model is None:
        model = load_model(model_path, device)
        if model is None:
            return None, None

    # 预处理图像
    tensor, original, original_size = preprocess_image(
        image_path, image_size=DATASET_CONFIG["image_size"]
    )
    tensor = tensor.to(device)

    # 预测
    output = model(tensor)
    prob = torch.sigmoid(output).squeeze().cpu().numpy()

    # 二值化
    mask = (prob > threshold).astype(np.uint8)

    # 后处理
    if postprocess:
        mask = postprocess_mask(mask)

    # 调整回原始大小
    mask_pil = Image.fromarray(mask * 255)
    mask_pil = mask_pil.resize(original_size, Image.Resampling.NEAREST)
    mask = np.array(mask_pil) // 255

    prob_pil = Image.fromarray((prob * 255).astype(np.uint8))
    prob_pil = prob_pil.resize(original_size, Image.Resampling.BILINEAR)
    prob = np.array(prob_pil) / 255.0

    return mask, prob


def predict_and_visualize(
    image_path,
    output_path=None,
    model=None,
    model_path=None,
    show=False,
):
    """
    预测并可视化结果

    Args:
        image_path: 输入图像路径
        output_path: 输出保存路径
        model: 已加载的模型
        model_path: 模型权重路径
        show: 是否显示图像

    Returns:
        mask: 预测掩码
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"⚠ 图像不存在: {image_path}")
        return None

    print(f"\n🔍 处理图像: {image_path.name}")

    # 加载原图
    original = np.array(Image.open(image_path).convert("L"))

    # 预测
    mask, prob = predict(image_path, model=model, model_path=model_path)

    if mask is None:
        return None

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原图
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 概率图
    axes[1].imshow(prob, cmap="jet")
    axes[1].set_title("Probability Map")
    axes[1].axis("off")

    # 分割掩码
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Segmentation Mask")
    axes[2].axis("off")

    # 叠加图
    axes[3].imshow(original, cmap="gray")
    axes[3].imshow(mask, cmap="Reds", alpha=0.5)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    # 保存
    if output_path is None:
        output_path = RESULTS_DIR / f"{image_path.stem}_segmented.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 结果保存于: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return mask


def predict_batch(
    input_dir,
    output_dir=None,
    model_path=None,
):
    """
    批量预测目录中的所有图像

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        model_path: 模型权重路径
    """
    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = RESULTS_DIR / "batch_predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有图像
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if len(image_files) == 0:
        print(f"⚠ 在 {input_dir} 中找不到图像文件")
        return

    print(f"\n📂 批量处理 {len(image_files)} 个图像...")

    # 加载模型
    model = load_model(model_path)
    if model is None:
        return

    for image_path in image_files:
        output_path = output_dir / f"{image_path.stem}_segmented.png"
        predict_and_visualize(
            image_path,
            output_path=output_path,
            model=model,
            show=False,
        )

    print(f"\n✓ 批量处理完成! 结果保存于: {output_dir}")


def demo_inference():
    """推理演示"""
    print("=" * 60)
    print("🏥 分割推理演示")
    print("=" * 60)

    # 检查是否有训练好的模型
    model_path = MODELS_DIR / "best_model.pth"

    if not model_path.exists():
        print("\n⚠ 模型不存在，请先训练:")
        print("   python main.py train --quick")
        return

    # 使用验证集中的图像进行演示
    from dataset import download_montgomery_dataset

    data_dir = download_montgomery_dataset()
    image_dir = data_dir / "CXR_png"

    images = list(image_dir.glob("*.png"))
    if len(images) > 0:
        # 随机选择一张图片
        import random

        image_path = random.choice(images)

        predict_and_visualize(
            image_path,
            output_path=RESULTS_DIR / "demo_inference.png",
            model_path=model_path,
            show=False,
        )
    else:
        print("⚠ 找不到图像文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分割推理")
    parser.add_argument("--source", type=str, default=None, help="输入图像路径或目录")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--show", action="store_true", help="显示结果")
    parser.add_argument("--demo", action="store_true", help="运行演示")

    args = parser.parse_args()

    if args.demo:
        demo_inference()
    elif args.source:
        source = Path(args.source)
        if source.is_dir():
            predict_batch(source, args.output, args.model)
        else:
            predict_and_visualize(
                source,
                output_path=args.output,
                model_path=args.model,
                show=args.show,
            )
    else:
        print("用法:")
        print("  python inference.py --demo")
        print("  python inference.py --source image.png")
        print("  python inference.py --source image_dir/")
