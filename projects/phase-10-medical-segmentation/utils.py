"""
工具函数
========

医学图像分割的辅助工具函数。
"""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


# ===========================================
# 数据下载
# ===========================================


def download_file(url, dest_path, progress=True):
    """
    下载文件

    Args:
        url: 下载链接
        dest_path: 保存路径
        progress: 是否显示进度
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"文件已存在: {dest_path}")
        return dest_path

    print(f"正在下载: {url}")
    print(f"保存到: {dest_path}")

    def reporthook(count, block_size, total_size):
        if progress and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\r下载进度: {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\n下载完成!")
        return dest_path
    except Exception as e:
        print(f"\n下载失败: {e}")
        # 尝试备用方法
        return download_with_requests(url, dest_path)


def download_with_requests(url, dest_path):
    """使用 requests 库下载（备用方法）"""
    try:
        import requests

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = int(downloaded * 100 / total)
                    print(f"\r下载进度: {percent}%", end="", flush=True)

        print("\n下载完成!")
        return dest_path
    except ImportError:
        print("请安装 requests: pip install requests")
        raise
    except Exception as e:
        print(f"下载失败: {e}")
        raise


def extract_zip(zip_path, extract_dir):
    """解压 ZIP 文件"""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    print(f"正在解压: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"解压完成: {extract_dir}")
    return extract_dir


# ===========================================
# 图像处理
# ===========================================


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    应用 CLAHE 对比度增强

    Args:
        image: 输入图像 (numpy array, uint8)
        clip_limit: 对比度限制
        grid_size: 网格大小

    Returns:
        增强后的图像
    """
    try:
        import cv2

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)
    except ImportError:
        # 如果没有 OpenCV，返回原图
        return image


def normalize_image(image, method="minmax"):
    """
    图像归一化

    Args:
        image: 输入图像
        method: "minmax" 或 "zscore"

    Returns:
        归一化后的图像 (0-1 范围)
    """
    if method == "minmax":
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            return (image - min_val) / (max_val - min_val)
        return image
    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        return image - mean
    else:
        raise ValueError(f"未知归一化方法: {method}")


# ===========================================
# 形态学后处理
# ===========================================


def postprocess_mask(mask, min_area=100, kernel_size=5):
    """
    对预测掩码进行后处理

    Args:
        mask: 二值掩码 (numpy array)
        min_area: 最小区域面积
        kernel_size: 形态学核大小

    Returns:
        处理后的掩码
    """
    try:
        import cv2
        from skimage.measure import label, regionprops

        # 二值化
        binary = (mask > 0.5).astype(np.uint8)

        # 形态学闭运算 (填充小孔)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 形态学开运算 (去除小噪点)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # 移除小区域
        labeled = label(opened)
        regions = regionprops(labeled)

        result = np.zeros_like(opened)
        for region in regions:
            if region.area >= min_area:
                result[labeled == region.label] = 1

        return result
    except ImportError:
        # 没有 OpenCV/skimage 时返回简单二值化
        return (mask > 0.5).astype(np.uint8)


# ===========================================
# 可视化
# ===========================================


def visualize_prediction(image, mask_true, mask_pred, save_path=None, show=True):
    """
    可视化分割结果

    Args:
        image: 原始图像
        mask_true: 真实掩码
        mask_pred: 预测掩码
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原图
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 真实掩码
    axes[1].imshow(mask_true, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # 预测掩码
    axes[2].imshow(mask_pred, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # 叠加可视化
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(mask_pred, cmap="Reds", alpha=0.5)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"保存可视化: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(images, masks_true, masks_pred, save_path=None, max_samples=4):
    """
    可视化一批预测结果

    Args:
        images: 图像张量 [N, C, H, W]
        masks_true: 真实掩码 [N, 1, H, W]
        masks_pred: 预测掩码 [N, 1, H, W]
        save_path: 保存路径
        max_samples: 最大显示数量
    """
    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()
        true = masks_true[i].squeeze().cpu().numpy()
        pred = masks_pred[i].squeeze().cpu().numpy()

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(true, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(pred, cmap="Reds", alpha=0.5)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_training_history(history, save_path=None):
    """
    绘制训练历史

    Args:
        history: 包含 loss, dice, val_loss, val_dice 的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Dice 曲线
    axes[1].plot(history["train_dice"], label="Train Dice")
    if "val_dice" in history:
        axes[1].plot(history["val_dice"], label="Val Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"保存训练曲线: {save_path}")

    plt.close()


# ===========================================
# 模型工具
# ===========================================


def save_checkpoint(model, optimizer, epoch, loss, dice, path):
    """保存检查点"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "dice": dice,
        },
        path,
    )
    print(f"保存检查点: {path}")


def load_checkpoint(model, optimizer, path):
    """加载检查点"""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["dice"]


# ===========================================
# 指标计算
# ===========================================


def dice_coefficient(pred, target, smooth=1e-5):
    """
    计算 Dice 系数

    Args:
        pred: 预测值 (0-1)
        target: 目标值 (0 或 1)
        smooth: 平滑项

    Returns:
        Dice 系数
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def iou_score(pred, target, smooth=1e-5):
    """
    计算 IoU (Jaccard) 系数

    Args:
        pred: 预测值 (0-1)
        target: 目标值 (0 或 1)
        smooth: 平滑项

    Returns:
        IoU 系数
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ===========================================
# 设备工具
# ===========================================


def get_device_info():
    """获取设备信息"""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    return info


def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    print("=" * 50)
    print("设备信息")
    print("=" * 50)
    print(f"PyTorch 版本: {info['pytorch_version']}")
    print(f"CUDA 可用: {info['cuda_available']}")
    print(f"MPS 可用: {info['mps_available']}")

    if info["cuda_available"]:
        print(f"CUDA 版本: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']}")
        print(f"显存: {info['gpu_memory']}")
    print("=" * 50)
