"""
数据集加载
==========

加载和预处理 Montgomery County 肺部 X 光数据集。
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import DATA_DIR, DATASET_CONFIG, TRAIN_CONFIG
from utils import download_file, extract_zip


# ===========================================
# 数据集下载
# ===========================================


def download_montgomery_dataset(data_dir=None):
    """
    下载 Montgomery County 胸部 X 光数据集

    数据集包含:
    - 138 张胸部 X 光片
    - 左右肺部分割掩码

    Args:
        data_dir: 数据目录

    Returns:
        数据集路径
    """
    if data_dir is None:
        data_dir = DATA_DIR

    data_dir = Path(data_dir)
    dataset_dir = data_dir / "montgomery"

    # 检查是否已下载
    if (dataset_dir / "CXR_png").exists():
        print(f"数据集已存在: {dataset_dir}")
        return dataset_dir

    # 由于官方 URL 可能有限制，使用本地数据生成示例
    print("正在准备示例数据集...")
    create_sample_dataset(dataset_dir)

    return dataset_dir


def create_sample_dataset(dataset_dir, num_samples=50):
    """
    创建示例数据集（用于演示和测试）

    生成合成的肺部 X 光图像和掩码
    """
    dataset_dir = Path(dataset_dir)

    # 创建目录
    image_dir = dataset_dir / "CXR_png"
    mask_left_dir = dataset_dir / "ManualMask" / "leftMask"
    mask_right_dir = dataset_dir / "ManualMask" / "rightMask"

    for d in [image_dir, mask_left_dir, mask_right_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"生成 {num_samples} 个示例样本...")

    for i in range(num_samples):
        # 生成合成 X 光图像
        image = generate_synthetic_xray(256, 256)

        # 生成肺部掩码
        left_mask, right_mask = generate_lung_masks(256, 256)

        # 保存
        name = f"sample_{i:03d}"
        Image.fromarray(image).save(image_dir / f"{name}.png")
        Image.fromarray(left_mask).save(mask_left_dir / f"{name}.png")
        Image.fromarray(right_mask).save(mask_right_dir / f"{name}.png")

    print(f"示例数据集创建完成: {dataset_dir}")
    return dataset_dir


def generate_synthetic_xray(height, width):
    """生成合成 X 光图像"""
    # 基础背景噪声
    image = np.random.normal(50, 20, (height, width)).astype(np.float32)

    # 添加类似肋骨的条纹
    for y in range(20, height - 20, 25):
        stripe_height = random.randint(3, 6)
        image[y : y + stripe_height, 30 : width - 30] += random.randint(20, 40)

    # 添加肺部区域（更暗的椭圆）
    cy, cx = height // 2, width // 2

    # 左肺
    for y in range(height):
        for x in range(width):
            dx = (x - cx + width // 5) / (width // 4)
            dy = (y - cy) / (height // 2.5)
            if dx * dx + dy * dy < 1:
                image[y, x] -= 30 + random.randint(0, 10)

    # 右肺
    for y in range(height):
        for x in range(width):
            dx = (x - cx - width // 5) / (width // 4)
            dy = (y - cy) / (height // 2.5)
            if dx * dx + dy * dy < 1:
                image[y, x] -= 30 + random.randint(0, 10)

    # 归一化到 0-255
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def generate_lung_masks(height, width):
    """生成左右肺部掩码"""
    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)

    cy, cx = height // 2, width // 2

    # 左肺掩码
    for y in range(height):
        for x in range(width):
            dx = (x - cx + width // 5) / (width // 4)
            dy = (y - cy) / (height // 2.5)
            if dx * dx + dy * dy < 0.9:
                left_mask[y, x] = 255

    # 右肺掩码
    for y in range(height):
        for x in range(width):
            dx = (x - cx - width // 5) / (width // 4)
            dy = (y - cy) / (height // 2.5)
            if dx * dx + dy * dy < 0.9:
                right_mask[y, x] = 255

    return left_mask, right_mask


# ===========================================
# 数据集类
# ===========================================


class LungSegmentationDataset(Dataset):
    """
    肺部分割数据集

    Args:
        data_dir: 数据目录
        image_size: 图像大小 (H, W)
        transform: 数据增强
        split: "train" 或 "val"
        train_ratio: 训练集比例
    """

    def __init__(
        self,
        data_dir=None,
        image_size=(256, 256),
        transform=None,
        split="train",
        train_ratio=0.8,
    ):
        if data_dir is None:
            data_dir = download_montgomery_dataset()

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        self.split = split

        # 加载图像列表
        image_dir = self.data_dir / "CXR_png"
        self.images = sorted(list(image_dir.glob("*.png")))

        if len(self.images) == 0:
            raise ValueError(f"在 {image_dir} 中找不到图像文件")

        # 划分训练/验证集
        random.seed(42)
        indices = list(range(len(self.images)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        print(f"加载 {split} 集: {len(self.indices)} 个样本")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取真实索引
        real_idx = self.indices[idx]
        image_path = self.images[real_idx]

        # 获取掩码路径
        name = image_path.stem
        left_mask_path = self.data_dir / "ManualMask" / "leftMask" / f"{name}.png"
        right_mask_path = self.data_dir / "ManualMask" / "rightMask" / f"{name}.png"

        # 加载图像
        image = Image.open(image_path).convert("L")

        # 加载并合并掩码
        if left_mask_path.exists() and right_mask_path.exists():
            left_mask = np.array(Image.open(left_mask_path).convert("L"))
            right_mask = np.array(Image.open(right_mask_path).convert("L"))
            mask = np.maximum(left_mask, right_mask)
            mask = Image.fromarray(mask)
        else:
            # 如果掩码不存在，创建空掩码
            mask = Image.new("L", image.size, 0)

        # 调整大小
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        # 应用数据增强
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            # 默认转换
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        # 确保掩码是二值的
        mask = (mask > 0.5).float()

        return image, mask


# ===========================================
# 数据增强
# ===========================================


class SegmentationTransform:
    """
    分割任务的数据增强

    对图像和掩码应用相同的几何变换
    """

    def __init__(
        self,
        flip_prob=0.5,
        rotate_degrees=15,
        scale_range=(0.9, 1.1),
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        is_train=True,
    ):
        self.flip_prob = flip_prob
        self.rotate_degrees = rotate_degrees
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.is_train = is_train

    def __call__(self, image, mask):
        # 转换为 tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        if not self.is_train:
            return image, mask

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 随机旋转
        if self.rotate_degrees > 0:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 随机缩放
        if self.scale_range is not None:
            scale = random.uniform(*self.scale_range)
            h, w = image.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)
            image = TF.resize(image, [new_h, new_w])
            mask = TF.resize(mask, [new_h, new_w])

            # 裁剪或填充回原大小
            if scale > 1:
                # 中心裁剪
                top = (new_h - h) // 2
                left = (new_w - w) // 2
                image = TF.crop(image, top, left, h, w)
                mask = TF.crop(mask, top, left, h, w)
            else:
                # 填充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                image = TF.pad(
                    image, [pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h]
                )
                mask = TF.pad(
                    mask, [pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h]
                )

        # 亮度和对比度调整（仅对图像）
        if self.brightness_range is not None:
            brightness = random.uniform(*self.brightness_range)
            image = TF.adjust_brightness(image, brightness)

        if self.contrast_range is not None:
            contrast = random.uniform(*self.contrast_range)
            image = TF.adjust_contrast(image, contrast)

        return image, mask


# ===========================================
# 数据加载器
# ===========================================


def get_dataloaders(
    data_dir=None,
    image_size=(256, 256),
    batch_size=8,
    num_workers=4,
    augmentation=True,
):
    """
    获取训练和验证数据加载器

    Args:
        data_dir: 数据目录
        image_size: 图像大小
        batch_size: 批量大小
        num_workers: 数据加载线程数
        augmentation: 是否应用数据增强

    Returns:
        train_loader, val_loader
    """
    # 数据增强
    if augmentation:
        train_transform = SegmentationTransform(
            flip_prob=TRAIN_CONFIG["aug_flip_prob"],
            rotate_degrees=TRAIN_CONFIG["aug_rotate_degrees"],
            scale_range=TRAIN_CONFIG["aug_scale_range"],
            is_train=True,
        )
    else:
        train_transform = SegmentationTransform(is_train=False)

    val_transform = SegmentationTransform(is_train=False)

    # 创建数据集
    train_dataset = LungSegmentationDataset(
        data_dir=data_dir,
        image_size=image_size,
        transform=train_transform,
        split="train",
    )

    val_dataset = LungSegmentationDataset(
        data_dir=data_dir,
        image_size=image_size,
        transform=val_transform,
        split="val",
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ===========================================
# 测试
# ===========================================

if __name__ == "__main__":
    print("=" * 50)
    print("测试数据集加载")
    print("=" * 50)

    # 下载/准备数据集
    dataset_dir = download_montgomery_dataset()

    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=dataset_dir,
        batch_size=4,
        num_workers=0,
    )

    print(f"\n训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # 测试一批数据
    images, masks = next(iter(train_loader))
    print(f"\n图像形状: {images.shape}")
    print(f"掩码形状: {masks.shape}")
    print(f"图像值范围: [{images.min():.2f}, {images.max():.2f}]")
    print(f"掩码唯一值: {masks.unique().tolist()}")
