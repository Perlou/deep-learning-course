"""
模型模块
========

YOLOv8 目标检测模型的高级封装。
"""

from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

from config import ModelConfig, InferenceConfig, PROJECT_ROOT


class ObjectDetector:
    """
    YOLOv8 目标检测器封装类

    提供简单易用的接口进行目标检测。
    """

    def __init__(
        self,
        model_size: str = "n",
        pretrained_weights: Optional[str] = None,
        device: str = "auto",
    ):
        """
        初始化检测器

        Args:
            model_size: 模型大小 (n/s/m/l/x)
            pretrained_weights: 自定义权重路径 (None 使用官方预训练)
            device: 推理设备 (auto/cuda/cpu/mps)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装 ultralytics: pip install ultralytics")

        self.model_size = model_size
        self.device = device

        # 加载模型
        if pretrained_weights:
            model_path = pretrained_weights
            print(f"加载自定义权重: {model_path}")
        else:
            model_path = f"yolov8{model_size}.pt"
            print(f"加载预训练模型: {model_path}")

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        print(f"✓ 模型加载成功")
        print(f"  类别数: {len(self.class_names)}")

    def predict(
        self,
        source: Union[str, np.ndarray, List],
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        执行目标检测

        Args:
            source: 图像路径、numpy 数组或图像列表
            conf: 置信度阈值
            iou: NMS IoU 阈值
            max_det: 最大检测数
            classes: 指定检测的类别 ID 列表 (None 为全部)
            verbose: 是否显示详细信息

        Returns:
            检测结果列表，每个元素包含:
            - boxes: 边界框 [N, 4] (xyxy)
            - scores: 置信度 [N]
            - class_ids: 类别 ID [N]
            - class_names: 类别名称 [N]
        """
        # 执行推理
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            max_det=max_det,
            classes=classes,
            verbose=verbose,
            device=self.device if self.device != "auto" else None,
        )

        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes

            detection = {
                "boxes": boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([]),
                "scores": boxes.conf.cpu().numpy() if len(boxes) > 0 else np.array([]),
                "class_ids": boxes.cls.cpu().numpy().astype(int)
                if len(boxes) > 0
                else np.array([]),
                "class_names": [self.class_names[int(cls)] for cls in boxes.cls]
                if len(boxes) > 0
                else [],
                "orig_shape": result.orig_shape,
            }
            detections.append(detection)

        return detections

    def predict_and_draw(
        self,
        source: Union[str, np.ndarray],
        conf: float = 0.25,
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> np.ndarray:
        """
        检测并绘制结果

        Args:
            source: 图像路径或 numpy 数组
            conf: 置信度阈值
            save_path: 保存路径 (可选)
            show: 是否显示图像

        Returns:
            绘制了检测结果的图像
        """
        import cv2
        from utils import draw_detections

        # 读取图像
        if isinstance(source, str):
            image = cv2.imread(source)
        else:
            image = source.copy()

        # 执行检测
        detections = self.predict(source, conf=conf)

        if len(detections) > 0 and len(detections[0]["boxes"]) > 0:
            det = detections[0]
            image = draw_detections(
                image,
                det["boxes"],
                det["class_ids"],
                det["scores"],
                list(self.class_names.values()),
            )

        # 保存或显示
        if save_path:
            cv2.imwrite(save_path, image)
            print(f"✓ 保存检测结果: {save_path}")

        if show:
            cv2.imshow("Detection Result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image

    def predict_video(
        self,
        source: Union[str, int],
        conf: float = 0.25,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        视频/摄像头检测

        Args:
            source: 视频路径或摄像头 ID (0 为默认摄像头)
            conf: 置信度阈值
            save_path: 保存路径 (可选)
            show: 是否显示实时画面
        """
        import cv2
        from utils import draw_detections

        # 打开视频源
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"视频信息: {width}x{height} @ {fps}FPS")

        # 设置视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测
                detections = self.predict(frame, conf=conf, verbose=False)

                if len(detections) > 0 and len(detections[0]["boxes"]) > 0:
                    det = detections[0]
                    frame = draw_detections(
                        frame,
                        det["boxes"],
                        det["class_ids"],
                        det["scores"],
                        list(self.class_names.values()),
                    )

                # 添加 FPS 显示
                frame_count += 1
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # 保存帧
                if writer:
                    writer.write(frame)

                # 显示
                if show:
                    cv2.imshow("Video Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("用户中断")
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
                print(f"✓ 保存视频: {save_path}")
            cv2.destroyAllWindows()

    def export(
        self, format: str = "onnx", output_path: Optional[str] = None, **kwargs
    ) -> str:
        """
        导出模型

        Args:
            format: 导出格式 (onnx/torchscript/tensorrt/...)
            output_path: 输出路径 (可选)
            **kwargs: 其他导出参数

        Returns:
            导出文件路径
        """
        print(f"导出模型为 {format.upper()} 格式...")
        export_path = self.model.export(format=format, **kwargs)
        print(f"✓ 导出完成: {export_path}")
        return str(export_path)

    def info(self) -> None:
        """打印模型信息"""
        print("\n" + "=" * 50)
        print("模型信息")
        print("=" * 50)
        print(f"模型大小: YOLOv8{self.model_size}")
        print(f"类别数: {len(self.class_names)}")
        print(f"设备: {self.device}")
        print("\n类别列表 (前 10 个):")
        for i, (idx, name) in enumerate(list(self.class_names.items())[:10]):
            print(f"  {idx}: {name}")
        if len(self.class_names) > 10:
            print(f"  ... (共 {len(self.class_names)} 个类别)")


def load_detector(
    model_path: Optional[str] = None, model_size: str = "n", device: str = "auto"
) -> ObjectDetector:
    """
    加载检测器的便捷函数

    Args:
        model_path: 模型权重路径 (None 使用预训练)
        model_size: 模型大小
        device: 设备

    Returns:
        ObjectDetector 实例
    """
    return ObjectDetector(
        model_size=model_size, pretrained_weights=model_path, device=device
    )


if __name__ == "__main__":
    print("=" * 50)
    print("目标检测模型模块演示")
    print("=" * 50)

    try:
        # 创建检测器
        detector = ObjectDetector(model_size="n")
        detector.info()

        print("\n使用示例:")
        print("""
# 单张图片检测
detections = detector.predict('image.jpg')
for det in detections:
    print(f"检测到 {len(det['boxes'])} 个物体")

# 检测并绘制
result_image = detector.predict_and_draw(
    'image.jpg',
    save_path='result.jpg'
)

# 视频检测
detector.predict_video(0)  # 摄像头
detector.predict_video('video.mp4', save_path='result.mp4')

# 导出模型
detector.export('onnx')
        """)

    except ImportError as e:
        print(f"⚠ {e}")
        print("请先安装 ultralytics: pip install ultralytics")
