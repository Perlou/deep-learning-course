import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms


def train():
    COCO_CLASSES = [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image = cv2.imread("../image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = transforms.ToTensor()
    img_tensor = transforms(image_rgb)

    with torch.no_grad():
        predictions = model([img_tensor])

    fig, ax = plt.subplots(1)
    ax.imshow(image_rgb)

    for box, label, score in zip(
        predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["scores"]
    ):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{COCO_CLASSES[label]}: {score:.2f}")

        plt.savefig("faster_rcnn_result.png")
