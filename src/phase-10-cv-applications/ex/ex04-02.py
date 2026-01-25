# import json
# import numpy as np
# from PIL import Image, ImageDraw


# def json_to_mask(json_path, output_path):
#     with open(json_path) as f:
#         data = json.load(f)

#     h, w = data["imageHeight"], data["imageWidth"]
#     mask = np.zeros((h, w), dtype=np.uint8)

#     for shape in data["shapes"]:
#         label = shape["label"]
#         points = shape["points"]

#         # 创建多边形掩码
#         img = Image.new("L", (w, h), 0)
#         ImageDraw.Draw(img).polygon([tuple(p) for p in points], outline=1, fill=1)
#         mask = np.maximum(mask, np.array(img) * label_to_id[label])

#     Image.fromarray(mask).save(output_path)
