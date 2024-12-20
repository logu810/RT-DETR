import onnxruntime as ort 
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
from src.data.coco.coco_dataset import mscoco_category2name, mscoco_category2label, mscoco_label2category
import torch
import torch.nn as nn 
import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

size = torch.tensor([[640, 640]])
# print(onnx.helper.printable_graph(mm.graph))

# Load the original image without resizing
img_path = input("Enter Image path: ")
original_im = Image.open(img_path).convert('RGB')
original_size = original_im.size

# Resize the image for model input
im = original_im.resize((640, 640))
im_data = ToTensor()(im)[None]
print(im_data.shape)

sess = ort.InferenceSession("/kaggle/working/RT-DETR_backbone/rtdetr_pytorch/model.onnx")
output = sess.run(
    # output_names=['labels', 'boxes', 'scores'],
    output_names=None,
    input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
)

# print(type(output))
# print([out.shape for out in output])

labels, boxes, scores = output

draw = ImageDraw.Draw(original_im)  # Draw on the original image
thrh = 0.6

for i in range(im_data.shape[0]):

    scr = scores[i]
    lab = labels[i][scr > thrh]
    box = boxes[i][scr > thrh]

    print(i, sum(scr > thrh))

    for b, l in zip(box, lab):
        # Scale the bounding boxes back to the original image size
        b = [coord * original_size[j % 2] / 640 for j, coord in enumerate(b)]
        # Get the category name from the label
        category_name = mscoco_category2name[mscoco_label2category[l-1]]
        draw.rectangle(list(b), outline='red', width=2)
        font = ImageFont.load_default()  
        draw.text((b[0], b[1]), text=category_name, fill='blue', font=font)

# Save the original image with bounding boxes
original_im.save('test.jpg')
