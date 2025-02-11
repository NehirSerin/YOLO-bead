import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

path='best.pt'

def get_yolov7_annotations(img_path):
    model = torch.hub.load(
        'WongKinYiu/yolov7', 
        'custom', 
        path,
        force_reload=True,
        trust_repo=True)
   
    img = cv2.imread(img_path)
    results = model(img)
    boxes = results.xywh[0].cpu().numpy()  # xywh format (x_center, y_center, width, height)
    if len(boxes) == 0:
        print("No objects detected.")
        return None
    coordinates = []
    for box in boxes:
        x_center, y_center, width, height, conf, cls = box
        x1 = int((x_center - width / 2))  # top-left x
        y1 = int((y_center - height / 2))  # top-left y
        x2 = int((x_center + width / 2))  # bottom-right x
        y2 = int((y_center + height / 2))  # bottom-right y
        coordinates.append((x1, y1, x2, y2))  # append the coordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red color, 2 thickness
        
        
    return coordinates
