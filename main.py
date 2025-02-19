import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

path='best.pt'

model = torch.hub.load(
    'WongKinYiu/yolov7', 
    'custom', 
    path,
    force_reload=True,
    trust_repo=True)

def get_yolov7_annotations(img_path):
img = cv2.imread(img_path)
    results = model(img)
    boxes = results.xyxy[0]
    if len(boxes) == 0:
        print("No objects detected.")
        return None
    boxes = boxes.cpu().numpy()
    coordinates = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1 = int(x1)  # top-left x,
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        coordinates.append((x1, y1, x2, y2))  # append the coordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red color, 2 thickness
        
        
    return coordinates
