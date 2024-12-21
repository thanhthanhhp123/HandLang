import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from utils import to_mask
from torchvision.models import resnet50
from PIL import Image
import copy 
import time


classes = os.listdir('data_mask')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(weights=None)
model.fc = nn.Linear(2048, 10)

model = model.to(device)

def predict(image):
    image = cv2.resize(image, (224, 224))
    mask = to_mask(image)
    mask = Image.fromarray(image)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])
    mask = transform(mask)
    mask = mask.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(mask)
        _, predicted = torch.max(output, 1)
    
    return classes[predicted.item()]

def camera():
    cap = cv2.VideoCapture(1)
    cap_region_x_begin = 0.5
    cap_region_y_end = 0.8
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break
        
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0), (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)
        roi = frame[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):]
        prediction = predict(roi)
        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera()