import cv2
import numpy
import os
import torch
import torch.nn as nn
import torchvision
from utils import to_mask
from torchvision.models import resnet50
from PIL import Image

classes = os.listdir('data_mask')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(weights='DEFAULT')
model.fc = nn.Linear(2048, 10)
model.load_state_dict(torch.load('model.pth', map_location=device))

cap_region_x_begin = 0.5
cap_region_y_end = 0.8

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                    (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 255, 0), 2)

    roi = frame[0:int(cap_region_y_end * frame.shape[0]),
                int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
    
    mask = to_mask(roi)
    mask = Image.fromarray(mask)
    mask = transform(mask).unsqueeze(0).to(device)
    output = model(mask)
    _, pred = torch.max(output, 1)

    cv2.putText(frame, classes[pred.item()], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    