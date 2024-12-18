import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def create_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Ngưỡng màu da để tạo mask (có thể điều chỉnh nếu cần)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)  # Giá trị HSV thấp
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)  # Giá trị HSV cao

    # Tạo mask bằng cách lọc màu da
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Làm sạch mask bằng các phép toán hình học (dilate và erode)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Đóng lỗ
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Loại bỏ nhiễu nhỏ

    # Tùy chọn: Làm mịn cạnh bằng GaussianBlur
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask


class SkinDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = ImageFolder('data/', transform=transform)
        self.classes = self.data.classes

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    