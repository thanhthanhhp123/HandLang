import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

def to_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask


class HandDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)
    
    def get_classes(self):
        return self.classes
    
    def __getitem__(self, idx):
        mask, label = self.dataset[idx]
        # mask = Image.fromarray(mask)

        if self.transform:
            mask = self.transform(mask)
        
        return mask, label
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    
    dataset = HandDataset('data_mask', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for mask, label in dataloader:
        plt.imshow(mask[0].permute(1, 2, 0))
        plt.title(dataset.classes[label[0].item()])
        plt.show()
        break


