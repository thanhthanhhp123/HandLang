import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import HandDataset
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = HandDataset('data_mask', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = resnet50(weights='DEFAULT')
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, len(dataset.classes))
model = model.cuda()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

metrics_history = []
for epoch in range(10):
    model.train()
    train_loss, train_preds, train_targets = 0, [], []
    for mask, label in train_loader:
        mask, label = mask.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(output.argmax(1).cpu().numpy())
        train_targets.extend(label.cpu().numpy())

    train_metrics = compute_metrics(train_targets, train_preds)
    train_loss /= len(train_loader)

    model.eval()
    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for mask, label in val_loader:
            mask, label = mask.to(device), label.to(device)
            output = model(mask)
            loss = criterion(output, label)

            val_loss += loss.item()
            val_preds.extend(output.argmax(1).cpu().numpy())
            val_targets.extend(label.cpu().numpy())

    val_metrics = compute_metrics(val_targets, val_preds)
    val_loss /= len(val_loader)

    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "val_loss": val_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"]
    }
    metrics_history.append(epoch_metrics)
    print(f"Epoch {epoch + 1}: {epoch_metrics}")

metrics_df = pd.DataFrame(metrics_history)
metrics_df.to_csv('metrics.csv', index=False)
torch.save(model.state_dict(), 'model.pth')
