import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
val_dir = "../data/cats_dogs/val"

# Transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Dataset
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class_names = val_dataset.classes


# model (same as training)
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 17 * 17, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.flatten(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x


# Load model
model = CatDogCNN().to(device)
model.load_state_dict(torch.load("../models/cats_dogs_basic_cnn_torch.pth"))
model.eval()


# Get one batch
images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)

# Predict
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Plot
plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    img = images[i].cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)

    pred_label = class_names[preds[i].item()]
    true_label = class_names[labels[i].item()]

    plt.title(f"P: {pred_label} | A: {true_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()