import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dir = "../data/cats_dogs/val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

class_names = val_dataset.classes

# Load same ResNet18 architecture
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load saved transfer model
model.load_state_dict(torch.load("../models/cats_dogs_transfer_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

images, labels = next(iter(val_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(2, 3, i + 1)

    # Unnormalize image for display
    img = images[i].cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    img = img.permute(1, 2, 0).numpy()
    img = img.clip(0, 1)

    plt.imshow(img)

    pred_label = class_names[preds[i].item()]
    true_label = class_names[labels[i].item()]

    plt.title(f"P: {pred_label} | A: {true_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()