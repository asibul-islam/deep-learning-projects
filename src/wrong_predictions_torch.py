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
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = val_dataset.classes

# Load model
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load("../models/cats_dogs_transfer_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

wrong_images = []
wrong_preds = []
wrong_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(len(images)):
            if preds[i].cpu().item() != labels[i].item():
                wrong_images.append(images[i].cpu())
                wrong_preds.append(preds[i].cpu().item())
                wrong_labels.append(labels[i].item())

# Show wrong predictions
plt.figure(figsize=(10, 6))

for i in range(min(6, len(wrong_images))):
    plt.subplot(2, 3, i + 1)

    img = wrong_images[i]

    # unnormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    img = img.permute(1, 2, 0).numpy()
    img = img.clip(0, 1)

    plt.imshow(img)

    pred = class_names[wrong_preds[i]]
    actual = class_names[wrong_labels[i]]

    plt.title(f"P: {pred} | A: {actual}")
    plt.axis("off")

plt.tight_layout()
plt.show()