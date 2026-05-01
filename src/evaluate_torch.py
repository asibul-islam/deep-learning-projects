import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path
val_dir = "../data/cats_dogs/val"

# Same transform used for ResNet18 transfer learning
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset and loader
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = val_dataset.classes

# Load ResNet18 model structure
model = models.resnet18(weights=None)

# Replace final layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load saved weights
model.load_state_dict(
    torch.load("../models/cats_dogs_transfer_resnet18.pth", map_location=device)
)

model = model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()
plt.title("Confusion Matrix - PyTorch ResNet18")
plt.show()