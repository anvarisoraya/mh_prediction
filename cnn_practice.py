import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)



NUM_CLASSES = 10

def build_model_feature_extraction():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, NUM_CLASSES)

    return model

def build_model_finetuning():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, NUM_CLASSES)

    return model

model = build_model_finetuning().to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f'Trainable parameters: {trainable:,}')
print(f'Frozen parameters:    {frozen:,}')
print(f'\nFinal layer:          {model.fc}')


BATCH_SIZE = 128

train_transform = T.Compose([
    T.Resize((64, 64)),  # ResNet expects larger than 32x32
    T.RandomHorizontalFlip(),
    T.RandomCrop(64, padding=8),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.CIFAR10('./data', train=True,  download=False, transform=train_transform)
val_dataset   = datasets.CIFAR10('./data', train=False, download=False, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / len(loader), correct / total, all_preds, all_labels




EPOCHS = 3

train_losses, val_losses   = [], []
train_accs,   val_accs     = [], []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    train_losses.append(tr_loss); val_losses.append(va_loss)
    train_accs.append(tr_acc);    val_accs.append(va_acc)

    print(f'Epoch {epoch:2d}/{EPOCHS}  '
          f'Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2%}  |  '
          f'Val Loss: {va_loss:.4f} Acc: {va_acc:.2%}')

print('\n✅ Training complete!')



fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, 'b-o', label='Train')
axes[0].plot(val_losses,   'r-s', label='Val')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(train_accs, 'b-o', label='Train')
axes[1].plot(val_accs,   'r-s', label='Val')
axes[1].set_title('Accuracy Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
_, _, final_preds, final_labels = evaluate(model, val_loader, criterion, device)
print('\nFinal Classification Report:')
print(classification_report(final_labels, final_preds, target_names=class_names))




torch.save(model.state_dict(), 'resnet18_cifar10.pth')

loaded_model = build_model_finetuning().to(device)
loaded_model.load_state_dict(torch.load('resnet18_cifar10.pth', map_location=device))
loaded_model.eval()


