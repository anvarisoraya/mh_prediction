# facial emotion recognition
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from collections import Counter


data_path = kagglehub.dataset_download("msambare/fer2013")
train_dir = os.path.join(data_path, 'train')
test_dir  = os.path.join(data_path, 'test')


sample_dir = os.path.join(train_dir, 'happy')
sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
img = Image.open(sample_file)

img_array = np.array(img)
print(f'Image shape: {img_array.shape}')     
print(f'Pixel range: {img_array.min()} to {img_array.max()}')
print(f'Data type:   {img_array.dtype}')

plt.figure(figsize=(3, 3))
plt.imshow(img_array, cmap='gray')           
plt.title('One FER2013 sample (48x48, 1 channel)')
plt.axis('off')
plt.show()




fer_train_transform = T.Compose([
    T.Grayscale(num_output_channels=1),                           
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.RandomCrop(48, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]) 
])

fer_val_transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])



train_dataset = datasets.ImageFolder(train_dir, transform=fer_train_transform)
val_dataset   = datasets.ImageFolder(test_dir,  transform=fer_val_transform)

class_names = train_dataset.classes 

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

print(f'Training samples:   {len(train_dataset)}')
print(f'Validation samples: {len(val_dataset)}')
print(f'Training batches:   {len(train_loader)}')

images, labels = next(iter(train_loader))
print(f'\nBatch shape: {images.shape}') 
print(f'Labels shape: {labels.shape}')


unnorm = lambda t: torch.clamp(t * 0.5 + 0.5, 0, 1)

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flatten()):
    img_show = unnorm(images[i]).squeeze(0).numpy()  # (1,48,48) -> (48,48)
    ax.imshow(img_show, cmap='gray')
    ax.set_title(class_names[labels[i]], fontsize=7)
    ax.axis('off')
plt.suptitle('Sample Augmented FER2013 Faces', fontsize=11)
plt.tight_layout()
plt.show()



counts = Counter(train_dataset.targets)
for idx, name in enumerate(class_names):
    print(f'{name:10s}: {counts[idx]:5d} images')
