import torch
import monai
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, ToTensor
from monai.metrics import ROCAUCMetric
from monai.data import DataLoader, CacheDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Define transformations for preprocessing
define_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])

# Prepare dataset and dataloaders
train_ds = CacheDataset(data=train_data, transform=define_transforms)
val_ds = CacheDataset(data=val_data, transform=define_transforms)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Define model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
loss_function = CrossEntropyLoss()
auc_metric = ROCAUCMetric()

# Training loop
best_auc = -1
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Validation
    model.eval()
    y_pred, y_true = torch.tensor([], device=device), torch.tensor([], device=device)
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            y_pred = torch.cat([y_pred, preds], dim=0)
            y_true = torch.cat([y_true, labels], dim=0)
    auc_metric(y_pred, y_true)
    auc_score = auc_metric.aggregate().item()
    auc_metric.reset()
    print(f"Epoch {epoch+1}, AUC: {auc_score:.4f}")
    
    if auc_score > best_auc:
        best_auc = auc_score
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model.")

print(f"Training completed. Best AUC: {best_auc:.4f}")