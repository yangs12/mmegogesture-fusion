import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

# ---- 1. Load Pretrained Float Model for Fine-Tuning ----
NUM_CLASSES = 10  # Change as needed

model_fp32 = models.mobilenet_v2(pretrained=True)
model_fp32.classifier[1] = nn.Linear(model_fp32.last_channel, NUM_CLASSES)

for name, module in model_fp32.named_modules():
    if isinstance(module, torch.nn.quantized.Conv2d) or isinstance(module, torch.nn.quantized.Linear):
        print(f"{name}: weight dtype = {module.weight().dtype}")

# Optional: freeze some layers
# for param in model_fp32.features.parameters():
#     param.requires_grad = False

# ---- 2. Dummy Dataset and Training ----
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

# Replace this with your actual dataset
train_dataset = datasets.FakeData(size=100, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fp32.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fp32.to(device)

print("Training...")
model_fp32.train()
for epoch in range(1):  # Replace with more epochs as needed
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_fp32(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ---- 3. Prepare Quantization-Ready Version ----
print("Transferring weights to quant-ready model...")
q_model = q_mobilenet_v2(pretrained=False, quantize=False)
q_model.classifier[1] = nn.Linear(q_model.last_channel, NUM_CLASSES)
q_model.load_state_dict(model_fp32.state_dict(), strict=False)

# ---- 4. Fuse Layers ----
q_model.eval()
q_model.fuse_model()

torch.save(q_model.state_dict(), "mobilenetv2_fused.pth")
print("Fused model weights saved.")
