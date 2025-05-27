import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torchvision.models.quantization import mobilenet_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def inspect_model_quantization(model, model_name="model"):
    print(f"\n🔍 Inspecting {model_name}...\n")

    # 1. Check if any quantized modules are used
    print("=== Layer types ===")
    for name, module in model.named_modules():
        if 'Quant' in str(type(module)) or 'quantized' in str(type(module)).lower():
            print(f"{name}: {type(module)}")

    # 2. Check parameter data types
    print("\n=== Parameter data types ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

    # 3. Check inference output data type
    try:
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            print("\n=== Inference output dtype ===")
            print(output.dtype)
            print("✅ Output shape:", output.shape)
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")



# -------------------------------
# Config
# -------------------------------
NUM_CLASSES = 12
EPOCHS = 1
BATCH_SIZE = 8
torch.backends.quantized.engine = 'qnnpack'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


seq = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

# -------------------------------
# 1. Load Two MobileNetV2 Models + Custom Classifiers
# -------------------------------
model1_fp32 = mobilenet_v2(pretrained=True, quantize=False)
model2_fp32 = mobilenet_v2(pretrained=True, quantize=False)

model1_fp32.classifier = seq
model2_fp32.classifier = seq

model1_fp32.to(DEVICE)
model2_fp32.to(DEVICE)

# -------------------------------
# 2. Prepare Fake Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

train_dataset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=NUM_CLASSES, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# 3. Train for 1 epoch (Late Fusion: average logits)
# -------------------------------
print("✅ Training late fusion model for 1 epoch...")
model1_fp32.train()
model2_fp32.train()
optimizer1 = optim.Adam(model1_fp32.parameters(), lr=1e-3)
optimizer2 = optim.Adam(model2_fp32.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        outputs1 = model1_fp32(images)
        outputs2 = model2_fp32(images)

        # Late fusion: average logits
        fused_outputs = (outputs1 + outputs2) / 2.0
        loss = criterion(fused_outputs, labels)

        loss.backward()
        optimizer1.step()
        optimizer2.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

# verify type
# inspect_model_quantization(model1_fp32, "Pre-Quantized Model 1")

# -------------------------------
# 4. Quantize Both Models
# -------------------------------
model1_fp32.eval().cpu()
model2_fp32.eval().cpu()

model1_fp32.fuse_model()
model2_fp32.fuse_model()

model1_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
model2_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

torch.quantization.prepare(model1_fp32, inplace=True)
torch.quantization.prepare(model2_fp32, inplace=True)

with torch.no_grad():
    for _ in range(10):
        dummy_input = torch.randn(1, 3, 224, 224)
        model1_fp32(dummy_input)
        model2_fp32(dummy_input)

torch.quantization.convert(model1_fp32, inplace=True)
torch.quantization.convert(model2_fp32, inplace=True)

# verify quantization
inspect_model_quantization(model1_fp32, "Post-Quantized Model 1")


# -------------------------------
# 5. Inference with Late Fusion
# -------------------------------
print("✅ Testing late fusion with quantized models...")
try:
    model1_fp32.eval()
    model2_fp32.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        output1 = model1_fp32(x)
        output2 = model2_fp32(x)
        fused_output = (output1 + output2) / 2.0
    print("✅ Late fusion inference succeeded. Output shape:", fused_output.shape)
except Exception as e:
    print("❌ Inference failed:", str(e))