import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torchvision.models.quantization import mobilenet_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------
# Config
# -------------------------------
NUM_CLASSES = 12
EPOCHS = 1
BATCH_SIZE = 8
torch.backends.quantized.engine = 'qnnpack'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FusionClassifier(nn.Module):
    def __init__(self, input_dimension, num_classes,dropout=True, batchnorm=True):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc = nn.Sequential(
                nn.Linear(input_dimension, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
                )
        self.dequant = torch.quantization.DeQuantStub()
        self._initialize_weights_block(self.fc)

    def _initialize_weights_block(self, apply_block):
        for m in apply_block.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc(x)
        x = self.dequant(x)
        return x


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

# -------------------------------
# 1. Load Two MobileNetV2s and strip classifiers
# -------------------------------
model1 = mobilenet_v2(pretrained=True, quantize=False)
model2 = mobilenet_v2(pretrained=True, quantize=False)

model1.classifier = nn.Identity()
model2.classifier = nn.Identity()

model1.to(DEVICE)
model2.to(DEVICE)

# -------------------------------
# 2. Define Fusion Classifier (separate)
# -------------------------------
# fusion_classifier = nn.Sequential(
#     nn.Linear(1280 * 2, 512),
#     nn.ReLU(),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Linear(256, NUM_CLASSES)
# ).to(DEVICE)

fusion_classifier = FloatInputFusionClassifier(1280 * 2, NUM_CLASSES).to(DEVICE)

# -------------------------------
# 3. Fake Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
train_dataset = datasets.FakeData(size=100, image_size=(3, 224, 224), num_classes=NUM_CLASSES, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# 4. Train Loop
# -------------------------------
print("✅ Training...")
model1.train()
model2.train()
fusion_classifier.train()

optimizer = optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(fusion_classifier.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        feat1 = model1(images)
        feat2 = model2(images)

        fused = torch.cat([feat1, feat2], dim=1)
        outputs = fusion_classifier(fused)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

# Save the model
# After training
torch.save(model1.state_dict(), "model1.pth")  
torch.save(model2.state_dict(), "model2.pth")  
torch.save(fusion_classifier.state_dict(), "fusion_classifier.pth")


# -------------------------------
# 5. Quantize All 3 Separately
# -------------------------------
model1.cpu().eval()
model2.cpu().eval()
fusion_classifier.cpu().eval()

model1.fuse_model()
model2.fuse_model()

model1.qconfig = torch.quantization.get_default_qconfig('qnnpack')
model2.qconfig = torch.quantization.get_default_qconfig('qnnpack')
fusion_classifier.qconfig = torch.quantization.get_default_qconfig('qnnpack')

torch.quantization.prepare(model1, inplace=True)
torch.quantization.prepare(model2, inplace=True)
torch.quantization.prepare(fusion_classifier, inplace=True)

# Calibrate the models
with torch.no_grad():
    for _ in range(10):  # Calibration
        dummy_input = torch.randn(1, 3, 224, 224)
        f1 = model1(dummy_input)
        f2 = model2(dummy_input)
        _ = fusion_classifier(torch.cat([f1, f2], dim=1))

# save calibrated models
torch.save(model1.state_dict(), "model1_quantized.pth")
torch.save(model2.state_dict(), "model2_quantized.pth")
torch.save(fusion_classifier.state_dict(), "fusion_classifier_quantized.pth")

with torch.no_grad():
    for _ in range(10):
        dummy_input = torch.randn(1, 3, 224, 224)
        f1 = model1(dummy_input)
        f2 = model2(dummy_input)
        _ = fusion_classifier(torch.cat([f1, f2], dim=1))

torch.quantization.convert(model1, inplace=True)
torch.quantization.convert(model2, inplace=True)
torch.quantization.convert(fusion_classifier, inplace=True)

# verify quantization
inspect_model_quantization(model1, model_name="model1")
inspect_model_quantization(model2, model_name="model2")
inspect_model_quantization(fusion_classifier, model_name="fusion")

# -------------------------------
# 6. Inference
# -------------------------------
print("✅ Testing inference...")
model1.eval()
model2.eval()
fusion_classifier.eval()

with torch.no_grad():
    x = torch.randn(1, 3, 224, 224)
    f1 = model1(x)
    f2 = model2(x)
    fused = torch.cat([f1, f2], dim=1)
    out = fusion_classifier(fused)
    print("✅ Output shape:", out.shape)