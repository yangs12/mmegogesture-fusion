import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Configuration
# -------------------------------
INPUT_DIM = 1280 * 2
NUM_CLASSES = 12
BATCH_SIZE = 16
EPOCHS = 3
torch.backends.quantized.engine = 'qnnpack'  # use fbgemm for training
DEVICE = torch.device("cpu")

# -------------------------------
# Classifier with optional BatchNorm
# -------------------------------
class FusionClassifier(nn.Module):
    def __init__(self, input_dimension, num_classes, dropout=True, batchnorm=False):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        layers = []
        layers.append(nn.Linear(input_dimension, 256))
        if batchnorm: layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        if dropout: layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(256, 128))
        if batchnorm: layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        if dropout: layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(128, num_classes))
        self.fc = nn.Sequential(*layers)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# -------------------------------
# Dummy Dataset
# -------------------------------
def get_dummy_loader():
    X = torch.randn(200, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (200,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Training Loop
# -------------------------------
def train(model, dataloader, epochs):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done. Loss = {loss.item():.4f}")
    model.eval()

# -------------------------------
# Quantize (qnnpack) Without BatchNorm
# -------------------------------
def quantize_for_pi(fp32_weights_path, dataloader):
    torch.backends.quantized.engine = 'qnnpack'  # switch for Raspberry Pi

    model = FusionClassifier(INPUT_DIM, NUM_CLASSES, dropout=True, batchnorm=False)
    model.load_state_dict(torch.load(fp32_weights_path), strict=False)
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate
    with torch.no_grad():
        for x, _ in dataloader:
            model(x)

    torch.quantization.convert(model, inplace=True)
    return model

# -------------------------------
# Evaluate
# -------------------------------
def evaluate(model, dataloader):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {correct/total:.2%}")

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    # Step 1: Train on x86 with BatchNorm + fbgemm
    train_loader = get_dummy_loader()
    model = FusionClassifier(INPUT_DIM, NUM_CLASSES, dropout=True, batchnorm=True)
    # print the layers
    print(model)
    train(model, train_loader, EPOCHS)
    torch.save(model.state_dict(), "fusion_fp32_with_bn.pt")
    print("✅ Trained FP32 model with BatchNorm saved.")

    # Step 2: Reload into BatchNorm-free version and Quantize for Pi
    quantized_model = quantize_for_pi("fusion_fp32_with_bn.pt", train_loader)
    torch.save(quantized_model.state_dict(), "fusion_int8_qnnpack.pt")
    print("✅ Quantized INT8 model for Pi saved.")

    # Step 3: Reload Quantized Model and Evaluate
    model_pi = FusionClassifier(INPUT_DIM, NUM_CLASSES, dropout=True, batchnorm=False)
    model_pi.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model_pi, inplace=True)
    torch.quantization.convert(model_pi, inplace=True)
    model_pi.load_state_dict(torch.load("fusion_int8_qnnpack.pt"))
    model_pi.eval()

    evaluate(model_pi, train_loader)