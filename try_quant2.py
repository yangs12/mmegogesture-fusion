import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models.quantization import mobilenet_v2 as q_mobilenet_v2

NUM_CLASSES = 10  # Change as needed

torch_backed_type = 'qnnpack'
# Rebuild the quantization-ready model
model = q_mobilenet_v2(pretrained=False, quantize=False)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.eval()
model.fuse_model()  # You must fuse again to match structure
model.load_state_dict(torch.load("mobilenetv2_fused.pth"))
print("Fused weights loaded.")

# Now you can continue with quantization:
torch.backends.quantized.engine = torch_backed_type
model.qconfig = torch.quantization.get_default_qconfig(torch_backed_type)
torch.quantization.prepare(model, inplace=True)
# ...

# ---- 5. Prepare for Quantization ----
model.qconfig = torch.quantization.get_default_qconfig(torch_backed_type)
torch.backends.quantized.engine = torch_backed_type
torch.quantization.prepare(model, inplace=True)

# ---- 7. Convert to Quantized ----
torch.quantization.convert(model, inplace=True)

# ---- 8. Run Inference ----
print("Running inference on quantized model...")
model.eval()
with torch.no_grad():
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    print("Output dtype:", output.dtype)
    print("Output shape:", output.shape)

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.quantized.Conv2d) or isinstance(module, torch.nn.quantized.Linear):
#         print(f"{name}: weight dtype = {module.weight().dtype}")