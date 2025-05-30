import math
import einops
import torch
import torch.nn as nn

from .myNet2D import MyNet2D
from .myNet3D import MyNet3D

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

# similar to the original FusionClassifier, but with dropout and batchnorm options
class FusionClassifierOptions(nn.Module):
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
