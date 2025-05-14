import math
import einops
import torch
import torch.nn as nn

from .myNet2D import MyNet2D
from .myNet3D import MyNet3D

class MyNet_Main(nn.Module):
    def __init__(self, args, device):
        super(MyNet_Main, self).__init__()
        print("===> Model: Late Fusion")
        self.device = device
        self.sensor = [args.sensor.select] if isinstance(args.sensor.select, str) else args.sensor.select
        self.encoder = nn.ModuleDict({})
        self.classifier = nn.ModuleDict({})

        for sensor_select in self.sensor:
            is_video = 'vid' in sensor_select
            self.encoder[sensor_select] = MyNet3D(args).to(device) if is_video else MyNet2D(args).to(device)

            dim = self.encoder[sensor_select].dim_last_layer
            self.classifier[sensor_select] = nn.Sequential(
                nn.Linear(dim, 256),
                nn.Dropout(p=0.5),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Dropout(p=0.5),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, args.train.n_class),
            ).to(device)

            self._initialize_weights_block(self.classifier[sensor_select])

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(self.sensor)), requires_grad=True)

    def _initialize_weights_block(self, apply_block):
        for m in apply_block.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        logits_list = []
        for i, sensor_select in enumerate(self.sensor):
            x_input = x[sensor_select]
            x_feat = self.encoder[sensor_select](x_input)
            logits = self.classifier[sensor_select](x_feat)
            logits_list.append(logits.unsqueeze(0))  # shape: (1, B, C)

        logits_all = torch.cat(logits_list, dim=0)  # shape: (N_sensors, B, C)

        # Softmax over fusion weights
        weights = torch.softmax(self.fusion_weights, dim=0)  # shape: (N_sensors,)

        # Weighted sum over sensors
        weighted_logits = (weights[:, None, None] * logits_all).sum(dim=0)  # shape: (B, C)

        return weighted_logits