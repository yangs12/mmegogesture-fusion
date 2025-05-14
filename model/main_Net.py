import math
import einops
import torch
import torch.nn as nn

from .myNet2D import MyNet2D
from .myNet3D import MyNet3D

class MyNet_Main(torch.nn.Module):
    def __init__(self, args, device):
        super(MyNet_Main, self).__init__()
        print("===> Model: Concatenation")
        self.device = device
        self.sensor = [args.sensor.select] if str(type(args.sensor.select))=="<class 'str'>" else args.sensor.select
        dim_last_layer = 0
        self.encoder = nn.ModuleDict({})
        for sensor_select in self.sensor:
            self.encoder[sensor_select] = MyNet3D(args).to(self.device) if 'vid' in sensor_select else MyNet2D(args).to(self.device)
            dim_last_layer += self.encoder[sensor_select].dim_last_layer
        self.classifier = nn.Sequential(
                        nn.Linear(dim_last_layer, 256),
                        nn.Dropout(p=0.5),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.Dropout(p=0.5),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, args.train.n_class),
                        )
        self._initialize_weights_block(self.classifier)
        
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
        x_encoded = {}
        for sensor_select in self.sensor:
            x_select = x[sensor_select]
            x_encoded[sensor_select] = self.encoder[sensor_select](x_select)
        # Fusion (now concat)
        for sensor_idx, sensor_select in enumerate(self.sensor):
            if sensor_idx==0:
                x_fuse = x_encoded[sensor_select]
            else:
                x_fuse = torch.cat((x_fuse,x_encoded[sensor_select]),dim=-1)
        y = self.classifier(x_fuse)
        return y