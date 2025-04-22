import math
import einops
import torch
import torch.nn as nn

class MyNet2D(torch.nn.Module):
    def __init__(self, args):
        super(MyNet2D, self).__init__()
        if 'mobilenet' in args.model.backbone:
            self.my2DNet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=args.train.pretrain)
            self.my2DNet.classifier = nn.Identity()
            self.dim_last_layer = 1280
        elif 'resnet' in args.model.backbone:
            self.my2DNet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', args.train.pretrain)
            self.my2DNet.fc = nn.Identity()
            self.dim_last_layer = 2048
        # self.my2DNet.features[0][0] = nn.Conv2d(channel_input, 32, (3,3), (2,2), bias=False)
        #initialize
        if args.train.pretrain==False:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
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
        if len(x.shape)==3: # if channel dim=1:
            x = einops.repeat(x, 'b h w -> b (copy) h w', copy=3)
        x = self.my2DNet(x)
        return x