import math
import torch
import torch.nn as nn

class MyNet3D(torch.nn.Module):
    def __init__(self, args):
        super(MyNet3D, self).__init__()
        # if 'resnet' in args.model.backbone:
        #     self.my3DNet = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=args.train.pretrain)
        #     dim_last_layer = 400
        if 'mobilenet' in args.model.backbone:
            from .mobilenetv2_3D import get_model
            self.my3DNet = get_model(num_classes=600)
            self.my3DNet = nn.DataParallel(self.my3DNet)
            if args.train.pretrain:
                pretrain = torch.load(args.model.video_pretrained_model_path)
                # torch.load('/workspace/mmWave_Gesture/Gesture_ML/model/weight/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')
                self.my3DNet.load_state_dict(pretrain['state_dict'])
            self.my3DNet.module.classifier = nn.Identity()
            self.dim_last_layer = 1280
        elif 'resnet' in args.model.backbone:
            from .resnet_3D import resnet50
            self.my3DNet = resnet50(num_classes=600)
            self.my3DNet = nn.DataParallel(self.my3DNet)
            if args.train.pretrain:
                pretrain = torch.load('/workspace/mmWave_Gesture/Gesture_ML/model/weight/kinetics_resnet_50_RGB_16_best.pth')
                self.my3DNet.load_state_dict(pretrain['state_dict'])
            self.my3DNet.module.fc = nn.Identity()
            self.dim_last_layer = 2048
        elif 'x3d' in args.model.backbone:
            model_name = args.model.backbone.split('-')[0]
            self.my3DNet = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=args.train.pretrain)
            self.dim_last_layer = 400
        # self.my3DNet.features[0][0] = nn.Conv2d(channel_input, 32, (3,3), (2,2), bias=False)
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
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
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
        x = self.my3DNet(x)
        return x