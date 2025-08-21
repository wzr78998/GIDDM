import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torch.autograd.variable import *
from efficientnet_pytorch import EfficientNet
import math

# torch.cuda.set_device(2)


class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass

class DCRN_02(nn.Module):

    def __init__(self, input_channels, patch_size,model_name='DCRN_02'):
        super(DCRN_02, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        # Finish

        # Combination shape
        # self.inter_size = 128 + 24
        self.inter_size = 192 + 96


        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # # Fully connected Layer
        # self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(1)  # (64,1,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1  # (32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (32,128,7,7)

        x2 = self.conv5(x)  # (32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)  # (32,24,1,7,7)
        x2 = self.conv6(x2)  # (32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)  # (32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)  # (32,152,7,7)

        ###################
        # attention map
        ###################
        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)

        #####################
        # attention map over
        #####################
        # CMMD

        return x
    def output_num(self):
        return 288
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
## Some classes from https://github.com/ksaito-ut/OPDA_BP/blob/master/models/basenet.py

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50', model_path=None, normalize=True):
        super(ResNetFc, self).__init__()

        self.model_resnet = resnet_dict[model_name](pretrained=True)

        if model_path:
            self.model_resnet.load_state_dict(torch.load(model_path))
        if model_path or normalize:
            self.normalize = True
            self.mean = False
            self.std = False
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class EfficientNetB0(BaseFeatureExtractor):
    def __init__(self, model_name='efficientnet', normalize=True):
        super(EfficientNetB0, self).__init__()
        self.model_eff = EfficientNet.from_pretrained('efficientnet-b0')
        self.normalize = normalize
        self.mean = False
        self.std = False
        self.features = self.model_eff.extract_features
        self.__in_features = self.model_eff._fc.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def output_num(self):
        return self.__in_features

class DenseNet(BaseFeatureExtractor):
    def __init__(self, model_name='densenet', normalize=True):
        super(DenseNet, self).__init__()
        self.model_dense = models.densenet121(pretrained=True)
        self.normalize = normalize
        self.mean = False
        self.std = False
        self.features = self.model_dense.features
        self.__in_features = self.model_dense.classifier.in_features

    def get_mean(self):
        if self.mean is False:
            self.mean = Variable(
                torch.from_numpy(np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.mean

    def get_std(self):
        if self.std is False:
            self.std = Variable(
                torch.from_numpy(np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1)))).cuda()
        return self.std

    def forward(self, x):
        if self.normalize:
            x = (x - self.get_mean()) / self.get_std()
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def output_num(self):
        return self.__in_features


class Net_CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Net_CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x):
        x = self.fc(x)
        return x
# class Net_CLS(nn.Module):
#     def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
#         super(Net_CLS, self).__init__()
#         self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
#         self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
#         self.main = nn.Sequential(self.bottleneck,
#                                   nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.SiLU(),
#                                                 self.fc))
#     def forward(self, x):
#         for module in self.main.children():
#             x = module(x)
#         return x

class Net_CLS_C(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Net_CLS_C, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(self.bottleneck,
                                  nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                                self.fc))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class Net_CLS_DC(nn.Module):
    def  __init__(self, in_dim, out_dim, bottle_neck_dim=None):
        super(Net_CLS_DC, self).__init__()
        if bottle_neck_dim is None:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
            )
        else:
            self.main = nn.Sequential(nn.Linear(in_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, out_dim))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        return x
    def output_num(self):
        return 100

