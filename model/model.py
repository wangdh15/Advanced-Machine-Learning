import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def transpose3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, output_padding=1)


class Transposed_Bottlneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(Transposed_Bottlneck, self).__init__()
        self.conv1 = conv1x1(planes * 4, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = transpose3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, inplanes)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)

        return out


class Transposed_ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(Transposed_ResNet, self).__init__()
        self.inplanes = 2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.relu = nn.ReLU(inplace=True)
        self.uppooling = nn.functional.interpolate
        self.bn1 = nn.BatchNorm2d(64)
        # self.conv1 = nn.Conv2d(64, 3, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=8, padding=3, stride=2, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Transposed_Bottlneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Transposed_Bottlneck):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        layers = []
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))

        if planes != 64:
            self.inplanes = planes * 2
        else:
            self.inplanes = planes
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                upsample = nn.Sequential(
                    conv1x1(planes * block.expansion, self.inplanes, stride),
                    nn.BatchNorm2d(self.inplanes),
                )
            else:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(planes * block.expansion, self.inplanes,
                                       kernel_size=1, stride=2, output_padding=1),
                    nn.BatchNorm2d(self.inplanes)
                )
        layers.append(block(self.inplanes, planes, stride, upsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.uppooling(x, scale_factor=2)
        x = self.relu(x)
        x = self.conv1(x)
        return x


def Transposed_resnet50(pretrained=False, **kwargs):
    model = Transposed_ResNet(Transposed_Bottlneck, [3, 4, 6, 3], **kwargs)
    return model


class encoder(nn.Module):
    def __init__(self, arch='resnet50'):
        super(encoder, self).__init__()
        if arch == 'resnet50':
            self.model = models.resnet50(pretrained=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = Transposed_resnet50()

    def forward(self, x):
        return self.model(x)


class encoder_decoder(nn.Module):
    def __init__(self):
        super(encoder_decoder, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return x
