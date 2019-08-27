import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

NormLayer = nn.BatchNorm2d

webroot = 'http://dl.yf.io/drn/'

model_urls = {
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1,1), residual=True, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = planes #if groups is 1 else int(planes / groups)
        # int(planes * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = NormLayer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1], groups=groups)
        self.bn2 = NormLayer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = NormLayer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 channels=[16, 32, 64, 128, 256, 512, 512, 512],
                 groups=1, width_per_group=64, pool_size=28):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.groups = groups
        self.width_per_group = width_per_group

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            NormLayer(channels[0]),
            nn.ReLU(inplace=True))
        self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
        self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4],
                                       dilation=2, new_level=False)
        self.layer6 = self._make_layer(block, channels[5], layers[5],
                                       dilation=4, new_level=False)
        self.layer7 = self._make_conv_layers(channels[6], layers[6], dilation=2)
        self.layer8 = self._make_conv_layers(channels[7], layers[7], dilation=1)

        if num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.fc = nn.Conv2d(channels[7], num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, NormLayer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                NormLayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            dilation=(1, 1) if dilation == 1 else (
                            dilation // 2 if new_level else dilation, dilation),
                            groups=self.groups, width_per_group=self.width_per_group,
                            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                groups=self.groups, width_per_group=self.width_per_group,
                                dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                NormLayer(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.fc(x)
        return x


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model

def drn_d_56(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 16
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
    return model

def dilated_resnext56_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    kwargs['channels'] = [64, 128, 256, 512, 1024, 2048, 2048, 2048]
    return DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], **kwargs)
    if pretrained:
        raise ValueError('No pretrained model available for this architecture.')
    return model

def drn_d_105(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 16
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model

def drn_d_107(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 16
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
    return model

def dilated_resnext107_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    kwargs['channels'] = [64, 128, 256, 512, 1024, 2048, 2048, 2048]
    return DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], **kwargs)
    if pretrained:
        raise ValueError('No pretrained model available for this architecture.')
    return model
