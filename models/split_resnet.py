import math
import torch
import torch.nn as nn
from layers import conv_type
from models.builder import get_builder

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64,slimming_factor=1):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(math.ceil(inplanes* slimming_factor), math.ceil(planes*slimming_factor), stride) ## Avoid residual links
        self.bn1 = builder.batchnorm(math.ceil(planes*slimming_factor))
        self.relu = builder.activation()

        self.conv2 = builder.conv3x3(math.ceil(planes * slimming_factor ),
                                         math.ceil(planes * slimming_factor))  ## Avoid residual links
        self.bn2 = builder.batchnorm(math.ceil(planes * slimming_factor), last_bn=True)  ## Avoid residual links

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        # print('1: ',torch.norm(residual[:,:residual.shape[1]//self.split]))
        out = self.conv1(x)
        # print('1.5 conv ',self.conv1.weight.shape)
        # print('1.5 conv ',torch.norm(self.conv1.weight[:self.conv1.weight.shape[0]//self.split,:self.conv1.weight.shape[1]//self.split]))
        # print('1.5: ', torch.norm(out[:, :out.shape[1] // self.split]))
        if self.bn1 is not None:
            out = self.bn1(out)
        # print('2: ', torch.norm(out[:,:out.shape[1]//self.split]))
        out = self.relu(out)
        out = self.conv2(out)
        # print('3: ', torch.norm(out[:,:out.shape[1]//self.split]))
        if self.bn2 is not None:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        # print('4: ', torch.norm(residual[:,:residual.shape[1]//self.split]))
        out += residual
        out = self.relu(out)
        # print('5: ', torch.norm(out[:,:out.shape[1]//self.split]))
        return out


# BasicBlock }}}

# Bottleneck {{{
class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64,slimming_factor=1,is_last_conv=False):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(math.ceil(inplanes * slimming_factor) , math.ceil(width * slimming_factor))
        self.bn1 = builder.batchnorm(math.ceil(width * slimming_factor))
        self.conv2 = builder.conv3x3(math.ceil(width * slimming_factor), math.ceil(width * slimming_factor), stride=stride)
        self.bn2 = builder.batchnorm(math.ceil(width * slimming_factor))
        self.conv3 = builder.conv1x1(math.ceil(width * slimming_factor), math.ceil(planes * self.expansion * slimming_factor))
        self.bn3 = builder.batchnorm(math.ceil(planes * self.expansion * slimming_factor))
        self.relu = builder.activation()
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


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self,cfg, builder, block, layers, base_width=64):

        super(ResNet, self).__init__()
        self.inplanes = 64
        slimming_factor = cfg.slimming_factor
        # assert slimming_factor == 1, 'ResNet does not support slimming because of the residual links'
        if slimming_factor < 1:
            cfg.logger.info('WARNING: You are using a slim network')

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")


        self.conv1 = builder.conv7x7(3, math.ceil(64*slimming_factor), stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(math.ceil(64*slimming_factor))
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0],slimming_factor=slimming_factor)

        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2,slimming_factor=slimming_factor)

        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2,slimming_factor=slimming_factor)

        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2,slimming_factor=slimming_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = builder.linear(math.ceil(512 * block.expansion * slimming_factor), cfg.num_cls, last_layer=True)


    def _make_layer(self, builder, block, planes, blocks, stride=1,slimming_factor=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(math.ceil(self.inplanes * slimming_factor),
                                    math.ceil(planes * block.expansion * slimming_factor), stride=stride) ## Going into a residual link
            dbn = builder.batchnorm(math.ceil(planes * block.expansion * slimming_factor))
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width,slimming_factor=slimming_factor))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width,
                                slimming_factor=slimming_factor))

        return nn.Sequential(*layers)

    def forward(self, x):
        # features = []
        x = self.conv1(x)
        # features.append(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # features.append(x)
        # print('resnet 1 ',torch.norm(x[:,:x.shape[1]//split]))
        # self.layer1[0].split = split
        # self.layer1[1].split = split
        x = self.layer1(x)
        # features.append(x)

        x = self.layer2(x)
        # features.append(x)
        x = self.layer3(x)
        # features.append(x)
        x = self.layer4(x)
        # features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)
        # features.append(x)
        return x

from collections import OrderedDict, namedtuple

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

def load_state_dict(model, state_dict,
                    strict: bool = True):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.
    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None and not (child.__class__.__name__ == 'SplitLinear' or child.__class__.__name__ == 'Linear'):
                load(child, prefix + name + '.')

    load(model)
    load = None  # break load->load reference cycle

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)

# ResNet }}}
def Split_ResNet18(cfg, progress=True):
    model = ResNet(cfg,get_builder(cfg), BasicBlock, [2, 2, 2, 2])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet18'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model

def Split_ResNet34(cfg, progress=True):
    model = ResNet(cfg,get_builder(cfg), BasicBlock, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet34'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model

def Split_ResNet50(cfg,progress=True):
    model = ResNet(cfg,get_builder(cfg), Bottleneck, [3, 4, 6, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet50'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model


def Split_ResNet101(cfg,progress=True):
    model = ResNet(cfg,get_builder(cfg), Bottleneck, [3, 4, 23, 3])
    if cfg.pretrained == 'imagenet':
        arch = 'resnet101'
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        load_state_dict(model,state_dict,strict=False)
    return model


# def WideResNet50_2(cfg,pretrained=False):
#     return ResNet(cfg,
#         get_builder(cfg), Bottleneck, [3, 4, 6, 3], base_width=64 * 2
#     )
#
#
# def WideResNet101_2(cfg,pretrained=False):
#     return ResNet(cfg,
#         get_builder(cfg), Bottleneck, [3, 4, 23, 3], base_width=64 * 2
#     )

