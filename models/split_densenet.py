import re
import math
import torch
import torch.nn as nn
from torch import Tensor
from models import common
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.jit.annotations import List
from models.builder import get_builder

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(self,builder, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False,in_channels_order=None):
        super(_DenseLayer, self).__init__()
        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                                    growth_rate, kernel_size=1, stride=1,
        #                                    bias=False)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                                    kernel_size=3, stride=1, padding=1,
        #                                    bias=False)),

        self.add_module('norm1', builder.batchnorm(num_input_features,in_channels_order=in_channels_order)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', builder.conv1x1(num_input_features, bn_size *
                                           growth_rate, stride=1,in_channels_order=in_channels_order)),
        self.add_module('norm2', builder.batchnorm(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', builder.conv3x3(bn_size * growth_rate, growth_rate,stride=1)),

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,builder, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False,in_channels_order=None):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(builder,
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                in_channels_order=in_channels_order
            )
            in_channels_order += ',{}'.format(growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        self.out_channels_order = in_channels_order

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self,builder, num_input_features, num_output_features,in_channels_order=None):
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(num_input_features))
        # self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                   kernel_size=1, stride=1, bias=False))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        #
        self.add_module('norm', builder.batchnorm(num_input_features,in_channels_order=in_channels_order))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', builder.conv1x1(num_input_features, num_output_features,stride=1,in_channels_order=in_channels_order))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self,cfg,builder, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):

        super(DenseNet, self).__init__()
        slim_factor = cfg.slim_factor
        if slim_factor < 1:
            cfg.logger.info('WARNING: You are using a slim network')

        # num_init_features = math.ceil(num_init_features * slim_factor)
        # growth_rate = math.ceil(growth_rate * slim_factor)

        slim = lambda x: math.ceil(x * slim_factor)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', builder.conv7x7(3, slim(num_init_features), stride=2)),
            ('norm0', builder.batchnorm(slim(num_init_features))),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # in_channels_order = '{}'.format(num_features)
        for i, num_layers in enumerate(block_config):
            in_channels_order = '{}'.format(num_features)
            block = _DenseBlock(builder,
                num_layers=num_layers,
                num_input_features=slim(num_features),
                bn_size=bn_size,
                growth_rate=slim(growth_rate),
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                in_channels_order=in_channels_order,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            pre_num_features = num_features

            if i != len(block_config) - 1:
                num_features = num_features + num_layers * growth_rate
                trans = _Transition(builder,num_input_features=slim(pre_num_features) + num_layers * slim(growth_rate),
                                    num_output_features=slim(num_features // 2),in_channels_order=block.out_channels_order)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            else:
                num_features = slim(num_features) + num_layers * slim(growth_rate)


            # else:
            #     num_features = slim(num_features) + num_layers * slim(growth_rate)



        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('norm5', builder.batchnorm(num_features,in_channels_order=block.out_channels_order))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = builder.linear(num_features, cfg.num_cls, last_layer=True,
                                             in_channels_order=block.out_channels_order)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    common.load_state_dict(model, state_dict, strict=False)
    # model.load_state_dict(state_dict)


def _densenet(cfg,builder,arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(cfg,builder,growth_rate, block_config, num_init_features, **kwargs)
    if cfg.pretrained == 'imagenet':
        _load_state_dict(model, model_urls[arch], progress)
    return model


def Split_densenet121(cfg,pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet(cfg,get_builder(cfg),'densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def Split_densenet161(cfg,pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet(cfg,get_builder(cfg),'densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def Split_densenet169(cfg,pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet(cfg,get_builder(cfg),'densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def Split_densenet201(cfg,pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet(cfg,get_builder(cfg),'densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)
