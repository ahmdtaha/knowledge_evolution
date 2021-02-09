import warnings
from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from models.builder import get_builder

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


def Split_googlenet(cfg, pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None
            model.aux2 = None
        return model

    return GoogLeNet(cfg,num_classes=cfg.num_cls,**kwargs)


class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self,cfg, num_classes=1000, aux_logits=False, transform_input=False, init_weights=None,
                 blocks=None): # AT : I disabled the aux_logits
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn('The default weight initialization of GoogleNet will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(blocks) == 3

        builder = get_builder(cfg)
        slim_factor = cfg.slim_factor
        if slim_factor < 1:
            cfg.logger.info('WARNING: You are using a slim network')

        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        slim = lambda x: math.ceil(x * slim_factor)
        self.conv1 = conv_block(builder,3, slim(64), kernel_size=7, stride=2) # , padding=3
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(builder,slim(64), slim(64), kernel_size=1)
        self.conv3 = conv_block(builder,slim(64), slim(192), kernel_size=3) # padding=1
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(builder, slim(192), slim(64)
                                           , slim(96), slim(128), slim(16), slim(32), slim(32))
        prev_out_channels = self.inception3a.out_channels # This is 256
        concat_order = self.inception3a.concat_order

        self.inception3b = inception_block(builder,prev_out_channels, slim(128), slim(128), slim(192), slim(32), slim(96), slim(64),in_channels_order=concat_order)
        prev_out_channels = self.inception3b.out_channels # This is 480
        concat_order = self.inception3b.concat_order

        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(builder,prev_out_channels, slim(192), slim(96), slim(208), slim(16), slim(48), slim(64),in_channels_order=concat_order)
        prev_out_channels = self.inception4a.out_channels # This is 512
        concat_order = self.inception4a.concat_order

        self.inception4b = inception_block(builder,prev_out_channels, slim(160), slim(112), slim(224), slim(24), slim(64), slim(64),in_channels_order=concat_order)
        prev_out_channels = self.inception4b.out_channels  # This is 512
        concat_order = self.inception4b.concat_order

        self.inception4c = inception_block(builder,prev_out_channels, slim(128), slim(128), slim(256), slim(24), slim(64), slim(64),in_channels_order=concat_order)
        prev_out_channels = self.inception4c.out_channels  # This is 512
        concat_order = self.inception4c.concat_order

        self.inception4d = inception_block(builder,prev_out_channels, slim(112), slim(144), slim(288), slim(32), slim(64), slim(64),in_channels_order=concat_order)
        prev_out_channels = self.inception4d.out_channels  # This is 528
        concat_order = self.inception4d.concat_order

        self.inception4e = inception_block(builder,prev_out_channels, slim(256), slim(160), slim(320), slim(32), slim(128), slim(128),in_channels_order=concat_order)
        prev_out_channels = self.inception4e.out_channels  # This is 832
        concat_order = self.inception4e.concat_order

        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(builder,prev_out_channels, slim(256), slim(160), slim(320), slim(32), slim(128), slim(128),in_channels_order=concat_order)
        prev_out_channels = self.inception5a.out_channels  # This is 832
        concat_order = self.inception5a.concat_order

        self.inception5b = inception_block(builder,prev_out_channels, slim(384), slim(192), slim(384), slim(48), slim(128), slim(128),in_channels_order=concat_order)
        prev_out_channels = self.inception5b.out_channels  # This is 1024
        concat_order = self.inception5b.concat_order

        if aux_logits:
            self.aux1 = inception_aux_block(builder,512, num_classes)
            self.aux2 = inception_aux_block(builder,528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        # self.fc = nn.Linear(1024, num_classes)
        self.fc = builder.linear(prev_out_channels, num_classes,last_layer=True,in_channels_order=concat_order)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)

        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)

        return x, aux2, aux1
        # return x

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x   # type: ignore[return-value]

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)

        # x = self._forward(x)
        # return x

        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):

    def __init__(self,builder ,in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 in_channels_order=None,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(builder,in_channels, ch1x1,in_channels_order=in_channels_order, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(builder, in_channels, ch3x3red,in_channels_order=in_channels_order, kernel_size=1),
            conv_block(builder, ch3x3red, ch3x3, kernel_size=3) # padding=1
        )

        self.branch3 = nn.Sequential(
            conv_block(builder, in_channels, ch5x5red,in_channels_order=in_channels_order, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(builder, ch5x5red, ch5x5, kernel_size=3) # padding=1
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(builder, in_channels, pool_proj,in_channels_order=in_channels_order, kernel_size=1)
        )
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        self.concat_order = '{},{},{},{}'.format(ch1x1,ch3x3,ch5x5,pool_proj)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        _res = torch.cat(outputs, 1)
        # print(_res.shape)
        return _res


class InceptionAux(nn.Module):

    def __init__(self, builder, in_channels, num_classes, slim_factor, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(builder,in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):

    def __init__(self,builder, in_channels, out_channels,in_channels_order=None, **kwargs):
        super(BasicConv2d, self).__init__()
        kernel_size = kwargs.pop('kernel_size', None)
        # self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        if kernel_size == 3:
            self.conv = builder.conv3x3( in_channels, out_channels , bias=False,in_channels_order=in_channels_order, **kwargs)
        elif kernel_size == 1:
            self.conv = builder.conv1x1(in_channels, out_channels, bias=False,in_channels_order=in_channels_order, **kwargs)
        elif kernel_size == 7:
            self.conv = builder.conv7x7(in_channels, out_channels, bias=False,in_channels_order=in_channels_order, **kwargs)
        elif kernel_size == 11:
            self.conv = builder.conv11x11(in_channels, out_channels, bias=False,in_channels_order=in_channels_order, **kwargs)
        elif kernel_size == 5:
            self.conv = builder.conv5x5(in_channels, out_channels, bias=False,in_channels_order=in_channels_order, **kwargs)
        else:
            raise NotImplemented("Invalid kernel size {}".format(kernel_size))

        self.bn = builder.batchnorm(out_channels , eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
