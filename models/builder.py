#from args import args
import math
import torch
import torch.nn as nn
import layers.conv_type
import layers.bn_type
import layers.linear_type


class Builder(object):
    def __init__(self, conv_layer, bn_layer,linear_layer,cfg=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.linear_layer = linear_layer
        self.first_layer = conv_layer
        self.cfg = cfg


    def linear(self,in_feat,out_feat,last_layer=False,in_channels_order=None):
        if self.linear_layer == nn.Linear:
            linear_layer = self.linear_layer(in_feat, out_feat)
        else:
            linear_layer =  self.linear_layer(in_feat,out_feat,split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,last_layer=last_layer,in_channels_order=in_channels_order)
        self._init_linear(linear_layer)
        return linear_layer

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            self.cfg.logger.info(f"==> Building first layer with {str(self.first_layer)}")

        if kernel_size == 3:
            if conv_layer == nn.Conv2d:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=bias,
                )
            else:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=bias,
                    split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,
                    in_channels_order=in_channels_order,
                )
        elif kernel_size == 1:
            if conv_layer == nn.Conv2d:
                conv = conv_layer(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                )
            else:
                conv = conv_layer(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                    split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,
                    in_channels_order=in_channels_order,
                )
        elif kernel_size == 5:
            if conv_layer == nn.Conv2d:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=5,
                    stride=stride,
                    padding=2,
                    bias=bias,
                )
            else:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=5,
                    stride=stride,
                    padding=2,
                    bias=bias,
                    split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,
                    in_channels_order=in_channels_order,
                )
        elif kernel_size == 7:
            if conv_layer == nn.Conv2d:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=7,
                    stride=stride,
                    padding=3,
                    bias=bias,
                )
            else:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=7,
                    stride=stride,
                    padding=3,
                    bias=bias,
                    split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,
                    in_channels_order=in_channels_order,
                )
        elif kernel_size == 11:
            if conv_layer == nn.Conv2d:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=11,
                    stride=stride,
                    padding=2,
                    bias=bias,
                )
            else:
                conv = conv_layer(
                    in_planes,
                    out_planes,
                    kernel_size=11,
                    stride=stride,
                    padding=2,
                    bias=bias,
                    split_mode=self.cfg.split_mode,
                    split_rate=self.cfg.split_rate,
                    in_channels_order=in_channels_order,
                )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer,bias=bias,in_channels_order=in_channels_order)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer,bias=bias,in_channels_order=in_channels_order)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer,bias=bias,in_channels_order=in_channels_order)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer,bias=bias,in_channels_order=in_channels_order)
        return c

    def conv11x11(self, in_planes, out_planes, stride=1, first_layer=False,bias=False,in_channels_order=None):
        """5x5 convolution with padding"""
        c = self.conv(11, in_planes, out_planes, stride=stride, first_layer=first_layer,bias=bias,in_channels_order=in_channels_order)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False,in_channels_order=None,**kwargs):
        if self.bn_layer == nn.BatchNorm2d:
            return self.bn_layer(planes, **kwargs)
        else:
            return self.bn_layer(planes,in_channels_order=in_channels_order,split_rate=self.cfg.split_rate,**kwargs)

    def activation(self):
        if self.cfg.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{self.cfg.nonlinearity} is not an initialization option!")

    def _init_linear(self, linear):
        if self.cfg.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(linear.weight, self.cfg.mode)
            if self.cfg.scale_fan:
                fan = fan * (1 - self.cfg.prune_rate)
            gain = nn.init.calculate_gain(self.cfg.nonlinearity)
            std = gain / math.sqrt(fan)
            linear.weight.data = linear.weight.data.sign() * std

        elif self.cfg.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(linear.weight, self.cfg.mode)
            if self.cfg.scale_fan:
                fan = fan * (1 - self.cfg.prune_rate)

            gain = nn.init.calculate_gain(self.cfg.nonlinearity)
            std = gain / math.sqrt(fan)
            linear.weight.data = torch.ones_like(linear.weight.data) * std

        elif self.cfg.init == "kaiming_normal":

            if self.cfg.scale_fan:
                fan = nn.init._calculate_correct_fan(linear.weight, self.cfg.mode)
                fan = fan * (1 - self.cfg.prune_rate)
                gain = nn.init.calculate_gain(self.cfg.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    linear.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    linear.weight, mode=self.cfg.mode, nonlinearity=self.cfg.nonlinearity
                )

        elif self.cfg.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                linear.weight, mode=self.cfg.mode, nonlinearity=self.cfg.nonlinearity
            )
        elif self.cfg.init == "xavier_normal":
            nn.init.xavier_normal_(linear.weight)
        elif self.cfg.init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            linear.weight.data = linear.weight.data.sign() * std

        elif self.cfg.init == "standard":
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{self.cfg.init} is not an initialization option!")


    def _init_conv(self, conv):
        if self.cfg.init == "signed_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, self.cfg.mode)
            if self.cfg.scale_fan:
                fan = fan * (1 - self.cfg.prune_rate)
            gain = nn.init.calculate_gain(self.cfg.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif self.cfg.init == "unsigned_constant":

            fan = nn.init._calculate_correct_fan(conv.weight, self.cfg.mode)
            if self.cfg.scale_fan:
                fan = fan * (1 - self.cfg.prune_rate)

            gain = nn.init.calculate_gain(self.cfg.nonlinearity)
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif self.cfg.init == "kaiming_normal":

            if self.cfg.scale_fan:
                fan = nn.init._calculate_correct_fan(conv.weight, self.cfg.mode)
                fan = fan * (1 - self.cfg.prune_rate)
                gain = nn.init.calculate_gain(self.cfg.nonlinearity)
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    conv.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(
                    conv.weight, mode=self.cfg.mode, nonlinearity=self.cfg.nonlinearity
                )

        elif self.cfg.init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode=self.cfg.mode, nonlinearity=self.cfg.nonlinearity
            )
        elif self.cfg.init == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)
        elif self.cfg.init == "xavier_constant":

            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif self.cfg.init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
        else:
            raise ValueError(f"{self.cfg.init} is not an initialization option!")


def get_builder(cfg):

    cfg.logger.info("==> Conv Type: {}".format(cfg.conv_type))
    cfg.logger.info("==> BN Type: {}".format(cfg.bn_type))

    conv_layer = getattr(layers.conv_type, cfg.conv_type)
    bn_layer = getattr(layers.bn_type, cfg.bn_type)
    linear_layer = getattr(layers.linear_type, cfg.linear_type)


    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer,linear_layer=linear_layer,cfg=cfg)

    return builder
