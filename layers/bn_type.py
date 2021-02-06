import math
import torch
import numpy as np
import torch.nn as nn

NormalBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim,**kwargs):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False,**kwargs)

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim,**kwargs):
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        split_rate = kwargs.pop('split_rate', None)
        super(SplitBatchNorm, self).__init__(dim, affine=True, **kwargs)

        if self.in_channels_order is not None:
            assert split_rate is not None, 'Should not be none if in_channels_order is not None'
            mask = np.zeros(self.weight.size()[0])
            conv_concat = self.in_channels_order.split(',')
            start_ch = 0
            for conv in conv_concat:
                mask[start_ch:start_ch + math.ceil(int(conv) * split_rate)] = 1
                start_ch += int(conv)

            self.bn_mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)

    def extract_slim(self,dst_m):
        c_out = self.weight.size()[0]
        d_out = dst_m.weight.size()[0]
        if self.in_channels_order is None:
            assert dst_m.weight.shape == self.weight[:d_out].shape
            dst_m.weight.data = self.weight[:d_out]
            dst_m.bias.data = self.bias[:d_out]
            dst_m.running_mean.data = self.running_mean[:d_out]
            dst_m.running_var.data = self.running_var[:d_out]
        else:
            assert dst_m.weight.shape == self.weight[self.bn_mask == 1].shape
            dst_m.weight.data = self.weight[self.bn_mask == 1]
            dst_m.bias.data = self.bias.data[self.bn_mask == 1]
            dst_m.running_mean.data = self.running_mean[self.bn_mask == 1]
            dst_m.running_var.data = self.running_var[self.bn_mask == 1]
