import math
import torch
import numpy as np
import torch.nn as nn

NormalBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim,**kwargs):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False,**kwargs)

class AffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim,**kwargs):
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        split_rate = kwargs.pop('split_rate', None)
        super(AffineBatchNorm, self).__init__(dim, affine=True,**kwargs)

        if self.in_channels_order is not None:
            assert split_rate is not None, 'Should not be none if in_channels_order is not None'
            mask = np.zeros(self.weight.size()[0])
            conv_concat = self.in_channels_order.split(',')
            start_ch = 0
            for conv in conv_concat:
                mask[start_ch:start_ch + math.ceil(int(conv) * split_rate)] = 1
                start_ch += int(conv)

            self.bn_mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)
