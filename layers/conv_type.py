import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from configs.base_config import args as parser_args


DenseConv = nn.Conv2d


# Not learning weights, finding subnet
class SplitConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.split_mode = kwargs.pop('split_mode', None)
        self.split_rate = kwargs.pop('split_rate', None)
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        # self.keep_rate = keep_rate
        super().__init__(*args, **kwargs)

        if self.split_mode == 'kels':
            if self.in_channels_order is None:
                mask = np.zeros((self.weight.size()))
                if self.weight.size()[1] == 3: ## This is the first conv
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :, :, :] = 1
                else:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), :math.ceil(self.weight.size()[1] * self.split_rate), :, :] = 1
            else:

                mask = np.zeros((self.weight.size()))
                conv_concat = [int(chs) for chs in self.in_channels_order.split(',')]
                # assert sum(conv_concat) == self.weight.size()[1],'In channels {} should be equal to sum(concat) {}'.format(self.weight.size()[1],conv_concat)
                start_ch = 0
                for conv in conv_concat:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate), start_ch:start_ch + math.ceil(conv * self.split_rate),
                    :, :] = 1
                    start_ch += conv

        elif self.split_mode == 'wels':
            mask = np.random.rand(*list(self.weight.shape))
            # threshold = np.percentile(scores, (1-self.keep_rate)*100)
            threshold = 1 - self.split_rate
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2,'Something is wrong with the mask {}'.format(np.unique(mask))
        else:
            raise NotImplemented('Invalid split_mode {}'.format(self.split_mode))

        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)


    # def reset_scores(self):
    #     if self.split_mode == 'wels':
    #         mask = np.random.rand(*list(self.weight.shape))
    #         threshold = 1 - self.split_rate
    #         mask[mask < threshold] = 0
    #         mask[mask >= threshold] = 1
    #         if self.split_rate != 1:
    #             assert len(np.unique(mask)) == 2,'Something is wrong with the score {}'.format(np.unique(mask))
    #     else:
    #         raise NotImplemented('Reset score randomly only with WELS. The current mode is '.format(self.split_mode))
    #     # scores = np.zeros((self.weight.size()))
    #     # rand_sub = random.randint(0, self.weight.size()[0] - math.ceil(self.weight.size()[0] * self.keep_rate))
    #     # if self.weight.size()[1] == 3:  ## This is the first conv
    #     #     scores[rand_sub:rand_sub+math.ceil(self.weight.size()[0] * self.keep_rate), :, :, :] = 1
    #     # else:
    #     #     scores[rand_sub:rand_sub+math.ceil(self.weight.size()[0] * self.keep_rate), :math.ceil(self.weight.size()[1] * self.keep_rate), :,
    #     #     :] = 1
    #
    #     self.mask.data = torch.Tensor(mask).cuda()
    #     # raise NotImplemented('Not implemented yet')
    #     # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    # def reset_bias_scores(self):
    #     pass

    # def set_split_rate(self, split_rate, bias_split_rate):
    #     self.split_rate = split_rate
    #     if self.bias is not None:
    #         self.bias_split_rate = bias_split_rate
    #     else:
    #         self.bias_split_rate = 1.0

    def forward(self, x):
        ## Debugging reasons only
        # if self.split_rate < 1:
        #     w = self.mask * self.weight
        #     if self.bias_split_rate < 1:
        #         # bias_subnet = GetSubnet.apply(self.clamped_bias_scores, self.bias_keep_rate)
        #         b = self.bias * self.mask[:, 0, 0, 0]
        #     else:
        #         b = self.bias
        # else:
        #     w = self.weight
        #     b = self.bias

        w = self.weight
        b = self.bias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

