import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

DenseLinear = nn.Linear



class SplitLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.split_mode = kwargs.pop('split_mode', None)
        split_rate = kwargs.pop('split_rate', None)
        last_layer = kwargs.pop('last_layer', None)
        self.in_channels_order = kwargs.pop('in_channels_order', None)

        self.split_rate = split_rate
        self.bias_split_rate = self.split_rate
        super().__init__(*args, **kwargs)

        ## AT : I am assuming a single FC layer in the network. Typical for most CNNs
        if self.split_mode == 'kels':
            if self.in_channels_order is None:
                if last_layer:
                    active_in_dim = math.ceil(self.weight.size()[1] * split_rate)
                    mask = np.zeros((self.weight.size()[0],self.weight.size()[1]))
                    mask[:,:active_in_dim] = 1
                else:
                    active_in_dim = math.ceil(self.weight.size()[1] * split_rate)
                    active_out_dim = math.ceil(self.weight.size()[0] * split_rate)
                    mask = np.zeros((self.weight.size()[0], self.weight.size()[1]))
                    mask[:active_out_dim, :active_in_dim] = 1
            else:
                mask = np.zeros((self.weight.size()[0], self.weight.size()[1]))
                conv_concat = self.in_channels_order.split(',')
                start_ch = 0
                for conv in conv_concat:
                    mask[:,start_ch:start_ch + math.ceil(int(conv) * split_rate)] = 1
                    start_ch += int(conv)

        elif self.split_mode == 'wels':
            mask = np.random.rand(*list(self.weight.shape))
            # threshold = np.percentile(scores, (1 - self.keep_rate) * 100)
            threshold = 1 - self.split_rate
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(mask)) == 2, 'Something is wrong with the mask {}'.format(np.unique(mask))
        else:
            raise NotImplemented('Invalid split_mode {}'.format(self.split_mode))

        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)

            # self.reset_scores()

    # def set_keep_rate(self, keep_rate, bias_keep_rate):
    #     self.split_rate = keep_rate
    #     self.bias_keep_rate = bias_keep_rate

    # def reset_scores(self):
    #     if self.split_mode == 'wels':
    #         scores = np.random.rand(*list(self.weight.shape))
    #         # threshold = np.percentile(scores, (1 - self.keep_rate) * 100)
    #         threshold = 1 - self.split_rate
    #         scores[scores < threshold] = 0
    #         scores[scores >= threshold] = 1
    #         if self.split_rate != 1:
    #             assert len(np.unique(scores)) == 2, 'Something is wrong with the score {}'.format(np.unique(scores))
    #     else:
    #         raise NotImplemented('Reset score randomly only with WELS. The current mode is '.format(self.split_mode))
    #     # active_in_dim = math.ceil(self.weight.size()[1] * self.keep_rate)
    #     # rand_sub = random.randint(0, self.weight.size()[1] - active_in_dim)
    #     # scores = np.zeros((self.weight.size()[0], self.weight.size()[1]))
    #     # scores[:, rand_sub:rand_sub+active_in_dim] = 1
    #     self.scores.data = torch.Tensor(scores).cuda()


    # def reset_bias_scores(self):
    #     pass

    def forward(self, x):
        ## Debugging purpose
        # if self.split_rate < 1:
        #     # subnet = GetSubnet.apply(self.clamped_scores, self.keep_rate)
        #     w = self.weight * self.scores
        # else:
        #     w = self.weight

        w = self.weight
        b = self.bias

        x =  F.linear(x, w, b)

        return x




