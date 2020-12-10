import os
import math
import torch
import shutil
import models
import pathlib
import numpy as np
import torch.nn as nn
from scipy import ndimage
from layers import conv_type
from layers import linear_type
import torch.backends.cudnn as cudnn
# from layers.conv_type import FixedSubnetConv


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    args.logger.info("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args)

    # applying sparsity to the network
    if (
            args.conv_type != "DenseConv"
            and args.conv_type != "SampleSubnetConv"
            and args.conv_type != "ContinuousSparseConv"
    ):
        if args.split_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        # set_model_split_rate(model,args.split_rate,args.bias_split_rate,args)
        args.logger.info(
            f"=> Rough estimate model params {sum(int(p.numel() * (1 - args.split_rate)) for n, p in model.named_parameters() if not n.endswith('mask'))}"
        )

    # freezing the weights if we are only doing subnet training
    # if args.freeze_weights:
    #     freeze_model_weights(model)

    return model



def move_model_to_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    # print('{}'.format(args.gpu))
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        args.logger.info(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


# def freeze_model_weights(model,cfg):
#     cfg.logger.info("=> Freezing model weights")
#
#     for n, m in model.named_modules():
#         if hasattr(m, "weight") and m.weight is not None:
#             cfg.logger.info(f"==> No gradient to {n}.weight")
#             m.weight.requires_grad = False
#             if m.weight.grad is not None:
#                 cfg.logger.info(f"==> Setting gradient of {n}.weight to None")
#                 m.weight.grad = None
#
#             if hasattr(m, "bias") and m.bias is not None:
#                 cfg.logger.info(f"==> No gradient to {n}.bias")
#                 m.bias.requires_grad = False
#
#                 if m.bias.grad is not None:
#                     cfg.logger.info(f"==> Setting gradient of {n}.bias to None")
#                     m.bias.grad = None


# def freeze_model_scores(model,cfg):
#     cfg.logger.info("=> Freezing model subnet")
#
#     for n, m in model.named_modules():
#         if hasattr(m, "scores"):
#             m.scores.requires_grad = False
#             cfg.logger.info(f"==> No gradient to {n}.scores")
#             if m.scores.grad is not None:
#                 cfg.logger.info(f"==> Setting gradient of {n}.scores to None")
#                 m.scores.grad = None
#         if hasattr(m, "bias_scores"):
#             m.bias_scores.requires_grad = False
#             cfg.logger.info(f"==> No gradient to {n}.bias_scores")
#             if m.bias_scores.grad is not None:
#                 cfg.logger.info(f"==> Setting gradient of {n}.bias_scores to None")
#                 m.bias_scores.grad = None


# def unfreeze_model_weights(model,cfg):
#     cfg.logger.info("=> Unfreezing model weights")
#
#     for n, m in model.named_modules():
#         if hasattr(m, "weight") and m.weight is not None:
#             cfg.logger.info(f"==> Gradient to {n}.weight")
#             m.weight.requires_grad = True
#             if hasattr(m, "bias") and m.bias is not None:
#                 cfg.logger.info(f"==> Gradient to {n}.bias")
#                 m.bias.requires_grad = True


# def unfreeze_model_scores(model,cfg):
#     cfg.logger.info("=> Unfreezing model subnet")
#
#     for n, m in model.named_modules():
#         if hasattr(m, "scores"):
#             cfg.logger.info(f"==> Gradient to {n}.scores")
#             m.scores.requires_grad = True
#
#         if hasattr(m, "bias_scores"):
#             cfg.logger.info(f"==> Gradient to {n}.bias_scores")
#             m.bias_scores.requires_grad = True


# def set_model_split_rate(model, split_rate, bias_split_rate, cfg):
#     cfg.logger.info(f"==> Setting prune rate of network to {split_rate}")
#
#     for n, m in model.named_modules():
#         if hasattr(m, "set_split_rate"):
#             m.set_split_rate(split_rate, bias_split_rate)


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_reinitialize(cfg,model,reset_hypothesis=False):
    cfg.logger.info('split_reinitialize')
    # zero_reset = True
    if cfg.evolve_mode == 'zero':
        cfg.logger.info('WARNING: ZERO RESET is not optimal')
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if hasattr(m, "mask"):
                # print(m.prune_rate)
                assert m.split_rate < 1.0

                if reset_hypothesis and (m.__class__ == conv_type.SplitConv or  m.__class__ == linear_type.SplitLinear):
                    before_sum = torch.sum(m.mask)
                    # cfg.logger.info('reset_hypothesis : True {}'.format())
                    m.reset_mask()
                    cfg.logger.info('reset_hypothesis : True {} : {} -> {}'.format(n,before_sum,torch.sum(m.mask)))
                else:
                    cfg.logger.info('reset_hypothesis : False {} : {}'.format(n, torch.sum(m.mask)))
                layer_mask = m.mask
                if m.__class__ == conv_type.SplitConv:
                    if cfg.evolve_mode == 'rand':
                        rand_tensor = torch.zeros_like(m.weight).cuda()
                        nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
                        m.weight.data = torch.where(layer_mask.type(torch.bool), m.weight.data, rand_tensor)
                    else:
                        raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))

                elif m.__class__ == linear_type.SplitLinear:
                    if cfg.evolve_mode == 'rand':
                        rand_tensor = torch.zeros_like(m.weight).cuda()
                        nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
                        m.weight.data = torch.where(layer_mask.type(torch.bool), m.weight.data, rand_tensor)
                    else:
                        raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
                else:
                    raise NotImplemented('Invalid layer {}'.format(m.__class__))

                if hasattr(m, "bias") and m.bias is not None and m.bias_split_rate < 1.0:
                    cfg.logger.info('Cashing on the bias term as well')
                    if m.__class__ == conv_type.SplitConv:
                        bias_mask = layer_mask[:,0,0,0] ## Same conv mask is used for bias terms
                        if cfg.evolve_mode == 'rand':
                            rand_tensor = torch.zeros_like(m.bias)
                            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                            bound = 1 / math.sqrt(fan_in)
                            nn.init.uniform_(rand_tensor, -bound, bound)

                            m.bias.data = torch.where(bias_mask.type(torch.bool), m.bias.data, rand_tensor)
                        else:
                            raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))


# def cash_supermask(cfg,model):
#     cfg.logger.info('Cashing the supermask weights')
#     # zero_reset = True
#     if cfg.evolve_mode == 'zero':
#         cfg.logger.info('WARNING: ZERO RESET is not optimal')
#     for n, m in model.named_modules():
#         if hasattr(m, "weight") and m.weight is not None:
#             if hasattr(m, "scores"):
#                 # print(m.prune_rate)
#                 assert m.keep_rate < 1.0
#
#                 score_mask = conv_type.GetSubnet.apply(m.clamped_scores, m.keep_rate).type(torch.bool)
#                 if m.__class__ == conv_type.SubnetCoarseConv:
#                     # num_convs = int(torch.sum(~score_mask).cpu().numpy())
#                     # selected_convs = m.weight[score_mask.squeeze()]
#                     if cfg.evolve_mode == 'zero':
#                         zero_tensor = torch.zeros(m.weight.size()).cuda()
#                         m.weight.data = torch.where(score_mask, m.weight.data,zero_tensor)
#
#                         # m.weight.data = torch.cat([selected_convs, zero_tensor], dim=0)
#                     elif cfg.evolve_mode == 'ke':
#                         # shuffle_idx = [torch.randperm(np.prod(m.weight.size()[1:])) for _ in range(num_convs)]
#                         # selected_conv = np.random.choice(selected_convs.size()[0],size=num_convs,replace=True)
#                         # shuffled_convs = torch.stack([selected_convs[selected_conv_idx].view(-1)[shuffle_idx[i]].view(m.weight.size()[1:]) for i, selected_conv_idx in enumerate(selected_conv)], dim=0)
#                         #
#                         # m.weight.data = torch.cat([selected_convs,shuffled_convs],dim=0)
#
#                         selected_weights = m.weight[score_mask.squeeze()]
#                         randomly_chosen_weights = np.random.choice(torch.flatten(selected_weights.data).cpu(), size=m.weight.shape,replace=True)
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,
#                                                     torch.tensor(randomly_chosen_weights).cuda())
#
#                         # num_convs,num_channels,spatial_dim = m.weight.size()[0:3]
#                         # shuffle_tensors = []
#                         # conv_shuffle = np.random.choice(np.array(torch.nonzero(score_mask,as_tuple=False)[:,0].cpu()),size=num_convs,replace=True)
#                         # for conv_idx in conv_shuffle:
#                         #     # channel_shuffle = torch.randperm(num_channels)
#                         #     spatial_shuffle = torch.randperm(spatial_dim * spatial_dim)
#                         #     for channel_idx in range(num_channels):
#                         #         shuffle_tensors.append(m.weight[conv_idx, channel_idx].view(-1)[spatial_shuffle])
#                         #
#                         # shuffle_tensors = torch.stack(shuffle_tensors, dim=0)
#                         # shuffle_tensors = shuffle_tensors.reshape(m.weight.size()).cuda()
#                         # m.weight.data = torch.where(score_mask, m.weight.data, shuffle_tensors)
#
#                     elif cfg.evolve_mode == 'ke_rot':
#                         num_convs,num_channels,spatial_dim = m.weight.size()[0:3]
#                         shuffle_tensors = []
#                         cpu_tensor_numpy = m.weight.cpu().detach().numpy()
#                         conv_shuffle = np.random.choice(np.array(torch.nonzero(score_mask,as_tuple=False)[:,0].cpu()),size=num_convs,replace=True)
#                         for conv_idx in conv_shuffle:
#                             rot_angle = np.random.choice([90,180,270])
#                             for channel_idx in range(num_channels):
#                                 shuffle_tensors.append(ndimage.rotate(cpu_tensor_numpy[conv_idx,channel_idx],rot_angle,mode='nearest'))
#
#                         shuffle_tensors = np.stack(shuffle_tensors, axis=0)
#                         shuffle_tensors = torch.tensor(shuffle_tensors).reshape(m.weight.size()).cuda()
#                         m.weight.data = torch.where(score_mask, m.weight.data, shuffle_tensors)
#
#                     elif cfg.evolve_mode == 'rand':
#                         # num_convs = int(torch.sum(~score_mask).cpu().numpy())
#                         # selected_convs = m.weight[score_mask.squeeze()]
#
#                         # rand_tensor = torch.normal(mean=0,std=1,size=m.weight.size()).cuda()
#                         rand_tensor = torch.zeros_like(m.weight).cuda()
#                         nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,rand_tensor)
#                     else:
#                         raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#
#                 else:
#                     if cfg.evolve_mode == 'zero':
#                         m.weight.data = torch.where(score_mask.type(torch.bool),m.weight.data,torch.tensor(0,dtype=torch.float).cuda())
#                     elif cfg.evolve_mode == 'ke' or cfg.evolve_mode == 'ke_rot':
#                         selected_weights = m.weight[score_mask]
#                         #
#                         randomly_chosen_weights = np.random.choice(selected_weights.data.cpu(), size=m.weight.shape,
#                                                                    replace=True)
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,
#                                                     torch.tensor(randomly_chosen_weights).cuda())
#
#                         # plt.hist(selected_weights.cpu().detach().numpy())
#                         # plt.hist(m.weight.data[:].cpu().detach().numpy().flat)
#                         # plt.show()
#                         # plt.savefig('./pre_hist.png')
#                         # plt.close()
#
#                         # sampling_prob = abs(selected_weights.data.cpu().numpy())
#                         # sampling_prob /= sum(sampling_prob)
#                         # , p = sampling_prob
#                     elif cfg.evolve_mode == 'rand':
#                         rand_tensor = torch.zeros_like(m.weight).cuda()
#                         nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,rand_tensor)
#                     else:
#                         raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#
#                 if hasattr(m, "bias") and m.bias is not None and m.bias_keep_rate < 1.0:
#                     cfg.logger.info('Cashing on the bias term as well')
#                     if m.__class__ == conv_type.SubnetCoarseConv:
#                         bias_score_mask = conv_type.GetSubnet.apply(m.clamped_scores.squeeze(), m.bias_keep_rate).type(
#                             torch.bool)
#                     # else:
#                     #     bias_score_mask = conv_type.GetSubnet.apply(m.clamped_bias_scores, m.bias_keep_rate).type(torch.bool)
#
#                         if cfg.evolve_mode == 'zero':
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data, torch.tensor(0,dtype=torch.float).cuda())
#                         elif cfg.evolve_mode == 'ke' or cfg.evolve_mode == 'ke_rot':
#                             selected_bias = m.bias[bias_score_mask]
#                             # sampling_prob = abs(selected_bias.data.cpu().numpy())
#                             # sampling_prob /= sum(sampling_prob)
#                             # randomly_chosen_bias = np.random.choice(selected_bias.data.cpu(), m.bias.shape,replace=True,p=sampling_prob)
#                             randomly_chosen_bias = np.random.choice(selected_bias.data.cpu(), m.bias.shape,replace=True)
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data,
#                                                         torch.tensor(randomly_chosen_bias).cuda())
#                         elif cfg.evolve_mode == 'rand':
#                             rand_tensor = torch.zeros_like(m.bias)
#                             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                             bound = 1 / math.sqrt(fan_in)
#                             nn.init.uniform_(rand_tensor, -bound, bound)
#
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data,rand_tensor)
#                         else:
#                             raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#
#
# def pick_random_hypothesis(cfg,model):
#     cfg.logger.info('pick random hypothesis')
#     # zero_reset = True
#     if cfg.evolve_mode == 'zero':
#         cfg.logger.info('WARNING: ZERO RESET is not optimal')
#     for n, m in model.named_modules():
#         if hasattr(m, "weight") and m.weight is not None:
#             if hasattr(m, "scores"):
#                 assert m.keep_rate < 1.0
#
#                 m.reset_scores()
#                 score_mask = conv_type.GetSubnet.apply(m.clamped_scores, m.keep_rate).type(torch.bool)
#                 if m.__class__ == conv_type.SubnetCoarseConv:
#                     if cfg.evolve_mode == 'zero':
#                         zero_tensor = torch.zeros(m.weight.size()).cuda()
#                         m.weight.data = torch.where(score_mask, m.weight.data,zero_tensor)
#
#                     elif cfg.evolve_mode == 'ke':
#
#                         selected_weights = m.weight[score_mask.squeeze()]
#                         randomly_chosen_weights = np.random.choice(torch.flatten(selected_weights.data).cpu(), size=m.weight.shape,replace=True)
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,
#                                                     torch.tensor(randomly_chosen_weights).cuda())
#                     elif cfg.evolve_mode == 'ke_rot':
#                         num_convs,num_channels,spatial_dim = m.weight.size()[0:3]
#                         shuffle_tensors = []
#                         cpu_tensor_numpy = m.weight.cpu().detach().numpy()
#                         conv_shuffle = np.random.choice(np.array(torch.nonzero(score_mask,as_tuple=False)[:,0].cpu()),size=num_convs,replace=True)
#                         for conv_idx in conv_shuffle:
#                             rot_angle = np.random.choice([90,180,270])
#                             for channel_idx in range(num_channels):
#                                 shuffle_tensors.append(ndimage.rotate(cpu_tensor_numpy[conv_idx,channel_idx],rot_angle,mode='nearest'))
#
#                         shuffle_tensors = np.stack(shuffle_tensors, axis=0)
#                         shuffle_tensors = torch.tensor(shuffle_tensors).reshape(m.weight.size()).cuda()
#                         m.weight.data = torch.where(score_mask, m.weight.data, shuffle_tensors)
#
#                     elif cfg.evolve_mode == 'rand':
#                         rand_tensor = torch.zeros_like(m.weight).cuda()
#                         nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,rand_tensor)
#                     else:
#                         raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#
#                 else:
#                     if cfg.evolve_mode == 'zero':
#                         m.weight.data = torch.where(score_mask.type(torch.bool),m.weight.data,torch.tensor(0,dtype=torch.float).cuda())
#                     elif cfg.evolve_mode == 'ke' or cfg.evolve_mode == 'ke_rot':
#                         selected_weights = m.weight[score_mask]
#                         #
#                         randomly_chosen_weights = np.random.choice(selected_weights.data.cpu(), size=m.weight.shape,
#                                                                    replace=True)
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,
#                                                     torch.tensor(randomly_chosen_weights).cuda())
#
#                     elif cfg.evolve_mode == 'rand':
#                         rand_tensor = torch.zeros_like(m.weight).cuda()
#                         nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
#                         m.weight.data = torch.where(score_mask.type(torch.bool), m.weight.data,rand_tensor)
#                     else:
#                         raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#
#                 if hasattr(m, "bias") and m.bias is not None and m.bias_keep_rate < 1.0:
#                     cfg.logger.info('Cashing on the bias term as well')
#                     m.reset_bias_scores()
#                     if m.__class__ == conv_type.SubnetCoarseConv:
#                         bias_score_mask = conv_type.GetSubnet.apply(m.clamped_scores.squeeze(), m.bias_keep_rate).type(
#                             torch.bool)
#                     # else:
#                     #     bias_score_mask = conv_type.GetSubnet.apply(m.clamped_bias_scores, m.bias_keep_rate).type(torch.bool)
#
#                         if cfg.evolve_mode == 'zero':
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data, torch.tensor(0,dtype=torch.float).cuda())
#                         elif cfg.evolve_mode == 'ke' or cfg.evolve_mode == 'ke_rot':
#                             selected_bias = m.bias[bias_score_mask]
#                             randomly_chosen_bias = np.random.choice(selected_bias.data.cpu(), m.bias.shape,replace=True)
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data,
#                                                         torch.tensor(randomly_chosen_bias).cuda())
#                         elif cfg.evolve_mode == 'rand':
#                             rand_tensor = torch.zeros_like(m.bias)
#                             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
#                             bound = 1 / math.sqrt(fan_in)
#                             nn.init.uniform_(rand_tensor, -bound, bound)
#
#                             m.bias.data = torch.where(bias_score_mask.type(torch.bool), m.bias.data,rand_tensor)
#                         else:
#                             raise NotImplemented('Invalid KE mode {}'.format(cfg.evolve_mode))
#

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def reset_mask(cfg,model):
    cfg.logger.info("=> reseting model mask")

    for n, m in model.named_modules():
        if hasattr(m, "mask"):
            cfg.logger.info(f"==> reset {n}.mask")
            # m.mask.requires_grad = True
            m.reset_mask()

        if hasattr(m, "bias_mask"):
            cfg.logger.info(f"==> reset {n}.bias_mask")
            m.reset_bias_mask()
            # m.bias_mask.requires_grad = True

def load_pretrained(pretrained_path,gpus, model,cfg):
    if os.path.isfile(pretrained_path):
        cfg.logger.info("=> loading pretrained weights from '{}'".format(pretrained_path))
        pretrained = torch.load(
            pretrained_path,
            map_location=torch.device("cuda:{}".format(gpus)),
        )["state_dict"]
        skip = ' '
        # skip = 'mask'
        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            # if k not in model_state_dict or v.size() != model_state_dict[k].size():
            if k not in model_state_dict or v.size() != model_state_dict[k].size() or skip in k:
                cfg.logger.info("IGNORE: {}".format(k))
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size() and skip not in k)
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        cfg.logger.info("=> no pretrained weights found at '{}'".format(pretrained_path))

    # for n, m in model.named_modules():
    #     if isinstance(m, FixedSubnetConv):
    #         m.set_subnet()

class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("mask"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum


