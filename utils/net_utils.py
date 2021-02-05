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

    args.logger.info("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args)

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


