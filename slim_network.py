import os
import math
import data
import torch
import getpass
import importlib
import os.path as osp
import torch.nn as nn
import KE_model
from layers import bn_type
# from utils import os_utils
from utils import net_utils
from layers import conv_type
# from utils import path_utils
# from utils import push_utils
from layers import linear_type
from utils import model_profile
from configs.base_config import Config

def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier

def extract_slim(split_model,model):
    for (dst_n, dst_m), (src_n, src_m) in zip(split_model.named_modules(), model.named_modules()):
        # print(src_n,dst_n)
        if hasattr(src_m, "weight") and src_m.weight is not None:
            # print(src_n, src_m.weight.size(), dst_n, dst_m.weight.size())
            if hasattr(src_m, "mask"):
                if src_m.__class__ == conv_type.SplitConv:
                    c_out, c_in, _, _, = src_m.weight.size()
                    d_out, d_in, _, _ = dst_m.weight.size()
                    if src_m.in_channels_order is None:
                        if c_in == 3:
                            selected_convs = src_m.weight[:d_out]
                            # is_first_conv = False
                        else:
                            selected_convs = src_m.weight[:d_out][:, :d_in, :,:]

                        assert selected_convs.shape == dst_m.weight.shape
                        dst_m.weight.data = selected_convs
                    else:
                        selected_convs = src_m.weight[:d_out,src_m.mask[0,:,0,0] == 1, :, :]
                        assert selected_convs.shape == dst_m.weight.shape,'{} {} {} {}'.format(dst_n,src_n,dst_m.weight.shape,selected_convs.shape)
                        dst_m.weight.data = selected_convs
                        # conv_concat = src_m.in_channels_order.split(',')
                        # full_start_ch = 0
                        # slim_start_ch = 0
                        # for conv in conv_concat:
                        #     num_in_channels =math.ceil(int(conv) * src_m.keep_rate)
                        #     dst_m.weight.data[:d_out,slim_start_ch:slim_start_ch+num_in_channels,:,:] \
                        #         = src_m.weight[:d_out][:, full_start_ch:full_start_ch+num_in_channels, :, :]
                        #
                        #     full_start_ch += int(conv)
                        #     slim_start_ch += num_in_channels

                elif src_m.__class__ == linear_type.SplitLinear:
                    c_out, c_in = src_m.weight.size()
                    d_out, d_in = dst_m.weight.size()

                    if src_m.in_channels_order is None:
                        assert dst_m.weight.shape == src_m.weight[:d_out,:d_in].shape
                        dst_m.weight.data = src_m.weight.data[:d_out,:d_in]
                        assert dst_m.bias.data.shape == src_m.bias.data[:d_out].shape
                        dst_m.bias.data = src_m.bias.data[:d_out]
                    else:
                        dst_m.weight.data = src_m.weight[:d_out,src_m.mask[0,:] == 1]
                        dst_m.bias.data = src_m.bias.data[:d_out]

            elif src_m.__class__ == bn_type.SplitBatchNorm:
                c_out = src_m.weight.size()[0]
                d_out = dst_m.weight.size()[0]
                if src_m.in_channels_order is None:
                    assert dst_m.weight.shape == src_m.weight[:d_out].shape
                    dst_m.weight.data = src_m.weight[:d_out]
                    dst_m.bias.data = src_m.bias[:d_out]
                    dst_m.running_mean.data = src_m.running_mean[:d_out]
                    dst_m.running_var.data = src_m.running_var[:d_out]
                else:
                    assert dst_m.weight.shape == src_m.weight[src_m.bn_mask == 1].shape
                    dst_m.weight.data = src_m.weight[src_m.bn_mask == 1]
                    dst_m.bias.data = src_m.bias.data[src_m.bn_mask == 1]
                    dst_m.running_mean.data = src_m.running_mean[src_m.bn_mask == 1]
                    dst_m.running_var.data = src_m.running_var[src_m.bn_mask == 1]

def main():
    arg_dataset = 'Flower102'
    arg_epochs = str(200)
    arg_evolve_mode = 'rand'
    arg_reset_mask = False
    arg_reset_hypothesis = False
    arg_pretrained = False  # imagnet or False
    arg_enable_cs_kd = False
    arg_enable_label_smoothing = True
    arg_arch = 'Split_ResNet18'  # ResNet18,densenet169,vgg11_bn
    arg_keep_top = '0.8'
    arg_bias_keep_top = '0.8'
    arg_num_generations = '10'
    # exp_name_suffix = 'cls_study'
    exp_name_suffix = '_rand_hypothesis_mask'
    arg_exp_name = 'SPLT_CLS_{}_{}_cskd{}_smth{}_imagenet{}_k{}_G{}_e{}_ev{}_hReset{}_{}/'.format(arg_dataset, arg_arch,
                                                                                                  arg_enable_cs_kd,
                                                                                                  arg_enable_label_smoothing,
                                                                                                  arg_pretrained,
                                                                                                  arg_keep_top,
                                                                                                  arg_num_generations,
                                                                                                  arg_epochs,
                                                                                                  arg_evolve_mode,
                                                                                                  arg_reset_hypothesis,
                                                                                                  exp_name_suffix)

    if arg_arch in ['alexnet', 'vgg11', 'vgg11_bn']:
        arg_weight_decay = '5e-4'
        arg_init = 'kaiming_normal'
    else:
        arg_weight_decay = '1e-4'
        arg_init = 'kaiming_normal'
    # arg_exp_name = 'CUB200_ResNet18_cskdFalse_smthTrue_cache_enabled_cosine_lr/'

    argv = [
        '--log_file', 'split_log.txt',
        '--name', arg_exp_name,
        '--evolve_mode', arg_evolve_mode,
        '--num_threads', '16',
        '--gpu', '0',
        '--epochs', arg_epochs,
        '--arch', arg_arch,
        # '--trainer', 'default', #'default', #lottery, # supermask

        '--data', '/mnt/data/datasets/',
        '--set', arg_dataset,  # Flower102, CUB200

        '--optimizer', 'sgd',
        # '--lr', '0.1',
        # '--lr_policy', 'step_lr',
        # '--warmup_length', '5',

        '--lr_policy', 'cosine_lr',
        '--warmup_length', '5',

        '--weight_decay', arg_weight_decay,
        '--momentum', '0.9',
        '--batch_size', '32',

        # '--conv_type', 'SubnetCoarseConv',  # 'SubnetConv','StrictSubnetConv
        # '--bn_type', 'SplitBatchNorm',
        # '--linear_type', 'SubnetLinear',

        '--conv_type', 'SplitConv',  # 'SubnetConv','StrictSubnetConv
        '--bn_type', 'SplitBatchNorm',
        '--linear_type', 'SplitLinear',

        '--keep_rate', arg_keep_top,
        '--bias_keep_rate', arg_bias_keep_top,
        '--init', arg_init,  # xavier_normal, kaiming_normal
        '--mode', 'fan_in',
        '--nonlinearity', 'relu',

        '--num_generations', arg_num_generations,
    ]

    if arg_enable_cs_kd:
        argv.extend(['--cs_kd'])

    if arg_enable_label_smoothing:
        argv.extend(['--label_smoothing', '0.1'])

    if arg_pretrained:
        argv.extend(['--pretrained', 'imagenet',
                     '--num_generations', '10',
                     '--lr', '0.00256'])
    else:
        if arg_arch in ['alexnet', 'vgg11', 'vgg11_bn']:
            argv.extend(['--lr', '0.0256'])
        else:
            argv.extend(['--lr', '0.256'])

    if arg_reset_mask:
        argv.extend(['--reset_mask'])

    if arg_reset_hypothesis:
        argv.extend(['--reset_hypothesis'])


    cfg = Config().parse(argv)

    split_model_path = '/mnt/data/checkpoints/task_mask/KE_CLS_Flower102_Split_ResNet18_cskdFalse_smthTrue_imagenetFalse_k0.5_G10_e200_evrand_ScoreFalse_rand_hypothesis_mask/task/keep_rate=1.0/0009/checkpoints/epoch_199.state'
    split_model_path = '/mnt/data/checkpoints/task_mask/SPLT_CLS_Flower102_Split_ResNet18_cskdFalse_smthTrue_imagenetFalse_k0.8_G10_e200_evrand_hResetFalse_re-start/task/keep_rate=1.0/0000/checkpoints/epoch_199_debug.state'
    split_model_path = '/mnt/data/checkpoints/task_mask/SPLT_CLS_Flower102_Split_ResNet18_cskdFalse_smthTrue_imagenetFalse_k0.8_G10_e200_evrand_hResetFalse_re-start/task/keep_rate=1.0/0001/checkpoints/epoch_199.state'
    split_model_path = '/mnt/data/checkpoints/task_mask/SPLT_CLS_Flower102_Split_ResNet18_cskdFalse_smthTrue_imagenetFalse_k0.8_G10_e20_evrand_hResetFalse_re-start/task/keep_rate=1.0/0000/checkpoints/epoch_19.state'
    split_model_path = '/mnt/data/checkpoints/task_mask/SPLT_CLS_Flower102_Split_ResNet18_cskdTrue_smthFalse_imagenetFalse_k0.5_G100_e200_evrand_hResetFalse_smkels_cls_study_fix_wels_w_cache_2/task/keep_rate=1.0/0000/checkpoints/epoch_199.state'

    cfg.trainer = 'default_cls'
    epoch = 1
    writer = None
    cfg.pretrained = split_model_path
    softmax_criterion = nn.CrossEntropyLoss().cuda()
    cfg.keep_rate = 1.0
    cfg.bias_keep_rate = 1.0

    model = net_utils.get_model(cfg)
    net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model,cfg)
    # cfg.gpu = cfg.multigpu[0]
    model = net_utils.move_model_to_gpu(cfg, model)



    dataset = getattr(data, cfg.set)(cfg)
    train, validate, modifier = get_trainer(cfg)



    # last_val_acc1, last_val_acc5 = validate(dataset.tst_loader, model, softmax_criterion, cfg, writer, epoch)
    # print('Original Model : ',last_val_acc1, last_val_acc5)

    # quit()

    cfg.slimming_factor = 0.8
    cfg.keep_rate = 1.0
    cfg.bias_keep_rate = 1.0
    split_model = net_utils.get_model(cfg)
    split_model = net_utils.move_model_to_gpu(cfg, split_model)
    extract_slim(split_model, model)

    print_mode = 3  # [names,dimension]
    # for (src_n,src_m) in zip(split_model.named_modules(),model.named_modules()):

    if print_mode in [1,2]:
        for (src_n,src_m) in split_model.named_modules():
            # print(src_n,dst_n)
            if hasattr(src_m, "weight") and src_m.weight is not None:
                if src_m.__class__ == conv_type.SubnetCoarseConv:
                    if print_mode == 0:
                        print(src_n,end='')
                    elif print_mode == 1:
                        print(*[src_m.weight.shape[i] for i in [0, 2, 3, 1]], sep=' $\\times$ ', end='')
                    else:
                        raise NotImplemented('Invalid print_mode {}'.format(print_mode))
                    # print(*[src_m.weight.shape[i] for i in [0,2,3,1]],sep=' $\\times$ ',end='')
                    # print(*[dst_m.weight.shape[i] for i in [0,2,3,1]],sep=' x ')
                    print()

                elif src_m.__class__ == linear_type.SplitLinear:
                    if print_mode == 0:
                        print(src_n,end='')
                    elif print_mode ==1:
                        print(*[src_m.weight.shape[i] for i in [0, 1]], sep=' $\\times$ ', end='')
                    else:
                        raise NotImplemented('Invalid print_mode {}'.format(print_mode))
                    print()
                    # print(src_n, src_m.weight.shape, dst_m.weight.shape)
                elif src_m.__class__ == bn_type.SplitBatchNorm:
                    if print_mode == 0:
                        print(src_n,end='')
                    elif print_mode == 1:
                        print(*src_m.weight.shape, sep=' $\\times$ ', end='')
                    else:
                        raise NotImplemented('Invalid print_mode {}'.format(print_mode))
                    print()
                    # print(src_n, src_m.weight.shape, dst_m.weight.shape)
                else:
                    print(src_n)



    dummy_input_tensor = torch.zeros((1, 3, 224, 224)).cuda()
    total_ops, total_params = model_profile.profile(model, dummy_input_tensor)
    print("Dense #Ops: %f GOps" % (total_ops / 1e9))
    print("Dense #Parameters: %f M" % (total_params / 1e6))
    last_val_acc1, last_val_acc5 = validate(dataset.tst_loader, model, softmax_criterion, cfg, writer, epoch)
    print('Dense Model : ', last_val_acc1, last_val_acc5)

    total_ops, total_params = model_profile.profile(split_model, dummy_input_tensor)
    print("Split #Ops: %f GOps" % (total_ops / 1e9))
    print("Split #Parameters: %f M" % (total_params / 1e6))



    last_val_acc1, last_val_acc5 = validate(dataset.tst_loader, split_model, softmax_criterion, cfg, writer, epoch)
    print('Split Model : ',last_val_acc1, last_val_acc5)


if __name__ == '__main__':
    main()