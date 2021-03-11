import os
import sys
import data
import torch
import getpass
import KE_model
import importlib
import os.path as osp
import torch.nn as nn
from utils import os_utils
from utils import net_utils
from utils import csv_utils
from layers import conv_type
from utils import path_utils
from utils import model_profile
from configs.base_config import Config

def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate

def train_dense(cfg, generation):

    model = net_utils.get_model(cfg)

    if cfg.pretrained and cfg.pretrained != 'imagenet':
        net_utils.load_pretrained(cfg.pretrained,cfg.gpu, model,cfg)
        model = net_utils.move_model_to_gpu(cfg, model)
        net_utils.split_reinitialize(cfg,model,reset_hypothesis=cfg.reset_hypothesis)
    else:
        model = net_utils.move_model_to_gpu(cfg, model)

    cfg.trainer = 'default_cls'
    # cfg.split_rate = 1.0
    # cfg.bias_split_rate = 1.0
    cfg.pretrained = None
    ckpt_path = KE_model.ke_cls_train(cfg, model,generation)

    return ckpt_path


def eval_slim(cfg, generation):
    original_num_epos = cfg.epochs
    # cfg.epochs = 0
    softmax_criterion = nn.CrossEntropyLoss().cuda()
    epoch = 1
    writer = None
    model = net_utils.get_model(cfg)
    net_utils.load_pretrained(cfg.pretrained, cfg.gpu, model,cfg)
    # if cfg.reset_mask:
    #     net_utils.reset_mask(cfg, model)
    model = net_utils.move_model_to_gpu(cfg, model)

    save_filter_stats = (cfg.arch in ['split_alexnet','split_vgg11_bn'])
    if save_filter_stats:
        for n, m in model.named_modules():
            if hasattr(m, "weight") and m.weight is not None:
                if hasattr(m, "mask"):
                    layer_mask = m.mask
                    if m.__class__ == conv_type.SplitConv:
                        # filter_state = [''.join(map(str, ((score_mask == True).type(torch.int).squeeze().tolist())))]
                        filter_mag = ['{},{}'.format(
                            float(torch.mean(torch.abs(m.weight[layer_mask.type(torch.bool)]))),
                            float(torch.mean(torch.abs(m.weight[(1-layer_mask).type(torch.bool)]))))
                        ]
                        os_utils.txt_write(osp.join(cfg.exp_dir, n.replace('.', '_') + '_mean_magnitude.txt'), filter_mag, mode='a+')

    dummy_input_tensor = torch.zeros((1, 3, 224, 224)).cuda()
    total_ops, total_params = model_profile.profile(model, dummy_input_tensor)
    cfg.logger.info("Dense #Ops: %f GOps" % (total_ops / 1e9))
    cfg.logger.info("Dense #Parameters: %f M" % (total_params / 1e6))

    original_split_rate = cfg.split_rate
    original_bias_split_rate = cfg.bias_split_rate

    if cfg.split_mode == 'kels':
        cfg.slim_factor = cfg.split_rate
        cfg.split_rate = 1.0
        cfg.bias_split_rate = 1.0
        split_model = net_utils.get_model(cfg)
        split_model = net_utils.move_model_to_gpu(cfg, split_model)

        total_ops, total_params = model_profile.profile(split_model, dummy_input_tensor)
        cfg.logger.info("Split #Ops: %f GOps" % (total_ops / 1e9))
        cfg.logger.info("Split #Parameters: %f M" % (total_params / 1e6))

        net_utils.extract_slim(split_model, model)
        dataset = getattr(data, cfg.set)(cfg)
        train, validate = get_trainer(cfg)
        last_val_acc1, last_val_acc5 = validate(dataset.tst_loader, split_model, softmax_criterion, cfg, writer, epoch)
        cfg.logger.info('Split Model : {} , {}'.format(last_val_acc1, last_val_acc5))
    else:
        last_val_acc1 = 0
        last_val_acc5 = 0

    csv_utils.write_cls_result_to_csv(
        ## Validation
        curr_acc1=0,
        curr_acc5=0,
        best_acc1=0,
        best_acc5=0,

        ## Test
        last_tst_acc1=last_val_acc1,
        last_tst_acc5=last_val_acc5,
        best_tst_acc1=0,
        best_tst_acc5=0,

        ## Train
        best_train_acc1=0,
        best_train_acc5=0,

        split_rate='slim',
        bias_split_rate='slim',

        base_config=cfg.name,
        name=cfg.name,
    )

    cfg.epochs = original_num_epos

    cfg.slim_factor = 1
    cfg.split_rate = original_split_rate
    cfg.bias_split_rate = original_bias_split_rate



def clean_dir(ckpt_dir,num_epochs):
    # print(ckpt_dir)
    if '0000' in str(ckpt_dir): ## Always keep the first model -- Help reproduce results
        return
    rm_path = ckpt_dir / 'model_best.pth'
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'epoch_{}.state'.format(num_epochs - 1)
    if rm_path.exists():
        os.remove(rm_path)

    rm_path = ckpt_dir / 'initial.state'
    if rm_path.exists():
        os.remove(rm_path)

def start_KE(cfg):
    # assert cfg.epochs % 10 == 0 or 'debug' in cfg.name, 'Epoch should be divisible by 10'
    assert cfg.cs_kd == False, 'CS-KD requires a different data loader, not available in this repos'

    ckpt_queue = []

    for gen in range(cfg.num_generations):
        cfg.start_epoch = 0

        # cfg.name = original_name + 'task'
        task_ckpt = train_dense(cfg, gen)
        ckpt_queue.append(task_ckpt)

        # cfg.name = original_name + 'mask'

        cfg.pretrained = task_ckpt / 'epoch_{}.state'.format(cfg.epochs - 1)

        if cfg.num_generations == 1:
            break

        eval_slim(cfg, gen)

        cfg.pretrained = task_ckpt / 'epoch_{}.state'.format(cfg.epochs - 1)

        if len(ckpt_queue) > 4:
            oldest_ckpt = ckpt_queue.pop(0)
            clean_dir(oldest_ckpt, cfg.epochs)

def main(arg_num_threads=16):
    print('Starting with {} threads'.format(arg_num_threads))
    # arg_dataset = 'CUB200'  # Flower102, CUB200,HAM,Dog120,MIT67,Aircraft100,MINI_MIT67,FCAM
    for arg_dataset in ['Flower102Pytorch']:
        arg_epochs = str(200)
        arg_evolve_mode = 'rand'
        arg_reset_hypothesis = False
        arg_enable_cs_kd = False
        arg_enable_label_smoothing = True
        arg_arch = 'Split_ResNet18'  # Split_ResNet18,Split_ResNet34,Split_ResNet50,split_googlenet,split_densenet169,split_vgg11_bn,split_densenet121
        arg_split_top = '0.5'
        arg_bias_split_top = arg_split_top
        arg_num_generations = '5'
        arg_split_mode = 'kels' # wels , kels

        exp_name_suffix = 'single_gpu_test2'
        arg_exp_name = 'SPLT_CLS_{}_{}_cskd{}_smth{}_k{}_G{}_e{}_ev{}_hReset{}_sm{}_{}/'.format(arg_dataset, arg_arch,
                                                                                           arg_enable_cs_kd,arg_enable_label_smoothing,
                                                                                           arg_split_top,arg_num_generations,
                                                                                           arg_epochs,arg_evolve_mode,arg_reset_hypothesis,arg_split_mode,exp_name_suffix)

        if arg_arch in ['split_alexnet','split_vgg11','split_vgg11_bn']:
            arg_weight_decay = '5e-4'
            arg_init = 'kaiming_normal'
        else:
            arg_weight_decay = '1e-4'
            arg_init = 'kaiming_normal'

        argv = [
            '--name', arg_exp_name,
            '--evolve_mode',arg_evolve_mode,
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

            '--conv_type', 'SplitConv',  # 'SubnetConv','StrictSubnetConv
            '--bn_type', 'SplitBatchNorm',
            '--linear_type', 'SplitLinear',

            '--split_rate', arg_split_top,
            '--bias_split_rate', arg_bias_split_top,
            '--init', arg_init, # xavier_normal, kaiming_normal
            '--mode', 'fan_in',
            '--nonlinearity', 'relu',
            '--num_generations', arg_num_generations,
            '--split_mode',arg_split_mode,
        ]

        if arg_enable_cs_kd:
            argv.extend(['--cs_kd'])

        if arg_enable_label_smoothing:
            argv.extend(['--label_smoothing', '0.1'])

        argv.extend(['--lr', '0.256'])

        if arg_reset_hypothesis:
            argv.extend(['--reset_hypothesis'])


        cfg = Config().parse(argv)

        start_KE(cfg)



if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        cfg = Config().parse(None)
        # print(cfg.name)
        start_KE(cfg)