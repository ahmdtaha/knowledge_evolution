import math
import torch
import numpy as np
import pandas as pd
import os.path as osp
from utils import path_utils
from configs.base_config import Config
from data.custom_dataset import CustomDataset


class Flower102Pytorch:


    def __init__(self, cfg):
        #imgs_path, lbls, is_training

        db_path = path_utils.get_datasets_dir(cfg.set)
        self.img_path = db_path + '/jpg/'

        csv_file = '/lists/trn.csv'
        trn_data_df = pd.read_csv(db_path + csv_file)

        lbls = trn_data_df['label']
        lbl2idx = np.sort(np.unique(lbls))
        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())



        self.train_loader = self.create_loader(csv_file, cfg,is_training=True)

        csv_file = '/lists/tst.csv'
        self.tst_loader = self.create_loader(csv_file,cfg,is_training=False)

        csv_file = '/lists/val.csv'
        self.val_loader = self.create_loader(csv_file,cfg,is_training=False)


    def create_loader(self,imgs_lst,cfg,is_training):
        db_path = path_utils.get_datasets_dir(cfg.set)
        if osp.exists(db_path + imgs_lst):
            data_df = pd.read_csv(db_path + imgs_lst)
            imgs, lbls = self.imgs_and_lbls(data_df)
            epoch_size = len(imgs)
            loader = torch.utils.data.DataLoader(CustomDataset(imgs, lbls, is_training=is_training),
                                                          batch_size=cfg.batch_size, shuffle=is_training,
                                                          num_workers=cfg.num_threads)

            loader.num_batches = math.ceil(epoch_size / cfg.batch_size)
            loader.num_files = epoch_size
        else:
            loader = None

        return  loader

    def imgs_and_lbls(self,data_df):
            """
            Load images' paths and int32 labels
            :param repeat: This is similar to TF.data.Dataset repeat. I use TF dataset repeat and no longer user this params.
            So its default is False

            :return: a list of images' paths and their corresponding int32 labels
            """

            imgs = data_df
            ## Faster way to read data
            images = imgs['file_name'].tolist()
            lbls = imgs['label'].tolist()
            for img_idx in range(imgs.shape[0]):
                images[img_idx] = self.img_path + images[img_idx]
                lbls[img_idx] = self.lbl2idx_dict[lbls[img_idx]]


            return images, lbls

def main():
    arg_dataset = 'Flower102Pytorch'
    arg_epochs = str(10)
    arg_evolve_mode = 'rand'
    # arg_reset_scores = False
    arg_reset_hypothesis = False
    arg_pretrained = False  # imagnet or False
    arg_enable_cs_kd = False
    arg_enable_label_smoothing = True
    arg_arch = 'Split_ResNet34'  # Split_ResNet18,Split_ResNet34,Split_ResNet50,split_googlenet,split_densenet169,split_vgg11_bn,split_densenet121
    arg_split_top = '0.5'
    arg_bias_split_top = arg_split_top
    arg_num_generations = '10'
    if arg_arch == 'split_densenet169':
        arg_split_mode = 'kels'
    else:
        arg_split_mode = 'kels'

    exp_name_suffix = 'single_gpu_cls_study_fix_wels_wo_cache'
    # exp_name_suffix = 'redo_debug'
    arg_exp_name = 'SPLT_CLS_{}_{}_cskd{}_smth{}_imagenet{}_k{}_G{}_e{}_ev{}_hReset{}_sm{}_{}/'.format(arg_dataset,
                                                                                                       arg_arch,
                                                                                                       arg_enable_cs_kd,
                                                                                                       arg_enable_label_smoothing,
                                                                                                       arg_pretrained,
                                                                                                       arg_split_top,
                                                                                                       arg_num_generations,
                                                                                                       arg_epochs,
                                                                                                       arg_evolve_mode,
                                                                                                       arg_reset_hypothesis,
                                                                                                       arg_split_mode,
                                                                                                       exp_name_suffix)

    if arg_arch in ['split_alexnet', 'split_vgg11', 'split_vgg11_bn']:
        arg_weight_decay = '5e-4'
        arg_init = 'kaiming_normal'
    else:
        arg_weight_decay = '1e-4'
        arg_init = 'kaiming_normal'
    # arg_exp_name = 'CUB200_ResNet18_cskdFalse_smthTrue_cache_enabled_cosine_lr/'

    argv = [
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

        '--conv_type', 'SplitConv',  # 'SubnetConv','StrictSubnetConv
        '--bn_type', 'SplitBatchNorm',
        '--linear_type', 'SplitLinear',

        '--split_rate', arg_split_top,
        '--bias_split_rate', arg_bias_split_top,
        '--init', arg_init,  # xavier_normal, kaiming_normal
        '--mode', 'fan_in',
        '--nonlinearity', 'relu',
        '--num_generations', arg_num_generations,
        '--split_mode', arg_split_mode,
    ]

    if arg_enable_cs_kd:
        argv.extend(['--cs_kd'])

    if arg_enable_label_smoothing:
        argv.extend(['--label_smoothing', '0.1'])

    if arg_pretrained:
        argv.extend(['--pretrained', 'imagenet',
                     '--num_generations', '1',
                     '--lr', '0.00256'])
    else:
        if arg_arch in ['split_alexnet', 'split_vgg11', 'split_vgg11_bn']:
            argv.extend(['--lr', '0.0256'])
        else:
            argv.extend(['--lr', '0.256'])

    # if arg_reset_scores:
    #     argv.extend(['--reset_scores'])

    if arg_reset_hypothesis:
        argv.extend(['--reset_hypothesis'])

    cfg = Config().parse(argv)
    cfg.num_threads = 0
    import data
    dataset = getattr(data, cfg.set)(cfg)
    trn_cnt = 0
    val_cnt = 0
    tst_cnt = 0
    for batch in dataset.train_loader:
        images, target = batch[0], batch[1].long().squeeze()
        trn_cnt += target.shape[0]

    for batch in dataset.val_loader:
        images, target = batch[0], batch[1].long().squeeze()
        val_cnt += target.shape[0]

    for batch in dataset.tst_loader:
        images, target = batch[0], batch[1].long().squeeze()
        tst_cnt += target.shape[0]

    print(trn_cnt,val_cnt,tst_cnt)

if __name__ == '__main__':
    main()

