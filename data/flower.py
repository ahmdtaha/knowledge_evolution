import math
import torch
import numpy as np
import pandas as pd
import os.path as osp
from utils import path_utils
from data.custom_dataset import CustomDataset


class Flower102Pytorch:


    def __init__(self, cfg):

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

    