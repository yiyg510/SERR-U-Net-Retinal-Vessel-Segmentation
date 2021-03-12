
"""
Copyright (c) 2020. All rights reserved.
Created by lixiang on 2020/5/10

"""
import os
from experiments.pre_process.standard_loader import DataLoader
# from infers.simple_mnist_infer import SimpleMnistInfer
from template.models.dense_unet import  SegmentionModel
from template.trainers.trainer import SegmentionTrainer
from conf.utils.config_utils import process_config
import numpy as np


def main_train():
    """
    训练模型

    :return:
    """
    print('[INFO] Reading Configs...')

    config = None

    try:
        config = process_config('conf/segmention_config.json')
    except Exception as e:
        print('[Exception] Config Error, %s' % e)
        exit(0)
    # np.random.seed(47)  # 固定随机数

    print('[INFO] Preparing Data...')
    dataloader = DataLoader(config=config)
    dataloader.prepare_dataset()

    train_imgs,train_gt=dataloader.get_train_data()
    val_imgs,val_gt=dataloader.get_val_data()

    print('[INFO] Building Model...')
    model = SegmentionModel(config=config)
    #
    print('[INFO] Training...')
    trainer = SegmentionTrainer(
         model=model.model,
         data=[train_imgs,train_gt,val_imgs,val_gt],
         config=config)
    trainer.train()
    print('[INFO] Finishing...')



if __name__ == '__main__':
    main_train()
    # test_main()
