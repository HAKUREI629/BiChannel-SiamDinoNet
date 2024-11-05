import os
import yaml
import numpy as np
import random
import shutil
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from nets.siamese_dino import SiameseDinoResNet, SiameseDinoTFAResNet, SiameseDinoTFAResNetV2
from utils.callbacks import LossHistory
from utils.dataloader_dino_2ch import SiameseDinoDataset, dataset_collate_dino, SiameseDinoDatasetV2, dataset_collate_dinov2
from utils.utils import (download_weights, get_lr_scheduler, load_dataset,
                         set_optimizer_lr, show_config)
from utils.utils_fit_dino import fit_one_epoch

from utils.data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from utils.rnd_resized_crop import RandomResizedCrop_diy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    #----------------------------------------------------#
    #   Whether to use CUDA
    #   if you don't have a GPU, you can set it to False
    #----------------------------------------------------#
    Cuda            = True
    #---------------------------------------------------------------------#
    #   distributed     Used to specify whether to use single-machine multi-GPU distributed operation.
    #                   Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES is used to specify GPUs under Ubuntu.
    #                   On Windows systems, all GPUs are invoked by default in DP mode; DDP is not supported.
    #   DP(default) ：
    #                   distributed = False
    #                   CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP：
    #                   distributed = True
    #                   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use sync_bn; multi-GPU is available in DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed-precision training
    #               Can reduce memory usage by about half; requires PyTorch version 1.7.1 or higher
    #---------------------------------------------------------------------#
    fp16            = False
    #----------------------------------------------------#
    #   Path where the dataset is stored
    #----------------------------------------------------#
    dataset_path    = "/wangyunhao/cds/wangyunhao/Siamese-pytorch/datasetnew/"
    #----------------------------------------------------#
    #   Features used
    #----------------------------------------------------#
    features        = "melfbank_10s_50ms_100_1khz_50overlap_wav"
    #----------------------------------------------------#
    #   Size of the input image, default is 399,300 for melfbank
    #----------------------------------------------------#
    input_shape     = [399,300]
    #----------------------------------------------------#
    #   Set to True when training on your own dataset (default: True)
    #----------------------------------------------------#
    train_own_data  = True
    #-------------------------------#
    #   Whether to use pre-trained weights of ResNet backbone
    #-------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   If training is interrupted, you can set model_path to the weights file in the logs folder to reload partially trained weights.
    #   Modify the parameters for the freeze or unfreeze stage below to ensure continuity in training epochs.
    #   
    #   When model_path = '', no weights are loaded for the entire model.
    #
    #   Here, the entire model's weights are loaded in train.py, so pretraining does not affect weight loading at this stage.
    #   To start training with pre-trained weights from the backbone, set model_path = '' and pretrain = True; only the backbone is loaded in this case.
    #   To train the model from scratch, set model_path = '' and pretrain = False, starting from scratch.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ""
    #------------------------------------------------------#
    #   Training parameters
    #   Init_Epoch      Start Epoch
    #   Epoch           Total Epochs
    #   batch_size      
    #------------------------------------------------------#
    Init_Epoch          = 0
    Epoch               = 100
    batch_size          = 64
    
    #------------------------------------------------------------------#
    #   Training parameters：learning rate, optimizer, learning rate decay strategy
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         maxium learning rate
    #   Min_lr          minium learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  adam or sgd
    #   momentum        
    #   weight_decay    Weight decay. 
    #                   Note that Adam may cause errors with weight decay, so it is recommended to set this to 0 when using Adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   learning rate decay strategy, step or cos
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     Save weights every specified number of epochs
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        Folder for saving weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     Used to set whether to use multithreading for data loading; 1 means multithreading is disabled
    #------------------------------------------------------------------#
    num_workers         = 4
    #------------------------------------------------------------------#
    pretrain_flag       = False
    #------------------------------------------------------------------#
    #   pretrain_flag   Whether to use AudioSet pretraining
    #------------------------------------------------------------------#

    #------------------------------------------------------------------#
    #   Adjust training parameters if using AudioSet pretraining 
    #------------------------------------------------------------------#
    if pretrain_flag:
        optimizer_type      = "adam"
        momentum            = 0.9
        weight_decay        = 0
        Init_lr             = 3e-4
        Min_lr              = Init_lr * 0.1

    #------------------------------------------------------#
    #   GPU Setting
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    model = SiameseDinoTFAResNetV2(input_shape, pretrained)
    # if pretrain_flag:
    #     model = SiameseResNetAudioSetPreTrain(input_shape, pretrained)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    

    time_str    = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    save_dir    = os.path.join(save_dir, features + "_dinotfav2_save_" + str(time_str))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    shutil.copy('./nets/siamese_dino.py', os.path.join(save_dir, 'siamese_dino.py'))
    shutil.copy('./data_aug.yml', os.path.join(save_dir, 'data_aug.yml'))
    shutil.copy('./train_dinotfa.py', os.path.join(save_dir, 'train_dinotfa.py'))
    shutil.copy('./utils/dataloader_dino_2ch.py', os.path.join(save_dir, 'dataloader_dino_2ch.py'))
    with open(os.path.join(save_dir, 'param.txt'), 'w') as file:
        file.write('pretrain flag: ' + str(pretrain_flag) + '\n')
        file.write('feature: ' + features + '\n')
        file.write('feature shape: ' + str(input_shape[0]) + ' ' + str(input_shape[1]) + '\n')
        file.write('lr: ' + str(Init_lr) + '\n')
        file.write('optim: ' + optimizer_type + '\n')

    loss = nn.BCEWithLogitsLoss()
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            #model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    params_yaml = args.config
    config = yaml.load(open(params_yaml), yaml.FullLoader)
    args = config
    if not pretrain_flag:
        train_ratio = 0.9
        train_lines, train_labels, val_lines, val_labels = load_dataset(dataset_path, train_own_data, train_ratio, features)
        #_, _, val_lines_shi, val_labels_shi              = load_dataset(dataset_path, train_own_data, train_ratio, features + '_shipsear')
        num_train   = len(train_lines)
        num_val     = len(val_lines)

        train_dataset   = SiameseDinoDatasetV2(input_shape, np.array([50, 300]), 2, train_lines, train_labels, args, '/wangyunhao/cds/wangyunhao/Siamese-pytorch/datasetnew/' + features + '/valid/', True)
        val_dataset     = SiameseDinoDatasetV2(input_shape, np.array([50, 300]), 2, val_lines, val_labels, args, '/wangyunhao/cds/wangyunhao/Siamese-pytorch/datasetnew/' + features + '/valid/', False, na='test')
        # shipsear valid set
        # val_dataset_shi = SiameseDinoDatasetV2(input_shape, np.array([50, 300]), 2, val_lines_shi, val_labels_shi, args, '/wangyunhao/cds/wangyunhao/Siamese-pytorch/datasetnew/' + features + '_shipsear' + '/valid/',False, na='test')
        num_train   = train_dataset.len
        num_val     = val_dataset.len
    else:
        train_path      = ''
        val_path        = ''
        # train_dataset   = SiameseAudioSetDataset(input_shape, train_path, args, True)
        # val_dataset     = SiameseAudioSetDataset(input_shape, val_path , args, False, na='test')

        num_train   = 0

    if local_rank == 0:
        show_config(
            model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 3e4 if optimizer_type == "sgd" else 1e4
        total_step  = num_train // batch_size * Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // batch_size) + 1
            print("\n\033[1;33;44m[Warning] Using %s optimizer, it is recommended to set the total training steps to above %d. \033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total amount of training data for this run is %d, with a batch size of %d, training for %d epochs, resulting in a total of %d training steps.\033[0m"%(num_train, batch_size, Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training steps are %d, which is less than the recommended total steps of %d, it is advised to set the total epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    if True:
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if (epoch_step == 0 or epoch_step_val == 0) and not pretrain_flag:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        if not pretrain_flag:
            gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate_dinov2, sampler=train_sampler)
            gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate_dinov2, sampler=val_sampler)
            # gen_val_shi     = DataLoader(val_dataset_shi, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
            #                        drop_last=True, collate_fn=dataset_collate_dinov2, sampler=val_sampler)
        # else:
        #     gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        #                             drop_last=True, collate_fn=dataset_collate_audioset, sampler=train_sampler)
        #     gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        #                             drop_last=True, collate_fn=dataset_collate_audioset, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, None, Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
