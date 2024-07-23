"""
Rethinking Portrait Matting with Privacy Preserving
Inferernce file.

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""
import os
import ipdb
import random
import argparse
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import datetime
import wandb
import shutil
from tqdm import tqdm
from yacs.config import CfgNode as CN

from config import *
from util import *
from evaluate import *
from test import test_p3m10k
from data import MattingDataset, MattingTransform
from network.build_model import build_model
from config_yacs import get_config
from logger import create_logger

######### Parsing arguments ######### 
def get_args():
    parser = argparse.ArgumentParser(description='Arguments for the training purpose.')
    parser.add_argument('--cfg', type=str, help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--arch', type=str, help="backbone architecture of the model")
    parser.add_argument('--train_from_scratch', action='store_true')
    parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for, 500 for ORI-Track and 100 for COMP-Track')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning Rate. Default=0.00001')
    parser.add_argument('--warmup_nEpochs', type=int, default=0, help='epochs for warming up')
    parser.add_argument('--lr_decay', action='store_true', default=None, help='whehter to use lr decay')
    parser.add_argument('--clip_grad', action='store_true', default=None, help='whether to clip gradient')
    parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--tag', type=str, default='debug')
    parser.add_argument('--enable_wandb', action='store_true', default=None)
    parser.add_argument('--test_freq', type=int)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--test_method', default=None)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_set', type=str)
    args, _ = parser.parse_known_args()

    if args.tag.lower().startswith('debug'):
        args.enable_wandb = False
    
    if args.auto_resume:
        # load existing cfg
        args_path = os.path.join(CKPT_SAVE_FOLDER, args.tag, "args.npy")
        if os.path.exists(args_path):
            args.existing_cfg = np.load(os.path.join(CKPT_SAVE_FOLDER, args.tag, 'args.npy'), allow_pickle=True).item()
    
    config = get_config(args)
    if args.auto_resume:
        set_seed(config.SEED)

    print(config)
    print(args)
    return args, config

def set_seed(seed=None):
    if seed is None:
        seed = torch.seed()
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    return seed

def load_dataset(config):
    train_transform = MattingTransform(crop_size=config.DATA.CROP_SIZE, resize_size=config.DATA.RESIZE_SIZE)
    train_set = MattingDataset(config, train_transform)
    collate_fn = None
    train_loader = DataLoader(dataset=train_set, num_workers=config.DATA.NUM_WORKERS, batch_size=config.DATA.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    return train_loader

def build_lr_scheduler(optimizer, total_epochs):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    return lr_scheduler

def warmup_lr(initial_lr, cur_iter, total_iter):
    return cur_iter/total_iter*initial_lr

def update_lr(cur_lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train(config, model, optimizer, train_loader, epoch, lr_scheduler, clip_grad_args):
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    torch.cuda.empty_cache()
    
    print("===============================")
    print("EPOCH: {}/{}".format(epoch, config.TRAIN.EPOCHS))
    for iteration, batch in tqdm(enumerate(train_loader, 1)):
        ### update lr
        if config.TRAIN.WARMUP_EPOCHS > 0 and epoch <= config.TRAIN.TRAIN.WARMUP_EPOCHS:
            cur_lr = warmup_lr(config.TRAIN.LR, len(train_loader)*(epoch-1)+iteration, config.TRAIN.WARMUP_EPOCHS*len(train_loader))
            update_lr(cur_lr, optimizer)
        elif config.TRAIN.LR_DECAY is True and epoch > config.TRAIN.WARMUP_EPOCHS:
            lr_scheduler.step()
            cur_lr = lr_scheduler.get_lr()[0]
        else:
            cur_lr = optimizer.param_groups[0]['lr']
        
        ### get data for general model
        batch_new = []
        for item in batch:
            if type(item) == torch.Tensor:
                item = Variable(item).cuda()
            batch_new.append(item)
        [ori, mask, fg, bg, trimap] = batch_new[:5]
        optimizer.zero_grad()

        ### get model prediction
        if config.MODEL.CUT_AND_PASTE.TYPE.upper() == 'NONE':
            out_list = model(ori)
        else:
            raise NotImplementedError()
        
        ### cal loss
        predict_global, predict_local, predict_fusion, predict_global_side2, predict_global_side1, predict_global_side0 = out_list
        predict_fusion = predict_fusion.cuda()
        loss_global =get_crossentropy_loss(trimap, predict_global)
        loss_global_side2 = get_crossentropy_loss(trimap, predict_global_side2)
        loss_global_side1 = get_crossentropy_loss(trimap, predict_global_side1)
        loss_global_side0 = get_crossentropy_loss(trimap, predict_global_side0)
        loss_global = loss_global_side2+loss_global_side1+loss_global_side0+3*loss_global	
        loss_local = get_alpha_loss(predict_local, mask, trimap) + get_laplacian_loss(predict_local, mask, trimap)
        loss_fusion_alpha = get_alpha_loss_whole_img(predict_fusion, mask) + get_laplacian_loss_whole_img(predict_fusion, mask)
        loss_fusion_comp = get_composition_loss_whole_img(ori, mask, fg, bg, predict_fusion)
        loss = loss_global/6+loss_local*2+loss_fusion_alpha*2+loss_fusion_alpha+loss_fusion_comp
        loss.backward()
        
        ### optimize and clip gradient
        if config.TRAIN.CLIP_GRAD is True:
            if clip_grad_args.moving_max_grad == 0:
                clip_grad_args.moving_max_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e+6).cpu().item()
                clip_grad_args.max_grad = clip_grad_args.moving_max_grad
            else:
                clip_grad_args.max_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 2 * clip_grad_args.moving_max_grad).cpu().item()
                clip_grad_args.moving_max_grad = clip_grad_args.moving_max_grad * clip_grad_args.moving_grad_moment + clip_grad_args.max_grad * (
                            1 - clip_grad_args.moving_grad_moment)
        optimizer.step()        
        
        if config.ENABLE_WANDB:
            loss_dict = {
                'train/loss': loss.item(),
                'train/loss_global': loss_global.item(),
                'train/loss_local': loss_local.item(),
                'train/loss_fusion_alpha': loss_fusion_alpha.item(),
                'train/loss_fusion_comp': loss_fusion_comp.item(),
                'train/lr': cur_lr,
            }
            wandb.log(loss_dict, step=(epoch-1)*len(train_loader)+iteration, commit=False)
        
        if config.TAG.startswith("debug") and iteration > 20:
            break

def save_latest_checkpoint(config, model, epoch, **kwargs):
    model_save_dir = os.path.join(CKPT_SAVE_FOLDER, config.TAG)
    create_folder_if_not_exists(model_save_dir)
    model_out_path = os.path.join(model_save_dir, 'ckpt_latest.pth')
    model_dict = {'state_dict':model.state_dict(), 'epoch':epoch}
    model_dict.update(kwargs)
    torch.save(model_dict, model_out_path)

def save_checkpoint(config, model, epoch, prefix, **kwargs):
    model_save_dir = os.path.join(CKPT_SAVE_FOLDER, config.TAG)
    create_folder_if_not_exists(model_save_dir)
    model_out_path = os.path.join(model_save_dir, prefix+'_ckpt.pth')
    model_dict = {'state_dict':model.state_dict(), 'epoch':epoch}
    model_dict.update(kwargs)
    torch.save(model_dict, model_out_path)

def save_all_ckpts(config, epoch):
    model_save_dir = os.path.join(CKPT_SAVE_FOLDER, config.TAG, str(epoch))
    create_folder_if_not_exists(model_save_dir)
    source_dir = os.path.join(CKPT_SAVE_FOLDER, config.TAG)
    file_list = os.listdir(source_dir)
    for name in file_list:
        if name.endswith('.pth'):
            shutil.copyfile(os.path.join(source_dir, name), os.path.join(model_save_dir, name))

def save_args(config):
    save_dir = os.path.join(CKPT_SAVE_FOLDER, config.TAG)
    create_folder_if_not_exists(save_dir)
    np.save(os.path.join(save_dir, 'args.npy'), config)

def main():
    _, config = get_args()
    save_args(config)

    now = datetime.datetime.now()
    str_time = now.strftime("%Y-%m-%d-%H:%M")
    logger = create_logger(os.path.join(TRAIN_LOGS_FOLDER, config.TAG+'_{}.log'.format(str_time)))
    
    if not torch.cuda.is_available():
        raise Exception("No GPU and cuda available, please try again")
    gpuNums = torch.cuda.device_count()
    logger.info(f'Running with GPUs and the number of GPUs: {gpuNums}')
    
    # Test configs
    config.defrost()
    config.TEST.DATASET = 'P3M_500_P'
    config.TEST.TEST_METHOD = config.TEST.TEST_METHOD
    config.TEST.CKPT_NAME = 'latest_epoch'
    config.TEST.FAST_TEST = True
    config.TEST.TEST_PRIVACY = True
    config.TEST.SAVE_RESULT = False
    config.freeze()
    
    # config wandb
    if config.ENABLE_WANDB:
        project_name = 'p3m_journal'
        WANDB_API_KEY = get_wandb_key(WANDB_KEY_FILE)  # setup your wandb account
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        os.environ["WANDB_DIR"] = WANDB_LOGS_FOLDER

        if config.AUTO_RESUME and os.path.exists(os.path.join(CKPT_SAVE_FOLDER, config.TAG, 'wandb_run_id.txt')):
            with open(os.path.join(CKPT_SAVE_FOLDER, config.TAG, 'wandb_run_id.txt'), 'r') as f:
                run_id = f.readline().strip('\n')
            wandb.init(id=run_id, project=project_name, resume='must')
        else:
            run_id = wandb.util.generate_id()
            wandb.init(project=project_name, entity='xymsh', id=run_id, resume='allow')
            wandb.config.update(get_wandb_config(config))
            wandb.run.name = '{}_{}'.format(config.TAG, str_time)
            with open(os.path.join(CKPT_SAVE_FOLDER, config.TAG, 'wandb_run_id.txt'), 'w') as f:
                f.write(run_id)
        logger.info('===> Enabled wandb run {} at {}'.format(run_id, WANDB_LOGS_FOLDER))

    # data loader
    logger.info('===> Load data')
    train_loader = load_dataset(config)

    # build model
    logger.info('===> Build the model {}'.format(config.MODEL.TYPE))
    model = build_model(config.MODEL.TYPE).cuda()
    start_epoch = config.TRAIN.START_EPOCH

    # build optimizer
    logger.info('===> Initialize optimizer {} and lr scheduler {}'.format(config.TRAIN.OPTIMIZER.TYPE, config.TRAIN.LR_DECAY))
    if config.TRAIN.OPTIMIZER.TYPE.upper() == 'ADAM':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.TRAIN.LR)
    else:
        raise NotImplementedError
    lr_scheduler = build_lr_scheduler(optimizer, config.TRAIN.EPOCHS-config.TRAIN.WARMUP_EPOCHS) if config.TRAIN.LR_DECAY is True else None
    # init clip grad params
    clip_grad_args = CN()
    clip_grad_args.moving_max_grad = 0.0
    clip_grad_args.moving_grad_moment = 0.999
    clip_grad_args.max_grad = 0.0

    # train parameters
    best_result = {}  # key format: "[metric]_[dataset]" and "[metric]_[dataset]_epoch"
    errors = {}

    # auto resume
    if config.AUTO_RESUME:
        try:
            # load latest ckpt
            ckpt_path = os.path.join(CKPT_SAVE_FOLDER, config.TAG, 'ckpt_latest.pth')
            ckpt_dict = torch.load(ckpt_path)
            
            model.load_state_dict(ckpt_dict['state_dict'])
            start_epoch = ckpt_dict['epoch'] + 1
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            if config.TRAIN.LR_DECAY:
                raise NotImplementedError
            
            clip_grad_args = ckpt_dict['clip_grad_args']
            best_result = ckpt_dict['best_result']
            logger.info('===> Auto resume succeeded')
            del ckpt_dict
        except:
            pass
    
    # start training
    logger.info('===> Start Training')
    for epoch in range(start_epoch, config.TRAIN.EPOCHS + 1):
        logger.info("TRAIN Epoch: {}/{}, Warmup: {}".format(epoch, config.TRAIN.EPOCHS, epoch<=config.TRAIN.WARMUP_EPOCHS))
        train(config, model, optimizer, train_loader, epoch, lr_scheduler, clip_grad_args)

        if (config.TEST_FREQ > 0 and epoch % config.TEST_FREQ == 0) or (epoch == config.TRAIN.EPOCHS):
            logger.info("TEST Epoch: {}/{}".format(epoch, config.TRAIN.EPOCHS))

            for test_dataset_choice in ["P3M_500_P", "P3M_500_NP"]:
                config.defrost()
                config.TEST.DATASET = test_dataset_choice
                if test_dataset_choice == 'P3M_500_NP':
                    config.TEST.TEST_PRIVACY = False
                else:
                    config.TEST.TEST_PRIVACY = True
                config.freeze()
                errors = test_p3m10k(config, model, logger)

                for m in ['SAD', 'SAD_PRIVACY']:
                    m_dataset = "{}_{}".format(m, test_dataset_choice)
                    if m in errors.keys():
                        if m_dataset not in best_result.keys() or errors[m] <= best_result[m_dataset]:
                            best_result[m_dataset] = errors[m]
                            best_result["{}_epoch".format(m_dataset)] = epoch
                            save_checkpoint(config, model, epoch, "best_{}".format(m_dataset), best_result=best_result)
                
                if config.ENABLE_WANDB:
                    log_errors = {}
                    for k, v in errors.items():
                        log_errors['test_{}/'.format(test_dataset_choice)+k] = v
                    wandb.log(log_errors, step=epoch*len(train_loader), commit=False)
            
            if config.ENABLE_WANDB:
                # record the best result
                wandb.run.summary.update(best_result)
                for k, v in best_result.items():
                    log_errors['best_result_{}/{}'.format(config.TEST.TEST_METHOD, k)] = v
                wandb.log(log_errors, step=epoch*len(train_loader), commit=False)
        
        save_latest_checkpoint(config, model, epoch, optimizer=optimizer.state_dict(), clip_grad_args=clip_grad_args)	
        
        
        if config.AUTO_RESUME:
            break
        if epoch in [50, 100]:
            save_all_ckpts(config, epoch)


if __name__ == "__main__":
    main()