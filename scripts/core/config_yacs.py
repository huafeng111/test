import os
import ipdb
import datetime
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 8
_C.DATA.DATASET = 'P3M10K'
_C.DATA.TRAIN_SET = 'TRAIN'  # other options: [TRAIN_NORMAL, TRAIN_MOSAIC, TRAIN_ZERO]
_C.DATA.HYBRID_OBFUSCATION = None
_C.DATA.NUM_WORKERS = 8
_C.DATA.RESIZE_SIZE = 512
_C.DATA.CROP_SIZE = [512, 768, 1024]  # RATIO: 1, 1.5, 2

# global data setting | for two stage network only
_C.DATA.GLOBAL = CN()
_C.DATA.GLOBAL.CROP_SIZE = [256, 384, 512]
_C.DATA.GLOBAL.RESIZE_SIZE = 256

# local data setting | for two stage network only
_C.DATA.LOCAL = CN()
_C.DATA.LOCAL.GLOBAL_SIZE = None
_C.DATA.LOCAL.PATCH_RESIZE_SIZE = 64
_C.DATA.LOCAL.PATCH_SIZE = 64
_C.DATA.LOCAL.PATCH_NUMBER = 64

# Settings for cut and paste at data level
_C.DATA.CUT_AND_PASTE = CN()
_C.DATA.CUT_AND_PASTE.TYPE = 'NONE'  # CHOICES: ['NONE', 'VANILLA', 'AUG', 'RESIZE2FILL', 'GRID_SAMPLE']
_C.DATA.CUT_AND_PASTE.SOURCE_DATASET = ''  # CHOICES: ['SELF', 'P3M10K', 'CELEBAMASK_HQ']
_C.DATA.CUT_AND_PASTE.PROB = 0.5
# cut and paste: aug
_C.DATA.CUT_AND_PASTE.AUG = CN()
_C.DATA.CUT_AND_PASTE.AUG.DEGREE = 30
_C.DATA.CUT_AND_PASTE.AUG.SCALE = [0.8,1.2]
_C.DATA.CUT_AND_PASTE.AUG.SHEAR = None
_C.DATA.CUT_AND_PASTE.AUG.FLIP = [0.5,0]
_C.DATA.CUT_AND_PASTE.AUG.RANDOM_PASTE = False
# cut and paste: grid_sample
_C.DATA.CUT_AND_PASTE.GRID_SAMPLE = CN()
_C.DATA.CUT_AND_PASTE.GRID_SAMPLE.SELECT_RANGE = [10, 2, 10, 2]
_C.DATA.CUT_AND_PASTE.GRID_SAMPLE.DOWN_SCALE = 4

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = ''
_C.MODEL.PRETRAINED = True

# Settings for swin transformer
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.USE_CHECKPOINT = False
_C.MODEL.SWIN.DROP_RATE = 0.0
_C.MODEL.SWIN.DROP_PATH_RATE = 0.2

# Settings for cut and paste at feature level
_C.MODEL.CUT_AND_PASTE = CN()
_C.MODEL.CUT_AND_PASTE.TYPE = 'NONE'  # CHOICES: ['NONE', 'CP', 'SHUFFLE'], cp shuffle 只能二选一
_C.MODEL.CUT_AND_PASTE.PROB = 0.5
_C.MODEL.CUT_AND_PASTE.START_EPOCH = 50
_C.MODEL.CUT_AND_PASTE.LAYER = []  # can be a list of layers
_C.MODEL.CUT_AND_PASTE.DETACH = True

# Settings for cut and paste cp
_C.MODEL.CUT_AND_PASTE.CP = CN()
_C.MODEL.CUT_AND_PASTE.CP.TYPE = 'VANILLA'  # CHOICES: ['VANILLA', 'MULTISCALE'], control the fea cut and paste data class type
_C.MODEL.CUT_AND_PASTE.CP.MODEL = 'SELF'  # CHOICES: ['COPY_EVERY_ITER', 'COPY_EVERY_EPOCH', 'SELF'], control the model used to extract source fea
_C.MODEL.CUT_AND_PASTE.CP.SOURCE_DATASET = 'SELF'  # CHOIES: ['SELF', 'P3M10K', 'CELEBAMASK_HQ'], control the data used to extract source fea
_C.MODEL.CUT_AND_PASTE.CP.SOURCE_BATCH_SIZE = 1
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ = CN()
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ.DEFREE = None  # degree ranges, [-degree, +degree]
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ.FLIP = None  # [prob horizon, prob vertical]
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ.CROP_SIZE = None
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ.RESIZE_SIZE = None
_C.MODEL.CUT_AND_PASTE.CP.CELEBAMASK_HQ.SCALE = [0.3, 0.7]  # scale < 1.0

# Settings for cut and paste shuffle
_C.MODEL.CUT_AND_PASTE.SHUFFLE = CN()
_C.MODEL.CUT_AND_PASTE.SHUFFLE.TYPE = ''  # CHOICES: ['NONE', 'FG2FACE']
_C.MODEL.CUT_AND_PASTE.SHUFFLE.KERNEL_SIZE = 1

# TODO
# .SHUFFLE.TYPE: FG2FACE

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.EPOCHS = 150
_C.TRAIN.WARMUP_EPOCHS = 0
_C.TRAIN.LR_DECAY = False
_C.TRAIN.LR = 0.00001
_C.TRAIN.CLIP_GRAD = False
_C.TRAIN.RESUME_CKPT = None

# Settings for optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'ADAM'

# -----------------------------------------------------------------------------
# Test settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.DATASET = 'VAL500P'
_C.TEST.TEST_METHOD = 'HYBRID'
_C.TEST.CKPT_NAME = 'best_SAD_VAL500P'
_C.TEST.FAST_TEST = False
_C.TEST.TEST_PRIVACY = False
_C.TEST.SAVE_RESULT = False
_C.TEST.LOCAL_PATCH_NUM = 1024  # for test only

# -----------------------------------------------------------------------------
# Other settings, e.g. logging, wandb, dist, tag, test freq, save ckpt
# -----------------------------------------------------------------------------
_C.TAG = 'debug'
_C.ENABLE_WANDB = False
_C.TEST_FREQ = 1
_C.AUTO_RESUME = False
_C.SEED = 10007
_C.DIST = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    # merge config from other config
    if getattr(args, 'existing_cfg', None):  # for test only
        if hasattr(args.existing_cfg.MODEL, 'CUT_AND_PASTE') and hasattr(args.existing_cfg.MODEL.CUT_AND_PASTE, 'LAYER'):
            if type(args.existing_cfg.MODEL.CUT_AND_PASTE.LAYER) != list:
                args.existing_cfg.defrost()
                args.existing_cfg.MODEL.CUT_AND_PASTE.LAYER = [args.existing_cfg.MODEL.CUT_AND_PASTE.LAYER]
                args.existing_cfg.freeze()
        config.merge_from_other_cfg(args.existing_cfg)
        assert args.tag == config.TAG

    # merge config from file
    if getattr(args, 'cfg', None):
        _update_config_from_file(config, args.cfg)
    
    config.defrost()
    if getattr(args, 'opts', None):
        config.merge_from_list(args.opts)
    
    # merge from specific arguments
    if getattr(args, 'arch', None):
        config.MODEL.TYPE = args.arch
    
    if getattr(args, 'train_from_scratch', None):
        config.MODEL.PRETRAINED = False
    
    if getattr(args, 'tag', None):
        config.TAG = args.tag

    if getattr(args, 'nEpochs', None):
        config.TRAIN.EPOCHS = args.nEpochs

    if getattr(args, 'warmup_nEpochs', None):
        config.TRAIN.WARMUP_EPOCHS = args.warmup_nEpochs

    if getattr(args, 'batchSize', None):
        config.DATA.BATCH_SIZE = args.batchSize

    if getattr(args, 'lr', None):
        config.TRAIN.LR = args.lr
    
    if getattr(args, 'lr_decay', None):
        config.TRAIN.LR_DECAY = args.lr_decay

    if getattr(args, 'clip_grad', None):
        config.TRAIN.CLIP_GRAD = args.clip_grad

    if getattr(args, 'threads', None):
        config.DATA.NUM_WORKERS = args.threads

    if getattr(args, 'test_freq', None):
        config.TEST_FREQ = args.test_freq
    
    if getattr(args, 'enable_wandb', None):
        config.ENABLE_WANDB = args.enable_wandb
    
    if getattr(args, 'auto_resume', None):
        config.AUTO_RESUME = args.auto_resume
    
    if getattr(args, 'source_batch_size', None):
        config.MODEL.CUT_AND_PASTE.CP.SOURCE_BATCH_SIZE = args.source_batch_size
    
    if getattr(args, 'dataset', None):
        config.DATA.DATASET = args.dataset
    
    if getattr(args, 'train_set', None):
        config.DATA.TRAIN_SET = args.train_set
    
    if getattr(args, 'seed', None):
        config.SEED = args.seed
    
    if getattr(args, 'test_dataset', None):
        config.TEST.DATASET = args.test_dataset
    
    if getattr(args, 'test_ckpt', None):
        config.TEST.CKPT_NAME = args.test_ckpt

    if getattr(args, 'test_method', None):
        config.TEST.TEST_METHOD = args.test_method

    if getattr(args, 'fast_test', None):
        config.TEST.FAST_TEST = args.fast_test
    
    if getattr(args, 'test_privacy', None):
        config.TEST.TEST_PRIVACY = args.test_privacy
    
    if getattr(args, 'save_result', None):
        config.TEST.SAVE_RESULT = args.save_result
    
    if getattr(args, 'local_rank', None):
        config.LOCAL_RANK = args.local_rank

    config.freeze()

def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config