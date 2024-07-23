"""
对test来说，和train是完全分开的。
build model是不一样的。因为cp的原因，train model和test model不一样。
path问题，path是不能和train一起公用的
另外很多args是train里没有的
但是test又需要用到config.MODEL里的参数。

"""
import torch
import torch.nn.functional as F
import ipdb
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torchvision import transforms
import logging
import warnings
import yaml
import torch.distributed as dist

from config import *
from util import *
from evaluate import *
from network.build_model import build_model
from config_yacs import get_config, CN
from logger import create_logger


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True, choices=VALID_TEST_DATA_CHOICE, help="which dataset to test")
    parser.add_argument('--test_ckpt', type=str, default='', required=False, help="path of model to use")
    parser.add_argument('--test_method', type=str, required=False, help="which dataset to test")
    parser.add_argument('--fast_test', action='store_true', default=False, help='skip conn and grad metrics for fast test')
    parser.add_argument('--test_privacy', action='store_true', default=False, help='test on the privacy content')
    parser.add_argument('--save_result', action='store_true')
    args, _ = parser.parse_known_args()
    
    if args.test_dataset == 'VAL500NP':
        args.test_privacy = False
        warnings.warn("NO FACEMASK FOR VAL500NP")
    
    args.existing_cfg = np.load(os.path.join(CKPT_SAVE_FOLDER, args.tag, 'args.npy'), allow_pickle=True).item()
    if type(args.existing_cfg) != CN:
        new_cfg = CN()
        new_cfg.TAG = args.tag
        new_cfg.MODEL = CN()
        new_cfg.MODEL.TYPE = args.existing_cfg.arch
        args.existing_cfg = new_cfg
    
    config = get_config(args)
    return args, config

def inference_once(config, model, scale_img, scale_trimap=None):
    if torch.cuda.device_count() > 0:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
    else:
        tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1)
    input_t = tensor_img
    input_t = input_t/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0)
    pred_global, pred_local, pred_fusion = model(input_t)[:3]
    pred_global = pred_global.data.cpu().numpy()
    pred_global = gen_trimap_from_segmap_e2e(pred_global)
    pred_local = pred_local.data.cpu().numpy()[0,0,:,:]
    pred_fusion = pred_fusion.data.cpu().numpy()[0,0,:,:]
    return pred_global, pred_local, pred_fusion

def inference_img_modnet(config, model, img, *args):
    im_h, im_w, c = img.shape
    
    if config.TEST.TEST_METHOD=='RESIZE512':
        ref_size = 512
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
            
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32

        scale_img = resize(img,(im_rh, im_rw))

        scale_img = scale_img*255.0
    
    elif config.TEST.TEST_METHOD=='ORIGIN':
        im_rh = min(MAX_SIZE_H, im_h - im_h % 32)
        im_rw = min(MAX_SIZE_W, im_w - im_w % 32)

        scale_img = resize(img, (im_rh, im_rw))
        scale_img = scale_img * 255.0
    else:
        raise NotImplementedError

    tensor_img = torch.from_numpy(scale_img.astype(np.float32)[:, :, :]).permute(2, 0, 1).cuda()
    tensor_img = (tensor_img/255.0).unsqueeze(0)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    tensor_img = normalize(tensor_img)
        
    _,_,pred= model(tensor_img, inference=True)
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()
    pred = resize(pred,(im_h,im_w))

    return pred

def inference_img_p3m(config, model, img, *args):
    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    if config.TEST.TEST_METHOD=='HYBRID':
        global_ratio = 1/2
        local_ratio = 1
        resize_h = int(h*global_ratio)
        resize_w = int(w*global_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_coutour_1, pred_retouching_1, pred_fusion_1 = inference_once(config, model, scale_img)
        torch.cuda.empty_cache()
        pred_coutour_1 = resize(pred_coutour_1,(h,w))*255.0
        resize_h = int(h*local_ratio)
        resize_w = int(w*local_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_coutour_2, pred_retouching_2, pred_fusion_2 = inference_once(config, model, scale_img)
        torch.cuda.empty_cache()
        pred_retouching_2 = resize(pred_retouching_2,(h,w))
        pred_fusion = get_masked_local_from_global_test(pred_coutour_1, pred_retouching_2)
        return pred_fusion
    elif config.TEST.TEST_METHOD=='RESIZE':
        resize_h = int(h/2)
        resize_w = int(w/2)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img,(new_h,new_w))*255.0
        pred_global, pred_local, pred_fusion = inference_once(config, model, scale_img)
        pred_local = resize(pred_local,(h,w))
        pred_global = resize(pred_global,(h,w))*255.0
        pred_fusion = resize(pred_fusion,(h,w))
        return pred_fusion
    else:
        raise NotImplementedError()

def test_p3m10k(config, model, logger):
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()
    
    arch_predict_dict = {
        'r34': inference_img_p3m,
        'swin': inference_img_p3m,
        'vitae': inference_img_p3m,
    }

    ############################
    # Some initial setting for paths
    ############################
    if config.TEST.DATASET == 'VAL500P':
        val_option = 'P3M_500_P'
    elif config.TEST.DATASET == 'VAL500NP':
        val_option = 'P3M_500_NP'
    else:
        val_option = config.TEST.DATASET
    ORIGINAL_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['ORIGINAL_PATH']
    MASK_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['MASK_PATH']
    TRIMAP_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['TRIMAP_PATH']
    if config.TEST.TEST_PRIVACY:
        PRIVACY_PATH = DATASET_PATHS_DICT['P3M10K'][val_option]['PRIVACY_MASK_PATH']
    ############################
    # Start testing
    ############################
    sad_diffs = 0.
    mse_diffs = 0.
    mad_diffs = 0.
    sad_trimap_diffs = 0.
    mse_trimap_diffs = 0.
    mad_trimap_diffs = 0.
    sad_fg_diffs = 0.
    sad_bg_diffs = 0.
    conn_diffs = 0.
    grad_diffs = 0.
    sad_privacy_diffs = 0.  # for test privacy only
    mse_privacy_diffs = 0.  # for test privacy only
    mad_privacy_diffs = 0.  # for test privacy only
    if config.TEST.SAVE_RESULT:
        result_dir = os.path.join(TEST_RESULT_FOLDER, 'test_{}_{}_{}_{}'.format(config.TAG, config.TEST.DATASET, config.TEST.TEST_METHOD, config.TEST.CKPT_NAME.replace('/', '-')))
        refresh_folder(result_dir)
    model.eval()
    img_list = listdir_nohidden(ORIGINAL_PATH)
    total_number = len(img_list)
    logger.info("===============================")
    logger.info(f'====> Start Testing\n\t--Dataset: {config.TEST.DATASET}\n\t--Test: {config.TEST.TEST_METHOD}\n\t--Number: {total_number}')

    if config.TAG.startswith("debug"):
        img_list = img_list[:10]
    
    if dist.is_initialized():
        img_list = img_list[GET_RANK()::GET_WORLD_SIZE()]
        total_number = len(img_list)
        print('rank {}/{}, total num: {}'.format(GET_RANK(), GET_WORLD_SIZE(), total_number))
        
    for img_name in tqdm(img_list):
        img_path = ORIGINAL_PATH+img_name
        alpha_path = MASK_PATH+extract_pure_name(img_name)+'.png'
        trimap_path = TRIMAP_PATH+extract_pure_name(img_name)+'.png'
        img = np.array(Image.open(img_path))
        trimap = np.array(Image.open(trimap_path))
        alpha = np.array(Image.open(alpha_path))/255.
        img = img[:,:,:3] if img.ndim>2 else img
        trimap = trimap[:,:,0] if trimap.ndim>2 else trimap
        alpha = alpha[:,:,0] if alpha.ndim>2 else alpha

        if config.TEST.TEST_PRIVACY:
            privacy_path = PRIVACY_PATH+extract_pure_name(img_name)+'.png'
            privacy = np.array(Image.open(privacy_path))
            privacy = privacy[:, :, 0] if privacy.ndim>2 else privacy

        with torch.no_grad():
            predict = arch_predict_dict.get(config.MODEL.TYPE, inference_img_p3m)(config, model, img)

            # test on whole image and trimap area
            sad_trimap_diff, mse_trimap_diff, mad_trimap_diff = calculate_sad_mse_mad(predict, alpha, trimap)
            sad_diff, mse_diff, mad_diff = calculate_sad_mse_mad_whole_img(predict, alpha)
            sad_fg_diff, sad_bg_diff = calculate_sad_fgbg(predict, alpha, trimap)
            if config.TEST.FAST_TEST:
                conn_diff = -1
                grad_diff = -1
            else:
                conn_diff = compute_connectivity_loss_whole_image(predict, alpha)
                grad_diff = compute_gradient_whole_image(predict, alpha)
            
            # test on privacy area
            if config.TEST.TEST_PRIVACY:
                sad_privacy_diff, mse_privacy_diff, mad_privacy_diff = calculate_sad_mse_mad_privacy(predict, alpha, privacy)
            else:
                sad_privacy_diff = -1
                mse_privacy_diff = -1
                mad_privacy_diff = -1
                
            logger.info(f"[{img_list.index(img_name)}/{total_number}]\nImage:{img_name}\nsad:{sad_diff}\nmse:{mse_diff}\nmad:{mad_diff}\nsad_trimap:{sad_trimap_diff}\nmse_trimap:{mse_trimap_diff}\nmad_trimap:{mad_trimap_diff}\nsad_fg:{sad_fg_diff}\nsad_bg:{sad_bg_diff}\nconn:{conn_diff}\ngrad:{grad_diff}\nsad_privacy:{sad_privacy_diff}\nmse_privacy:{mse_privacy_diff}\nmad_privacy:{mad_privacy_diff}\n-----------")
            sad_diffs += sad_diff
            mse_diffs += mse_diff
            mad_diffs += mad_diff
            mse_trimap_diffs += mse_trimap_diff
            sad_trimap_diffs += sad_trimap_diff
            mad_trimap_diffs += mad_trimap_diff
            sad_fg_diffs += sad_fg_diff
            sad_bg_diffs += sad_bg_diff
            conn_diffs += conn_diff
            grad_diffs += grad_diff
            sad_privacy_diffs += sad_privacy_diff
            mse_privacy_diffs += mse_privacy_diff
            mad_privacy_diffs += mad_privacy_diff

            if config.TEST.SAVE_RESULT:
                save_test_result(os.path.join(result_dir, extract_pure_name(img_name)+'.png'),predict)
            
    res_dict = {}			
    logger.info("===============================")
    logger.info(f"Testing numbers: {total_number}")
    # res_dict['number'] = total_number
    logger.info("SAD: {}".format(sad_diffs / total_number))
    res_dict['SAD'] = sad_diffs / total_number
    logger.info("MSE: {}".format(mse_diffs / total_number))
    res_dict['MSE'] = mse_diffs / total_number
    logger.info("MAD: {}".format(mad_diffs / total_number))
    res_dict['MAD'] = mad_diffs / total_number
    logger.info("SAD TRIMAP: {}".format(sad_trimap_diffs / total_number))
    res_dict['SAD_TRIMAP'] = sad_trimap_diffs / total_number
    logger.info("MSE TRIMAP: {}".format(mse_trimap_diffs / total_number))
    res_dict['MSE_TRIMAP'] = mse_trimap_diffs / total_number
    logger.info("MAD TRIMAP: {}".format(mad_trimap_diffs / total_number))
    res_dict['MAD_TRIMAP'] = mad_trimap_diffs / total_number
    logger.info("SAD FG: {}".format(sad_fg_diffs / total_number))
    res_dict['SAD_FG'] = sad_fg_diffs / total_number
    logger.info("SAD BG: {}".format(sad_bg_diffs / total_number))
    res_dict['SAD_BG'] = sad_bg_diffs / total_number
    logger.info("CONN: {}".format(conn_diffs / total_number))
    res_dict['CONN'] = conn_diffs / total_number
    logger.info("GRAD: {}".format(grad_diffs / total_number))
    res_dict['GRAD'] = grad_diffs / total_number
    logger.info("SAD PRIVACY: {}".format(sad_privacy_diffs / total_number))
    res_dict['SAD_PRIVACY'] = sad_privacy_diffs / total_number
    logger.info("MSE PRIVACY: {}".format(mse_privacy_diffs / total_number))
    res_dict['MSE_PRIVACY'] = mse_privacy_diffs / total_number
    logger.info("MAD PRIVACY: {}".format(mad_privacy_diffs / total_number))
    res_dict['MAD_PRIVACY'] = mad_privacy_diffs / total_number

    # return int(sad_diffs/total_number)
    print("SAD: {}\nMSE: {}\nMAD: {}\n".format(res_dict['SAD'], res_dict['MSE'], res_dict['MAD']))
    
    if dist.is_initialized():
        for k in res_dict.keys():
            res_dict[k] = torch.tensor(k)  # for reduce only
        print('rank')
    
    return res_dict

def test_samples(config, model):

    arch_predict_dict = {
        'r34': inference_img_p3m,
        'swin': inference_img_p3m,
        'vitae': inference_img_p3m,
    }

    print(f'=====> Test on samples and save alpha results')
    model.eval()
    img_list = listdir_nohidden(SAMPLES_ORIGINAL_PATH)
    refresh_folder(SAMPLES_RESULT_ALPHA_PATH)
    refresh_folder(SAMPLES_RESULT_COLOR_PATH)
    for img_name in tqdm(img_list):
        img_path = SAMPLES_ORIGINAL_PATH+img_name
        try:
            img = np.array(Image.open(img_path))[:,:,:3]
        except Exception as e:
            print(f'Error: {str(e)} | Name: {img_name}')
        h, w, c = img.shape
        if min(h, w)>SHORTER_PATH_LIMITATION:
            if h>=w:
                new_w = SHORTER_PATH_LIMITATION
                new_h = int(SHORTER_PATH_LIMITATION*h/w)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                new_h = SHORTER_PATH_LIMITATION
                new_w = int(SHORTER_PATH_LIMITATION*w/h)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        with torch.no_grad():
            if torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            predict = arch_predict_dict.get(config.MODEL.TYPE, inference_img_p3m)(config, model, img)

        composite = generate_composite_img(img, predict)
        cv2.imwrite(os.path.join(SAMPLES_RESULT_COLOR_PATH, extract_pure_name(img_name)+'.png'),composite)
        predict = predict*255.0
        predict = cv2.resize(predict, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(img_name)+'.png'),predict.astype(np.uint8))

def load_model_and_deploy(config):

    ### build model
    model = build_model(config.MODEL.TYPE)

    ### load ckpt
    ckpt_path = os.path.join(CKPT_SAVE_FOLDER, '{}/{}.pth'.format(config.TAG, config.TEST.CKPT_NAME))
    if torch.cuda.device_count()==0:
        print(f'Running on CPU...')
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['state_dict'], strict=True)
    else:
        print(f'Running on GPU with CUDA...')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        model = model.cuda()
    model.cuda()
    
    ### Test
    if config.TEST.DATASET=='SAMPLES':
        test_samples(config, model)
    elif config.TEST.DATASET in VALID_TEST_DATASET_CHOICE:
        logname = 'test_{}_{}_{}_{}'.format(config.TAG, config.TEST.DATASET, config.TEST.TEST_METHOD, config.TEST.CKPT_NAME.replace('/', '-'))
        logging_filename = TEST_LOGS_FOLDER+logname+'.log'
        logger = create_logger(logging_filename)
        test_p3m10k(config, model, logger)
    else:
        print('Please input the correct dataset_choice (SAMPLES, P3M_500_P or P3M_500_NP).')

if __name__ == '__main__':
    args, config = get_args()
    load_model_and_deploy(config)
