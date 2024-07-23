<h1 align="center">Rethinking Portrait Matting with Privacy Preserving</h1>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#prepare-datasets">Prepare Datasets</a> |
  <a href="#pretrained-models">Pretrained Models</a> |
  <a href="#train-on-p3m-10k">Train on P3M-10k</a> |
  <a href="#test">Test</a> |
  <a href="#inference-code---how-to-test-on-your-images">Inference</a>
</p>


## Installation
Requirements:

- Python 3.7.7+ with Numpy and scikit-image
- Pytorch (version>=1.7.1)
- Torchvision (version 0.8.2)

1. Clone this repository

    `git clone https://github.com/ViTAE-Transformer/P3M-Net.git`;

2. Go into the repository

    `cd P3M-Net`;

3. Create conda environment and activate

    `conda create -n p3m python=3.7.7`,

    `conda activate p3m`;

4. Install dependencies, install pytorch and torchvision separately if you need

    `pip install -r requirements.txt`,

    `conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`.

Our code has been tested with Python 3.7.7, Pytorch 1.7.1, Torchvision 0.8.2, CUDA 10.2 on Ubuntu 18.04.


## Prepare Datasets

<!-- | Dataset | <p>Dataset Link<br>(Google Drive)</p> | <p>Dataset Link<br>(Baidu Wangpan 百度网盘)</p> | Dataset Release Agreement|
| :----:| :----: | :----: | :----: | 
|<strong>P3M-10k</strong>|[Link](https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1)|[Link](https://pan.baidu.com/s/1X9OdopT41lK0pKWyj0qSEA) (pw: fgmc)|[Agreement (MIT License)](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf)| 
|<strong>P3M-10k facemask</strong> (optional)|[Link](https://drive.google.com/file/d/1I-71PbkWcivBv3ly60V0zvtYRd3ddyYs/view?usp=sharing)|[Link](https://pan.baidu.com/s/1D9Kj_OIJbFTsqWfbMPzh_g) (pw: f772)|[Agreement (MIT License)](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf)|  -->

| Dataset | <p>Dataset Link<br>(Google Drive)</p> | <p>Dataset Link<br>(Baidu Wangpan 百度网盘)</p> | Dataset Release Agreement|
| :----:| :----: | :----: | :----: | 
|<strong>P3M-10k</strong>|[Link](https://drive.google.com/file/d/1odzHp2zbQApLm90HH_Cvr5b5OwJVhEQG/view?usp=sharing)|[Link](https://pan.baidu.com/s/1aEmEXO5BflSp5hiA-erVBA?pwd=cied) (pw: cied) |[Agreement (MIT License)](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf)| 


1. Download the datasets P3M-10k from the above links and unzip to the folders `P3M_DATASET_ROOT_PATH`, set up the configuratures in the file `core/config.py`. Please make sure that you have checked out and agreed to the agreements.

After dataset preparation, the structure of the complete datasets should be like the following. 
```text
P3M-10k
├── train
    ├── blurred_image
    ├── mask (alpha mattes)
    ├── fg_blurred
    ├── bg
    ├── facemask
├── validation
    ├── P3M-500-P
        ├── blurred_image
        ├── mask
        ├── trimap
        ├── facemask
    ├── P3M-500-NP
        ├── original_image
        ├── mask
        ├── trimap
```

2. If you want to test on RWP Test set, please download the original images and alpha mattes at <a href='https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/EXVd6ga9f9xBjkDv6nPMDtcB_rYaJhnkkS6XGvmzc_6Rfw'>this link</a>.

After datasets preparation, the structure of the complete datasets should be like the following.
```text
RealWorldPortrait-636
├── image
├── alpha
├── ...
```

## Pretrained Models

Here we provide the model <strong>P3M-Net(ViTAE-S)</strong> that is trained on P3M-10k for testing.

| Model|  Google Drive | Baidu Wangpan(百度网盘) | 
| :----: | :----:| :----: | 
| P3M-Net(ViTAE-S)  | [Link](https://drive.google.com/file/d/1QbSjPA_Mxs7rITp_a9OJiPeFRDwxemqK/view?usp=sharing) | [Link](https://pan.baidu.com/s/19FuiR1RwamqvxfhdXDL1fg) (pw: hxxy) |

Here we provide the pretrained models of all backbones for training.

| Model|  Google Drive | Baidu Wangpan(百度网盘) | 
| :----: | :----:| :----: | 
| pretrained models  | [Link](https://drive.google.com/file/d/1V2xt0BWCVx550Ll7GGfquvopTLseX9gY/view?usp=sharing) | [Link](https://pan.baidu.com/s/1eJ7mTLQszEtMJHJ2zn3dag?pwd=gxn9) (pw:gxn9) |


## Train on P3M-10k

1. Download P3M-10k dataset in root `P3M_DATASET_ROOT_PATH` (set up in `core/config.py`);

2. Download the pretrained models of all backbones in the previous section, and set up the output folder `REPOSITORY_ROOT_PATH` in `core/config.py`. The folder structure should be like the following,
```text
[REPOSITORY_ROOT_PATH]
├── logs
├── models
    ├── pretrained
        ├── r34mp_pretrained_imagenet.pth.tar
        ├── swin_pretrained_epoch_299.pth
        ├── vitae_pretrained_ckpt.pth.tar
    ├── trained
```

3. Set up parameters in `scripts/train.sh`, specify config file `cfg`, name for the run `nickname`, etc. Run the file:

    `chmod +x scripts/train.sh`

    `./scripts/train.sh`

## Test

Set up parameters in `scripts/test.sh`, specify config file `cfg`, name for the run `nickname`, etc. Run the file:

    `chmod +x scripts/test.sh`

    `./scripts/test.sh`


<details>
  <summary><b>Test using provided model</b></summary>

### Test on P3M-10k

1. Download provided model on P3M-10k as shown in the previous section, unzip to the folder `models/pretrained/`;

2. Download P3M-10k dataset in root `P3M_DATASET_ROOT_PATH` (set up in `core/config.py`);

3. Setup parameters in `scripts/test_dataset.sh`, choose `dataset=P3M10K`, and `valset=P3M_500_NP` or `valset=P3M_500_P` depends on which validation set you want to use, run the file:

    `chmod +x scripts/test_dataset.sh`

    `./scripts/test_dataset.sh`

4. The results of the alpha matte will be saved in folder `args.test_result_dir`. Note that there may be some slight differences of the evaluation results with the ones reported in the paper due to some packages versions differences and the testing strategy. 

### Test on RWP

1. Download provided model on P3M-10k as shown in the previous section, unzip to the folder `models/pretrained/`;

2. Download RWP dataset in root `RWP_TEST_SET_ROOT_PATH` (set up in `core/config.py`). Download link is <a href='https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/qyu13_jh_edu/EXVd6ga9f9xBjkDv6nPMDtcB_rYaJhnkkS6XGvmzc_6Rfw'>here</a>;

3. Setup parameters in `scripts/test_dataset.sh`, choose `dataset=RWP` and `valset=RWP`, run the file:

    `chmod +x scripts/test_dataset.sh`

    `./scripts/test_dataset.sh`

4. The results of the alpha matte will be saved in folder `args.test_result_dir`. Note that there may be some slight differences of the evaluation results with the ones reported in the paper due to some packages versions differences and the testing strategy. 

### Test on Samples

1. Download provided model on P3M-10k as shown in the previous section, unzip to the folder `models/pretrained/`;

2. Download images in root `SAMPLES_ROOT_PATH/original` (set up in config.py)

3. Set up parameters in `scripts/test_samples.sh`, and run the file:

    `chmod +x samples/original/*`

    `chmod +x scripts/test_samples.sh`

    `./scripts/test_samples.sh`

4. The results of the alpha matte will be saved in folder `SAMPLES_RESULT_ALPHA_PATH` (set up in config.py). The color results will be saved in folder `SAMPLES_RESULT_COLOR_PATH` (set up in config.py). Note that there may be some slight differences of the evaluation results with the ones reported in the paper due to some packages versions differences and the testing strategy. 

</details>

## Inference Code - How to Test on Your Images

<p align="justify">Here we provide the procedure of testing on sample images by our pretrained <strong>P3M-Net(ViTAE-S)</strong> model:</p>

1. Setup environment following this [instruction page](https://github.com/ViTAE-Transformer/P3M-Net/tree/main/core);

2. Insert the path `REPOSITORY_ROOT_PATH` in the file `core/config.py`;

3. Download the pretrained P3M-Net(ViTAE-S) model from here ([Google Drive](https://drive.google.com/file/d/1QbSjPA_Mxs7rITp_a9OJiPeFRDwxemqK/view?usp=sharing) | [Baidu Wangpan](https://pan.baidu.com/s/19FuiR1RwamqvxfhdXDL1fg) (pw: hxxy))) and unzip to the folder `models/pretrained/`;

4. Save your sample images in folder `samples/original/.`;
    
5. Setup parameters in the file `scripts/test_samples.sh` and run by:

    `chmod +x scripts/test_samples.sh`

    `scripts/test_samples.sh`;

6. The results of alpha matte and transparent color image will be saved in folder `samples/result_alpha/.` and `samples/result_color/.`.

<p align="justify">We show some sample images, the predicted alpha mattes, and their transparent results as below. We use the pretrained <strong>P3M-Net(ViTAE-S)</strong> model from section <a href="#p3m-net-and-variants">P3M-Net and Variants</a> with `RESIZE` test strategy.</p>

<img src="../samples/original/p_015cd10e.jpg" width="33%"><img src="../samples/result_alpha/p_015cd10e.png" width="33%"><img src="../samples/result_color/p_015cd10e.png" width="33%">
<img src="../samples/original/p_819ea202.jpg" width="33%"><img src="../samples/result_alpha/p_819ea202.png" width="33%"><img src="../samples/result_color/p_819ea202.png" width="33%">
<img src="../samples/original/p_0865636e.jpg" width="33%"><img src="../samples/result_alpha/p_0865636e.png" width="33%"><img src="../samples/result_color/p_0865636e.png" width="33%">

