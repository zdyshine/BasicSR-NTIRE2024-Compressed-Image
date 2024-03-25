# NTIRE 2024 Blind Compressed Image Enhancement Challenge - PixelAtiAI
NTIRE 2024 Blind Compressed Image Enhancement Challenge    
https://codalab.lisn.upsaclay.fr/competitions/17548#learn_the_details    

## Title: Blind JPEG Artifacts Removal via Enhanced Swin-Conv-UNet
Based on the original SCUNet, we made several modifications to en-hance the model’s performance while maintaining a maximum of 300M parameters.Firstly, we increased the number of channels in the model from [64, 128,256, 512] to [96, 192, 384, 768]. Secondly, we increased the number of downsampling and upsampling modules in the model. This changes in SCUNet allows the model to capture more features and details in the input data. By increasing the model parameters, the model has gained stronger learning capabilities. Compared to the baseline, the performance metrics have been significantly improved.    

## Description
#### train code
Code Description:
The training code is based on BasicSR and has undergone simple modifications. The installation and running commands are as follows:
Detailed reference： https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md

#### Training environment configuration
```
cd BasicSR
pip install -r requirements.txt
python setup.py develop
```

#### Modifying Training Configuration Files
Specify the following data path: 

```
cd BasicSR
# Phase 1
vim options/JPEG_001_train_SCUNet.yml
line: 14~20 # for train
    dataroot_gt: /dataset/SR/SR_Image_Mix/DIV2K_train_HR
    dataroot_DIV8K: /dataset/SR/SR_Image_Mix/DIV8K
    dataroot_Flicker2K: /dataset/SR/SR_Image_Mix/Flicker2K
    dataroot_Unsplash2K: /dataset/SR/SR_Image_Mix/Unsplash2K
    dataroot_LSDIR: /dataset/SR/LSDIR_Dataset
    dataroot_nomos8k: /dataset/SR/nomos8k
    dataroot_nomos_uni_gt: /dataset/SR/nomos_uni_gt
    
line: 38 # for val
    dataroot_gt: /dataset/SR/DIV2K_0001_0020

# Phase 2
vim options/JPEG_001_train_SCUNet_PSNR.yml
```

#### Training commands
```
cd BasicSR
# Phase 1 # 4*3090
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/JPEG_001_train_SCUNet.yml --launcher pytorch

# Phase 2 # 4*A6000
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/JPEG_001_train_SCUNet_PSNR.yml --launcher pytorch

```
before Phase 2, the 50th line of JPEG_001_train_SCUNet_PSNR.yml should be changed to the model path with the best validation set metric in Phase 1.

#### test
```
# 1*3090
cd test_code
python main_test_ntire.py --test_path './validation_JPEG' --checkpoint './net_g_ema_68000.pth' --enhance

# --test_path: is the path of validation/test dataset
# --checkpoint: is the best model path in Phase 2
# --enhance: is self-ensemble for x4
```

## Training data
DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip    
Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar    
DIV8K: https://competitions.codalab.org/competitions/22217#participate    
Unsplash2K：https://drive.google.com/file/d/1IDxEUM6QL7JE8p8ms2Q-aeOAvu9Cj4vl/view?usp=sharing    
nomos8k: https://drive.google.com/file/d/1ppTpi1-FQEBp908CxfnbI5Gc9PPMiP3l/view?usp=sharing    
nomos_uni: https://drive.google.com/file/d/1LVS7i9J3mP9f2Qav2Z9i9vq3y2xsKxA_/view?usp=sharing    
LSDIR: https://data.vision.ee.ethz.ch/yawli/    

#### Dataset placement directory

```
dataset/SR:
        --|Image_Mix
             --|DIV2K
                 00021.png, ..., 0800.png
             --|DIV8K
                 0001.png, ..., 1490.png
             --|Flickr2K
                 000001.png, ..., 002650.png
             --|Unsplash2K
                 000000.png,..., 000497.png
        --|LSDIR
             --|0001000
                 * .png
                ...
             --|0085000
                 * .png
        --|nomos8k
             * .png
        --|nomos_uni_gt
             * .png
        --|DIV2K_0001_0020
             * .png
```
DIV2K_0001_0020 is validation dataset directory: 
00000.png ~ 00021.png of DIV2K

## Thanks
https://github.com/XPixelGroup/BasicSR    
https://github.com/cszn/SCUNet    
