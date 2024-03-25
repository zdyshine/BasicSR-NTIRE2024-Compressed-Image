import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data
import glob
from copy import deepcopy
import PIL
from PIL import Image, features
from io import BytesIO

#print('PIL.versionn: ', PIL.__version__) # 6.2
#print('Image.core.jpeglib_version: ', Image.core.jpeglib_version) # 6.2
#print('libjpeg_turbo:', features.check_feature("libjpeg_turbo"))

assert(PIL.__version__=='10.0.1')
assert(Image.core.jpeglib_version=='9.0')
assert(features.check_feature("libjpeg_turbo")==False)

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def jpeg_artifact(image, quality=90, verbose=False):
    with BytesIO() as f:
        image.save(f, format='JPEG', quality=int(quality))
        f.seek(0)
        image_jpeg = Image.open(f).convert('RGB')
    if verbose:
        print('JPEG quality =', quality)
    return image_jpeg


@DATASET_REGISTRY.register(suffix='basicsr')
class NTIREJPEGDataset(data.Dataset):
    """
    """

    def __init__(self, opt):
        super(NTIREJPEGDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        # self.gt_folder_face = opt['dataroot_gt_face']
        self.dataroot_DIV8K_Crop = opt.get('dataroot_DIV8K_Crop', False)
        self.dataroot_Flicker2K = opt.get('dataroot_Flicker2K', False)
        self.dataroot_Unsplash2K = opt.get('dataroot_Unsplash2K', False)
        self.dataroot_LSDIR = opt.get('dataroot_LSDIR', False)
        self.dataroot_nomos8k = opt.get('dataroot_nomos8k', False)
        self.dataroot_nomos_uni_gt = opt.get('dataroot_nomos_uni_gt', False)

        # self.paths = glob.glob(os.path.join(self.gt_folder, '*.png')) # 85791
        self.paths = glob.glob(os.path.join(self.gt_folder, '*.png')) # [:85120] # [:83200]
        print('=================== Train With DIV2K_train_HR ===================', len(self.paths))
        if self.dataroot_DIV8K_Crop:
            self.paths += glob.glob(os.path.join(self.dataroot_DIV8K_Crop, '*.png'))
            self.paths += glob.glob(os.path.join(self.dataroot_DIV8K_Crop+'1', '*.png'))
            self.paths += glob.glob(os.path.join(self.dataroot_DIV8K_Crop+'2', '*.png'))
            print('=================== Train With DIV8K ===================', len(self.paths))
        if self.dataroot_Flicker2K:
            self.paths += glob.glob(os.path.join(self.dataroot_Flicker2K, '*.png'))
            print('=================== Train With Flicker2K ===================', len(self.paths))
        if self.dataroot_Unsplash2K:
            self.paths += glob.glob(os.path.join(self.dataroot_Unsplash2K, '*.png'))
            print('=================== Train With Unsplash2K   ===================', len(self.paths))
        if self.dataroot_LSDIR:
            self.paths += glob.glob(os.path.join(self.dataroot_LSDIR, '*/*.png'))
            print('=================== Train With LSDIR   ===================', len(self.paths))
        if self.dataroot_nomos8k:
            self.paths += glob.glob(os.path.join(self.dataroot_nomos8k, '*.png'))
            print('=================== Train With nomos8k   ===================', len(self.paths))
        if self.dataroot_nomos_uni_gt:
            self.paths += glob.glob(os.path.join(self.dataroot_nomos_uni_gt, '*.png'))
            print('=================== Train With nomos_uni_gt   ===================', len(self.paths))
        
        if self.opt['phase'] != 'train':
            self.paths = glob.glob(os.path.join(self.gt_folder, '*.png'))
            self.paths = self.paths * 4
            self.paths = self.paths[:61] # 61
            self.qf_list = list(range(10, 71))

        self.paths = sorted(self.paths)

    def __getitem__(self, index):
        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            gt_path = self.paths[index]
            pillow_img = Image.open(gt_path)  # 0~255, uint8
            pillow_img = pillow_img.crop((100, 100, 1024, 1024))
            img_gt = np.float32(np.array(deepcopy(pillow_img))) / 255.  # RGB, 0~1
            # print('img_gt: ', img_gt.max(), img_gt.min())
            # add jpeg
            qf = self.qf_list[int(index)] # 30张图，0~60的idx
            img_lq = jpeg_artifact(pillow_img, quality=qf)
            img_lq = np.float32(np.array(img_lq)) / 255.  # RGB, 0~1

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            noise_level = (100 - qf) / 100.0
            noise_level = torch.FloatTensor([noise_level])
            # del image_stream

            return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path, 'qf': noise_level}

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        pillow_img = Image.open(gt_path) # 0~255, uint8
        pillow_img = png_temp(pillow_img)
        try:
            img_gt = np.float32(np.array(deepcopy(pillow_img))) / 255.  # RGB, 0~1
        except:
            print('1111111111 gt_path: ',gt_path)
            exit()
        #print('img_gt: ', img_gt.max(), img_gt.min())
        # add jpeg
        qf = random.randint(10, 71)
        img_lq = jpeg_artifact(pillow_img, quality=qf)
        img_lq = np.float32(np.array(img_lq)) / 255. # RGB, 0~1
        # random crop
        gt_size = self.opt['gt_size']
        scale = 1
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
        # flip, rotation
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        noise_level = (100 - qf) / 100.0
        noise_level = torch.FloatTensor([noise_level])
        # del image_stream
        return_d = {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'qf': noise_level}
        return return_d

    def __len__(self):
        return len(self.paths)
