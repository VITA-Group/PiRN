import os
import sys
import math
import time
import lpips
import random
import datetime
import functools
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

from datapipe.datasets import create_dataset
from models.resample import UniformSampler

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils

from utils import util_net
from utils import util_common
from utils import util_image
from tqdm import tqdm


images_lq = util_common.readline_txt(os.path.join("./train_data/test_data_lq.txt"))
images_restored = [x.replace("turb", "restored") for x in images_lq]
images_gt = util_common.readline_txt(os.path.join("./train_data/test_data_gt.txt"))

fopen = open("train_data/restored_gt_psnr_ssim.txt", "a")
psnr_lq_gt = 0
ssim_lq_gt = 0
count = 0
for i in tqdm(range(1, len(images_gt))):
    try:
        img_lq = util_image.imread(images_restored[i - 1], chn='bgr', dtype='uint8')
        img_gt = util_image.imread(images_gt[i - 1], chn='bgr', dtype='uint8')
        psnr_lq_gt += util_image.calculate_psnr(img_lq, img_gt)
        ssim_lq_gt += util_image.calculate_ssim(img_lq, img_gt)
        count += 1
    except:
        continue

    if i % 500 == 0:
        print(f"[{count}|{i}] Current PSNR: {psnr_lq_gt/i} and SSIM: {ssim_lq_gt/i}")
        fopen.write(f"[{count}|{i}] Current PSNR: {psnr_lq_gt/i} and SSIM: {ssim_lq_gt/i}\n")
        fopen.flush()