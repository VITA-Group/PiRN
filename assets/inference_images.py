import os
from random import random
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifIRSampler
from ResizeRight.resize_right import resize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",type=str,default='1',help="GPU Index",)
    parser.add_argument("-s","--started_timesteps", type=int, default='10', help='Started timestep for DifFace, parameter N in our paper (Default:100)',)
    parser.add_argument("--aligned", default=True, action='store_true', help='Input are alinged faces')
    parser.add_argument("--draw_box", action='store_true', help='Draw box for face in the unaligned case',)
    parser.add_argument("-t", "--timestep_respacing", type=str,default='250', help='Sampling steps for Improved DDPM, parameter T in out paper (default 250)',)
    parser.add_argument("--in_path", type=str, default='/home/aj32632/pirn/input_image', help='Folder to save the low quality image',)
    parser.add_argument("--out_path", type=str, default='./restored_results/', help='Folder to save the restored results',)
    args = parser.parse_args()

    cfg_path = '/home/aj32632/pirn/configs/configuration.yaml'

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = args.gpu_id
    configs.aligned = args.aligned
    assert args.started_timesteps < int(args.timestep_respacing)
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # build the sampler for diffusion
    sampler_dist = DifIRSampler(configs)

    exts_all = ('jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'bmp')

    #In case the given image is 
    if args.in_path.endswith(exts_all):
        im_path_list = [Path(args.in_path), ]
    else: # for folder
        im_path_list = []
        for ext in exts_all:
            im_path_list.extend([x for x in Path(args.in_path).glob(f'*.{ext}')])

    im_path_list.sort()

     # prepare result path
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    restored_face_dir = Path(args.out_path) / 'output'
    if not restored_face_dir.exists():
        restored_face_dir.mkdir()

    for ii, im_path in tqdm(enumerate(im_path_list)):
        if (ii+1) % 5 == 0:
            print(f"Processing: {ii+1}/{len(im_path_list)}...")

        image_list = []
        im_lq = util_image.imread(im_path, chn='bgr', dtype='uint8')
        input_x, input_y = im_lq.shape[0], im_lq.shape[1]
        im_lq = cv2.resize(im_lq, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        image_list.append(im_lq.astype(np.float32))

        for i in range(0, 5):
            try:
                im_lq = util_image.imread(im_path_list[ii + i], chn='bgr', dtype='uint8')
                im_lq = cv2.resize(im_lq, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                image_list.append(im_lq.astype(np.float32))
            except:
                continue

        im_lq = sum(image_list)/len(image_list)

        face_restored = sampler_dist.sample_func_ir_aligned(
                    y0=im_lq,
                    start_timesteps=args.started_timesteps,
                    need_restoration=True,
                    )[0] #[0,1], 'rgb'
        face_restored = util_image.tensor2img(
                face_restored,
                rgb2bgr=True,
                min_max=(0.0, 1.0),
                ) # uint8, BGR

        face_restored = cv2.resize(face_restored, dsize=(input_y, input_x), interpolation=cv2.INTER_CUBIC)
        save_path = restored_face_dir / im_path.name
        util_image.imwrite(face_restored, save_path, chn='bgr', dtype_in='uint8')

if __name__ == '__main__':
    main()