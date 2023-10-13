import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import cv2
import time
import queue
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from skimage import img_as_ubyte
from timeit import default_timer as timer
from datetime import timedelta

from utils import util_opts
from utils import util_image
from utils import util_common

from sampler import DifIRSampler
from ResizeRight.resize_right import resize

def read_obj_label(csv_pd_obj, entry): 
    '''
    This function is a wrapper around the panda dataframe. it returns an all the useful labels for restoration
    obj_labels[num_frames]: total number of frames
    obj_labels[detection][i]: True if successfull detection
    obj_labels[crop][i]: crop bounding box of the ith frame. 
    ps: object idx: BTS1:[0-432], BTS1.1[433,559], BTS2[560:911]
    '''
    pd_obj = csv_pd_obj[csv_pd_obj['media_path'] == entry]
    obj_labels = {}
    if len(pd_obj) == 0: 
        obj_labels['num_frames'] = 0
    else: 
        obj_labels['num_frames'] = pd_obj.iloc[-1][5]+1
    obj_labels['detection'] = []
    obj_labels['crop'] = []
    for i in range(len(pd_obj)):
        if pd_obj.iloc[i][7] > 0: 
            continue
        obj_labels['detection'] += [True if pd_obj.iloc[i][8]=='t' else False]
        obj_labels['crop'].append([pd_obj.iloc[i][9],pd_obj.iloc[i][10],pd_obj.iloc[i][11]+1,pd_obj.iloc[i][12]+1])   
            
    return obj_labels

def get_queue_average(_queue):
    return sum(_queue.queue)/_queue.qsize()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id",type=str,default='1',help="GPU Index",)
    parser.add_argument("-s","--started_timesteps", type=int, default='10', help='Started timestep for DifFace, parameter N in our paper (Default:100)',)
    parser.add_argument("--aligned", default=True, action='store_true', help='Input are alinged faces')
    parser.add_argument("--draw_box", action='store_true', help='Draw box for face in the unaligned case',)
    parser.add_argument("-t", "--timestep_respacing", type=str,default='250', help='Sampling steps for Improved DDPM, parameter T in out paper (default 250)',)
    parser.add_argument("--in_path", type=str, default='./test_outputs/averaged.png', help='Folder to save the low quality image',)
    parser.add_argument("--out_path", type=str, default='./results', help='Folder to save the restored results',)
    args = parser.parse_args()

    cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'
    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.gpu_id = args.gpu_id
    configs.aligned = args.aligned
    assert args.started_timesteps < int(args.timestep_respacing)
    configs.diffusion.params.timestep_respacing = args.timestep_respacing

    # build the sampler for diffusion
    sampler_dist = DifIRSampler(configs)
    
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)
    restored_face_dir = Path(args.out_path) / 'restored_faces'
    if not restored_face_dir.exists():
        restored_face_dir.mkdir()

    data_root = "/data/ajay_data/iarpa_data_iccv/"
    with open("{}poor_f.txt".format(data_root)) as file: 
        lines = file.readlines()
    csv_pd_obj = pd.read_csv('{}uodetect_bts1_probe_face.csv'.format(data_root),sep=',')
    print("BTS Loading Done")

    start = timer()

    line = 'BGC1/BTS1/full/G00448/field/400m/face/G00448_set2_stand_1622658508188_ecb70455.mp4'
    label = read_obj_label(csv_pd_obj, line)
    turb_vid = cv2.VideoCapture(os.path.join(data_root+line))
    
    frame_queue = queue.Queue(maxsize=8)
    currentframe = 0
    out_dict = {}
    while(True):
        ret, frame = turb_vid.read()
        
        if ret:
            if label["detection"][currentframe] == False:
                currentframe += 1
                if not frame_queue.empty(): frame_queue.get()
                out_dict[currentframe] = None
                continue

            x1, y1, x2, y2 = label['crop'][currentframe][0] - 30, label['crop'][currentframe][1] - 30, label['crop'][currentframe][2] + 30, label['crop'][currentframe][3] + 30
            crop_region = [x1, y1, x2, y2]
            cropped = frame[y1:y2, x1:x2, :]
            
            im_queue = cv2.resize(cropped, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            
            if frame_queue.full():
                frame_queue.get()
                frame_queue.put(im_queue.astype(np.float32))
            else:
                frame_queue.put(im_queue.astype(np.float32))
            im_lq = get_queue_average(frame_queue)
            im_lq = im_lq.astype(np.uint8)

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
            face_restored = cv2.resize(face_restored, dsize=(cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_CUBIC)
            out_dict[currentframe] = face_restored
            save_path = restored_face_dir / "{}.jpg".format(currentframe)
            util_image.imwrite(face_restored, save_path, chn='bgr', dtype_in='uint8')
            
            currentframe += 1
            mid = timer()
            if currentframe % 100 == 0: 
                print("Number of frames processed : {}/{} in {}".format(currentframe, label["num_frames"], timedelta(seconds=mid-start)))
        else:
            break
    end = timer()
    print(timedelta(seconds=end-start))
    save_path = data_root + "restoration_output/" + line.replace(".mp4", "")
    if not os.path.exists(save_path):
        Path(save_path).mkdir(parents=True)
    print(f"{save_path}/restored.npy")
    np.save(f"{save_path}/restored.npy", out_dict, allow_pickle=True)
    print("Processing Done")