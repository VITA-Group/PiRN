import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import random
from PIL import Image
import os
import torch.nn.functional as F
from simulator_new import Simulator
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as utils
from tqdm import tqdm

mode = "test"
path = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112"

image_dirs = os.listdir(os.path.join(path, "imgs"))
print("Number of classes/Dirs: {}".format(len(image_dirs)))

device = torch.device('cuda:0')
turb_params = {
                'img_size': (112,112),
                'D':0.071,        # Apeture diameter
                'r0':0.071,      # Fried parameter 
                'L':100,       # Propogation distance
                'thre':0.02,   # Used to suppress small values in the tilt correlation matrix. Increase 
                                # this threshold if the pixel displacement appears to be scattering
                'adj':1,        # Adjusting factor of delta0 for tilt matrix
                'wavelength':0.500e-6,
                'corr':-0.05,    # Correlation strength for PSF without tilt. suggested range: (-1 ~ -0.01)
                'zer_scale':1   # manually adjust zernike coefficients of the PSF without tilt.
            }
transform = T.Compose([T.ToTensor(),T.RandomCrop(112)])
simulator = Simulator(turb_params).to(device,dtype=torch.float32)

for img_dir in tqdm(image_dirs):
    path_img = os.path.join(path, "imgs", img_dir)
    os.mkdir(os.path.join(path, "gt", img_dir))
    os.mkdir(os.path.join(path, "noise", img_dir))
    os.mkdir(os.path.join(path, "sim", img_dir))

    path_images = os.listdir(path_img)
    for item in path_images:
        if "jpg" not in item:
            continue
        input_image  = os.path.join(path, "gt", img_dir, item)
        output_image  = os.path.join(path, "sim", img_dir, item)
        noise_image = os.path.join(path, "noise", img_dir, item.replace("jpg", "pt"))
        item_tensor = transform(Image.open(os.path.join(path_img, item)))
        im = item_tensor.unsqueeze(0).to(device)
        noise, _, _, sim = simulator(im)
        noise_to_save = (noise[0].detach().cpu(), noise[1].detach().cpu())
        utils.save_image(item_tensor, input_image)
        utils.save_image(sim.squeeze(0), output_image)
        torch.save(noise_to_save, noise_image)