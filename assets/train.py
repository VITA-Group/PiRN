import os
import shutil
import sys
import time
import torch
import argparse
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloder import CustomImageFolderDataset
from models.TurbulenceNet import *
from utils.misc import to_psnr, adjust_learning_rate, print_log, ssim, lr_schedule_cosdecay
from torchvision.models import vgg16
import torchvision.utils as utils
import math
import torchvision.transforms as T
from tqdm import tqdm
from simulator_new import Simulator
from models.adaface_model import *
import loss
import optim

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
use_cuda = torch.cuda.is_available()

if __name__ == "__main__":
    
    fopen = open("training_log/log.txt", "w")

    train_batch_size, test_batch_size = 6, 32
    num_epochs = 80
    all_T = 100000
    save_dir = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112/checkpoint"

    if not os.path.isfile(save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    dataset = CustomImageFolderDataset(root = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112/gt",
                                       transform=T.Compose([T.ToTensor(),
                                                            T.Normalize(
                                                                         mean=[0.485, 0.456, 0.406],
                                                                         std= [0.229, 0.224, 0.225])]),
                                       target_transform=None)
    adaface_num_subjects = len(dataset.classes)
    print("Number of Subjects for Adaface Training : {}".format(adaface_num_subjects))
    params = {'batch_size': train_batch_size,
              'shuffle': True,
              'num_workers': 8}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    print("DATALOADER DONE!")

    net = get_model()
    adaface_model = build_model(model_name='ir_101', loc_cross_att=2, aux_feature_dim=3)
    head = loss.AdaFace(classnum=adaface_num_subjects)
    net.cuda()
    head.cuda()
    adaface_model.cuda()

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    optimizer, lr_scheduler = optim.configure_optimizers(net, adaface_model, head, 
                                                         lr=1e-4, momentum=0.9, lr_milestones=[6, 15, 25, 50], lr_gamma=0.1)
    
    
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
    simulator = torch.nn.DataParallel(Simulator(turb_params, data_path="utils")).cuda()
    print("===> Training Start ...")
    for epoch in range(num_epochs):
        print("====> Training Eoch [{}/{}]".format(epoch, num_epochs))
        start_time = time.time()
        psnr_list = []

        # --- Save the network parameters --- #
        checkpoint = {
            'epoch': epoch,
            'model_net': net.state_dict(),
            'model_head': head.state_dict(),
            'model_adaface': adaface_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler
        }
        torch.save(checkpoint, '{}/restoration_ckp_{}.pth'.format(save_dir, epoch))
        # torch.save(net.state_dict(), '{}/checkpoint_restore_adam{}.pth'.format(save_dir, epoch))
        # torch.save(head.state_dict(), '{}/checkpoint_head{}_adam.pth'.format(save_dir, epoch))
        # torch.save(adaface_model.state_dict(), '{}/checkpoint_adaface_adam{}.pth'.format(save_dir, epoch))

        for batch_id, train_data in tqdm(enumerate(dataloader)):
            if batch_id > 5000:
                break
            # step_num = batch_id + epoch * 5000 + 1
            # lr=lr_schedule_cosdecay(step_num,all_T)
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr
            turb, gt, noise_loaded, target = train_data
            turb = turb.cuda()
            gt = gt.cuda()
            noise = (noise_loaded[0].squeeze(1).cuda(), noise_loaded[1].squeeze(1).cuda())
            target = target.cuda()

            optimizer.zero_grad()
            # optimizer2.zero_grad()

            # --- Forward + Backward + Optimize --- #
            net.train()
            head.train()
            adaface_model.train()
            feature_embedding, J, T, I = net(turb)
            embeddings, norms = adaface_model(J, feature_embedding)
            normalized_embedding  = embeddings / norms

            noise, _, _, sim_I = simulator(J, noise)

            Rec_Loss1 = F.smooth_l1_loss(J, gt)
            Rec_Loss2 = F.smooth_l1_loss(I, turb)
            Rec_Loss3 = F.smooth_l1_loss(sim_I, turb)
            cosine_with_margin = head(normalized_embedding, norms, target)
            AdaFace_loss = torch.nn.CrossEntropyLoss()(cosine_with_margin, target)

            loss = Rec_Loss1 + Rec_Loss2 + Rec_Loss3 + AdaFace_loss
            loss.backward()
            optimizer.step()
            # optimizer2.step()
            # lr_scheduler2.step()
            # --- To calculate average PSNR --- #
            psnr_list.extend(to_psnr(J, gt))

            if not (batch_id % 200):
                fopen.write('Epoch: {}, Iteration: {}, Loss: {:.3f}, Adaface_Loss: {:.3f}, Rec_Loss1: {:.3f}, Rec_loss2: {:.3f}, Rec_loss3: {:.3f}'.format(epoch, batch_id, loss, AdaFace_loss, Rec_Loss1, Rec_Loss2, Rec_Loss3))
                print('Epoch: {}, Iteration: {}, Loss: {:.3f}, Adaface_Loss: {:.3f}, Rec_Loss1: {:.3f}, Rec_loss2: {:.3f}, Rec_loss3: {:.3f}'.format(epoch, batch_id, loss, AdaFace_loss, Rec_Loss1, Rec_Loss2, Rec_Loss3))
                fopen.flush()
            if not (batch_id % 1000):
                utils.save_image(J[0], "{}/training_samples/recons_batch_id_{}.jpg".format(save_dir, batch_id))
                utils.save_image(gt[0], "{}/training_samples/gt_batch_id_{}.jpg".format(save_dir, batch_id))
                utils.save_image(sim_I[0], "{}/training_samples/sim_batch_id_{}.jpg".format(save_dir, batch_id))

        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)
        print("Train PSNR : {:.3f}".format(train_psnr))

    fopen.close()
    
    
