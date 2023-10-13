import cv2
import torch
import argparse, os, time, csv
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from model.TMT_MS import TMT_MS
import torch.nn.functional as F
import pandas as pd
import math

def split_to_patches(h, w, s):
    nh = h // s + 1
    nw = w // s + 1
    
    if nh == 1: 
        hpos=[0]
    else:
        ol_h = int((nh * s - h) / (nh - 1))
        h_start = 0
        hpos = [h_start]
        for i in range(1, nh):
            h_start = hpos[-1] + s - ol_h
            if h_start+s > h:
                h_start = h-s
            hpos.append(h_start)
    
    if nw == 1:
        wpos = [0]
    else: 
        ol_w = int((nw * s - w) / (nw - 1))
        w_start = 0
        wpos = [w_start]
        for i in range(1, nw):
            w_start = wpos[-1] + s - ol_w
            if w_start+s > w:
                w_start = w-s
            wpos.append(w_start)
        
    return hpos, wpos
    
def test_spatial_overlap(input_blk, model, patch_size):
    _,c,l,h,w = input_blk.shape
    hpos, wpos = split_to_patches(h, w, patch_size)
    out_spaces = torch.zeros_like(input_blk)
    out_masks = torch.zeros_like(input_blk)
    for hi in hpos:
        for wi in wpos:
            input_ = input_blk[..., hi:hi+patch_size, wi:wi+patch_size]
            output_ = model(input_)
            out_spaces[..., hi:hi+patch_size, wi:wi+patch_size].add_(output_)
            out_masks[..., hi:hi+patch_size, wi:wi+patch_size].add_(torch.ones_like(input_))
    return out_spaces / out_masks
    
      
def restore_PIL(tensor, b, fidx):
    img = tensor[b, fidx, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW-RGB to HWC-BGR
    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8

    return img 

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
    obj_labels['num_frames'] = pd_obj.iloc[-1][3]+1
    obj_labels['detection'] = []
    obj_labels['crop'] = []
    for i in range(len(pd_obj)):
        if pd_obj.iloc[i][5] > 0: 
            continue
        obj_labels['detection'] += [True if pd_obj.iloc[i][6]=='t' else False]
        obj_labels['crop'].append([pd_obj.iloc[i][7],pd_obj.iloc[i][8],pd_obj.iloc[i][9]+1,pd_obj.iloc[i][10]+1])   
            
    return obj_labels

def crop_coor_calc(cropinfo,pad_size=30,min_size=240, multiple_of_eight=False): 
    '''
    This function calculates the cropping information. 
    cropinfo: a list of the cropping info for a chunk of data (For multiple frame restoration). 
    The output will be padded by pad_size on each side. (This is due to the fact that face bounding box can sometimes be outside the body bounding box.)
    The output will have a minimum size of (min_size, min_size). 
    The output dimensions will be multiple of 8 if multiple_of_eight is True. (This is due to the input requirement of TMT)
    
    Return
    crop_coor: Unions of all areas in cropinfo, with possible padding as describe above. 
    recover_coor: a list of cropping coordinates to crop_back the area in cropinfo. The recover_coor contains pad_size padding on each side of image.
    '''
    temp = list(cropinfo)
    cropinfo = [crop_ for crop_ in cropinfo if crop_ is not None]
    crop_coor = [min([crop[0] for crop in cropinfo]), min([crop[1] for crop in cropinfo]), 
                         max([crop[2] for crop in cropinfo]), max([crop[3] for crop in cropinfo])]
    recover_coor = []
    pad = [pad_size]*4
    
    if crop_coor[2]-crop_coor[0] <= min_size-2*pad_size: 
        pad[0] += math.floor((min_size-2*pad_size-(crop_coor[2]-crop_coor[0]))/2)
        pad[2] += math.ceil((min_size-2*pad_size-(crop_coor[2]-crop_coor[0]))/2)
    elif multiple_of_eight:
        pad[0] += 8-(crop_coor[2]-crop_coor[0]+2*pad_size)%8
            
    
    if crop_coor[3]-crop_coor[1] <= min_size-2*pad_size: 
        pad[1] += math.floor((min_size-2*pad_size-(crop_coor[3]-crop_coor[1]))/2)
        pad[3] += math.ceil((min_size-2*pad_size-(crop_coor[3]-crop_coor[1]))/2)
    elif multiple_of_eight:
        pad[1] += 8-(crop_coor[3]-crop_coor[1]+2*pad_size)%8
        
    for crop_ in temp: 
        if crop_ is None: 
            recover_coor.append(None)
        else: 
            recover_coor.append((crop_[0]-crop_coor[0]+pad[0]-pad_size, crop_[1]-crop_coor[1]+pad[1]-pad_size, crop_[2]-crop_coor[0]+pad[0]+pad_size, crop_[3]-crop_coor[1]+pad[1]+pad_size))

    crop_coor[0] -= pad[0]
    crop_coor[1] -= pad[1]
    crop_coor[2] += pad[2]
    crop_coor[3] += pad[3]

    return tuple(crop_coor), recover_coor
    
def get_args():
    parser = argparse.ArgumentParser(description='Video inference with overlapping patches')
    parser.add_argument('--csvfile', type=str, help='path to the csv file with detection labels',required=True)
    parser.add_argument('--process_list', type=str, help='path to the txt containing the list of file to process', required=True)
    parser.add_argument('--object_id', nargs='+',type=int, help='object_id to process, [start, end], inclusive', required=True)
    
    parser.add_argument('--cuda_number',type=int,default=0,help='cuda GPU id. ie. 0')
    
    parser.add_argument('--root', type=str, default='/scratch/gilbreth/mao114/IARPA/', help='root to the data path')
    parser.add_argument('--save_root', type=str, default='/scratch/gilbreth/mao114/IARPA/processed', help='root to the save path')
    
    parser.add_argument('--start_frame', type=float, default=0.0, help='first frame to be processed, if < 1, it is ratio w.r.t. the entire video, if >1, it is absolute value')
    parser.add_argument('--total_frames', type=int, default=-1, help='number of total frames to be processed')
    parser.add_argument('--patch_size', '-ps', dest='patch_size', type=int, default=240, help='saptial patch size')
    parser.add_argument('--temp_patch', type=int, default=12, help='temporal patch size')
    parser.add_argument('--resize_ratio', type=float, default=1.0, help='saptial resize ratio for both w and h')
    
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    patch_size = args.patch_size
    torch.cuda.set_device(args.cuda_number)
    t = time.time()
    csv_file = pd.read_csv(args.csvfile,sep=',')
    with open(args.process_list) as f: 
        lines = f.readlines()
    print('processing object id %d to %d'%(args.object_id[0],args.object_id[1]))
    for object_id in range(args.object_id[0],args.object_id[1]+1):
        print('start processing object id %d'%object_id)
        object_name = lines[object_id][:-1]
        labels = read_obj_label(csv_file, object_name)

        temp_patch=args.temp_patch
        checkpoint = torch.load('./model_zoo/shuffle_MS_video.pth')



        net = TMT_MS(num_blocks=[2,3,3,4], 
                            heads=[1,2,4,8], 
                            num_refinement_blocks=2, 
                            warp_mode='none', 
                            n_frames=temp_patch, 
                            att_type='shuffle').cuda()
        net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
        for name, param in net.named_parameters():
            param.requires_grad = False

        save_path = os.path.join(args.save_root,*object_name.split('/')[:-1])
        os.makedirs(save_path, exist_ok=True)
        vpath = os.path.join(args.root, object_name)
        turb_vid = cv2.VideoCapture(vpath)

        h, w = int(turb_vid.get(4)), int(turb_vid.get(3))

        all_frames = [turb_vid.read()[1] for i in range(labels['num_frames'])]
        total_frames = len(all_frames)

        test_frame_info = [{'start':0, 'range':[0,9]}]
        num_chunk = (total_frames-1) // 6
        for i in range(1, num_chunk):
            if i == num_chunk-1:
                test_frame_info.append({'start':total_frames-args.temp_patch, 'range':[i*6+3,total_frames]})
            else:
                test_frame_info.append({'start':i*6, 'range':[i*6+3,i*6+9]})

        turb_vid.release()
        if args.resize_ratio != 1:
            hh = int(h * args.resize_ratio)
            ww = int(w * args.resize_ratio)
            all_frames = [cv2.resize(f, (hh,ww)) for f in all_frames]
        else:
            hh, ww = h, w

        inp_frames = []
        out_dict = {}
        frame_idx = 0

        start_t = time.time()
        for i in range(num_chunk):
#             print(f'{i}/{num_chunk}')
            out_range = test_frame_info[i]['range']
            in_range = [test_frame_info[i]['start'], test_frame_info[i]['start']+args.temp_patch]
            crop_info = []
            for j in range(in_range[0],in_range[1]): 
                if labels['detection'][j] == True:
                    crop_info.append(labels['crop'][j])
                else: 
                    crop_info.append(None)
            
            if all(crop_ is None for crop_ in crop_info): 
                for j in range(out_range[0]-in_range[0], out_range[1]-in_range[0]): 
                    out_dict[frame_idx] = None
                    frame_idx += 1
                continue
            crop_coor, recover_coor = crop_coor_calc(crop_info,multiple_of_eight=True)

            inp_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).crop(crop_coor) for img in all_frames[in_range[0]:in_range[1]]]
            ww,hh = inp_imgs[0].size
            inp_imgs = [TF.to_tensor(img) for img in inp_imgs]
            input_ = torch.stack(inp_imgs, dim=1).unsqueeze(0).cuda()

            if max(hh,ww)<patch_size:
                recovered = net(input_)
                recovered = net(recovered)
                recovered = net(recovered)
            else:
                recovered = test_spatial_overlap(input_, net, patch_size)

            recovered = recovered.permute(0,2,1,3,4)
            
            for j in range(out_range[0]-in_range[0], out_range[1]-in_range[0]):
                if crop_info[j] is None:
                    out_dict[frame_idx] = None
                else: 
                    out = restore_PIL(recovered, 0, j)
                    out = Image.fromarray(out).crop(recover_coor[j])
                    out_dict[frame_idx] = np.array(out)

                frame_idx += 1


        np.save(os.path.join(save_path, object_name.split('/')[-1][:-4]+'.npy'), out_dict, allow_pickle=True)

