import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

from augmenter import Augmenter

class CustomImageFolderDataset(datasets.ImageFolder):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 low_res_augmentation_prob=0.0,
                 crop_augmentation_prob=0.0,
                 photometric_augmentation_prob=0.0,
                 swap_color_channel=False,
                 output_dir='./',
                 ):

        super(CustomImageFolderDataset, self).__init__(root,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       loader=loader,
                                                       is_valid_file=is_valid_file)
        self.root = root
        self.is_augumenter = False
        self.augmenter = Augmenter(crop_augmentation_prob, photometric_augmentation_prob, low_res_augmentation_prob)
        self.swap_color_channel = swap_color_channel
        self.output_dir = output_dir  # for checking the sanity of input images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        try:
            path, target = self.samples[index]
            sample_gt = self.loader(path)
            sample_gt = Image.fromarray(np.asarray(sample_gt)[:,:,::-1])
            sample_input = self.loader(path.replace("gt", "sim"))
            sample_input = Image.fromarray(np.asarray(sample_input)[:,:,::-1])
            noise = torch.load(path.replace("gt", "noise").replace("jpg", "pt"))
        except:
            import random
            return self.__getitem__(random.randint(0, 100))

        if self.swap_color_channel:
            # swap RGB to BGR if sample is in RGB
            # we need sample in BGR
            sample_input = Image.fromarray(np.asarray(sample_input)[:,:,::-1])
            sample_gt = Image.fromarray(np.asarray(sample_gt)[:,:,::-1])

        if self.is_augumenter:
            sample_input = self.augmenter.augment(sample_input)

        sample_input_save_path = os.path.join(self.output_dir, 'training_samples', 'sample_test.jpg')
        if not os.path.isfile(sample_input_save_path):
            os.makedirs(os.path.dirname(sample_input_save_path), exist_ok=True)
            cv2.imwrite(sample_input_save_path, np.array(sample_input))  # the result has to look okay (Not color swapped)

        if self.transform is not None:
            sample_input = self.transform(sample_input)
            sample_gt = self.transform(sample_gt)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample_input, sample_gt, noise, target # turb, gt, noise_loaded, target

if __name__ == '__main__':
    dataset = CustomImageFolderDataset(root = "/data/ajay_data/cvpr2023/iarpa/faces_webface_112x112/gt",
                                       transform=T.Compose([T.ToTensor(),T.RandomCrop(112)]),
                                       target_transform=None)
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 6}
    dataloader = torch.utils.data.DataLoader(dataset, **params)
    batch_data = next(iter(dataloader))
    print(batch_data[0].shape)