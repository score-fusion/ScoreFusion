import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)
from dataset_brats import NiftiPairImageGenerator
import os
from skimage.transform import resize
# from nilearn import surface
import nibabel as nib
from skimage import exposure
import argparse
import pandas as pd
import glob

import sys
import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
# from nilearn import surface
import nibabel as nib
from skimage import exposure
import argparse
import pandas as pd
import glob

# from dataset_brast import *

class Brast_2D(data.Dataset):
    def __init__(self, dataset_folder,
                input_modality, 
                target_modality, 
                input_size = 192, 
                depth_size = 152, 
                transform = None, 
                target_transform = None,
                full_channel_mask=False,
                combine_output=False, 
                train = True,
                mask_mode = 'center',
                data_len=-1,
                slice_direction = 1,
                global_pos_emb = True,
                none_zero_mask = True,
                residual_train = True,
                nearby_slices = 0):
        print("input modality:", input_modality, "target modality",target_modality,"slice_direction", slice_direction)
        self.input_modality = input_modality
        self.target_modality = target_modality
        self.slice_direction = slice_direction
        imgs = NiftiPairImageGenerator(dataset_folder, 
                                       input_modality, 
                                       target_modality, 
                                       input_size, 
                                       depth_size, 
                                       input_channel = 8,
                                       transform = transform, 
                                       target_transform = target_transform, 
                                       full_channel_mask = full_channel_mask, 
                                       combine_output = combine_output, 
                                       global_pos_emb = global_pos_emb,
                                       none_zero_mask = none_zero_mask,
                                       residual_training = residual_train,
                                       train = train)
        print("train:", train, "len img:", len(imgs) )
        print("folders:", imgs.input_folders)
        if data_len > 0:
            print("!! using only {} data samples".format(data_len))
            imgs.input_folders = imgs.input_folders[:int(data_len)]
        self.imgs = imgs
        self.slice_direction = slice_direction
        if self.slice_direction not in [1,2,3]:
            raise NotImplementedError("Slice direction out of range")
        if self.slice_direction == 1:
            self.slice_len = 152
        else:
            self.slice_len = 192
        print("slicing in direction", self.slice_direction, "with slice len", self.slice_len)
        self.image_size = (depth_size,input_size)
        self.mask_mode = mask_mode
        self.train = train
        self.nearby_slice = nearby_slices
        if nearby_slices != 0 and nearby_slices != 2:
            raise NotImplementedError("Nearby slices only support 0 and 2")

    def __getitem__(self, index):
        ret = {}
        idx_3d = index // self.slice_len
        idx_slice = index % self.slice_len

        img_dict = self.imgs.__getitem__(idx_3d)
        
        cache_name = "./cache/"

        cache_name_input = cache_name + str(self.slice_direction) + "_" + str(index) + "_" + self.input_modality + ".npz"
        cache_name_target = cache_name + str(self.slice_direction) + "_" + str(index) + "_" + self.target_modality + ".npz"

        if os.path.exists(cache_name_input) and os.path.exists(cache_name_target): 
            cond_image = torch.from_numpy(np.load(cache_name_input)['data'])
            target_image = torch.from_numpy(np.load(cache_name_target)['data']) 
        else:
            print("else!!!")
            cond_image, target_image = img_dict["input"], img_dict["target"]
            # print("debug in get item input")
            # print(np.shape(cond_image), np.shape(target_image))
            # print difference between cond and target

            # print(np.shape(cond_image), np.shape(target_image))
            # print(torch.mean(torch.abs(cond_image - target_image)))
            if self.nearby_slice > 0:
                # get near by slices with idx_slice-nearby_slice:idx_slice+nearby_slice
                # pad zeros if needed
                if self.slice_direction == 3:
                    target_image = target_image[:, :, :, idx_slice]
                    idx_slice_start = max(0, idx_slice - self.nearby_slice)
                    idx_slice_end = min(self.slice_len, idx_slice + self.nearby_slice + 1)
                    cond_image = cond_image[:, :, :, idx_slice_start:idx_slice_end]

                    if idx_slice - self.nearby_slice < 0:
                        pad_len = self.nearby_slice - idx_slice
                        pad = torch.zeros_like(cond_image[:, :, :, 0:pad_len])
                        cond_image = torch.cat([pad, cond_image], axis = 3)
                    elif idx_slice + self.nearby_slice + 1 > self.slice_len:
                        pad_len = idx_slice + self.nearby_slice + 1 - self.slice_len
                        pad = torch.zeros_like(cond_image[:, :, :, 0:pad_len])
                        cond_image = torch.cat([cond_image, pad], axis = 3)
                    
                    # transpose slices into channels and make it 2D
                    cond_image = np.transpose(cond_image, (0, 3, 1, 2))
                    cond_image = np.reshape(cond_image, (cond_image.shape[0]*cond_image.shape[1], cond_image.shape[2], cond_image.shape[3]))

                elif self.slice_direction == 2:
                    target_image = target_image[:, :, idx_slice, :]
                    idx_slice_start = max(0, idx_slice - self.nearby_slice)
                    idx_slice_end = min(self.slice_len, idx_slice + self.nearby_slice + 1)
                    cond_image = cond_image[:, :, idx_slice_start:idx_slice_end, :]
                    
                    if idx_slice - self.nearby_slice < 0:
                        pad_len = self.nearby_slice - idx_slice
                        pad = torch.zeros_like(cond_image[:, :, 0:pad_len, :])
                        cond_image = torch.cat([pad, cond_image], axis = 2)
                    elif idx_slice + self.nearby_slice + 1 > self.slice_len:
                        pad_len = idx_slice + self.nearby_slice + 1 - self.slice_len
                        pad = torch.zeros_like(cond_image[:, :, 0:pad_len, :])
                        cond_image = torch.cat([cond_image, pad], axis = 2)
                        
                    cond_image = np.transpose(cond_image, (0, 2, 1, 3))
                    cond_image = np.reshape(cond_image, (cond_image.shape[0]*cond_image.shape[1], cond_image.shape[2], cond_image.shape[3]))
                        
                elif self.slice_direction == 1:
                    target_image = target_image[:, idx_slice, :, :]
                    idx_slice_start = max(0, idx_slice - self.nearby_slice)
                    idx_slice_end = min(self.slice_len, idx_slice + self.nearby_slice + 1)
                    cond_image = cond_image[:, idx_slice_start:idx_slice_end, :, :]
                    
                    if idx_slice - self.nearby_slice < 0:
                        pad_len = self.nearby_slice - idx_slice
                        pad = torch.zeros_like(cond_image[:, 0:pad_len, :, :])
                        cond_image = torch.cat([pad, cond_image], axis = 1)
                    elif idx_slice + self.nearby_slice + 1 > self.slice_len:
                        pad_len = idx_slice + self.nearby_slice - self.slice_len + 1
                        pad = torch.zeros_like(cond_image[:, 0:pad_len, :, :])
                        cond_image = torch.cat([cond_image, pad], axis = 1)
                    
                    cond_image = np.reshape(cond_image, (cond_image.shape[0]*cond_image.shape[1], cond_image.shape[2], cond_image.shape[3]))

                else:
                    print(self.slice_direction)
                    raise NotImplementedError("Slice direction out of range")
                
                # print("debug in get item")
                # print(np.shape(cond_image), np.shape(target_image), np.shape(mask))
                # debug dataset
                # plot cond_image with plt and save it to png

                # if cond_image.shape[0] != 25 or type(cond_image) != torch.Tensor:
                #     import pdb; pdb.set_trace()
                # print(cond_image.shape)
                if cond_image.shape[0] == 25:
                    # input condition is 1
                    channel_idx = [0,1,2,3,4] # data
                    channel_idx += [7,12,17] # pos emb
                    channel_idx += [22] # none zero mask 
                    cond_image = cond_image[channel_idx, :, :]
                elif cond_image.shape[0] != 25:
                    # print("debug in get item")
                    # import matplotlib.pyplot as plt
                    # for viz_i in range(cond_image.shape[0]):
                    #     plt.imshow(cond_image[viz_i, :, :])
                    #     plt.savefig("debug/cond_image_{}.png".format(viz_i))
                    # raise
                    image_channel_idx = cond_image.shape[0] - 20
                    channel_idx = list(range(image_channel_idx)) # data
                    channel_idx += [2+image_channel_idx,7+image_channel_idx,12+image_channel_idx] # pos emb
                    channel_idx += [17+image_channel_idx] # none zero mask 
                    # print(channel_idx) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 22, 27]
                    cond_image = cond_image[channel_idx, :, :]
                # print(cond_image.shape)

            else:
                if self.slice_direction == 3:
                    target_image = target_image[:, :, :, idx_slice]
                    cond_image = cond_image[:, :, :, idx_slice]
                elif self.slice_direction == 2:
                    target_image = target_image[:, :, idx_slice, :]
                    cond_image = cond_image[:, :, idx_slice, :]
                elif self.slice_direction == 1:
                    target_image = target_image[:, idx_slice, :, :]
                    cond_image = cond_image[:, idx_slice, :, :]
                else:
                    print(self.slice_direction)
                    raise NotImplementedError("Slice direction out of range")
            
            # save cache here
            os.makedirs(cache_name, exist_ok=True)                                                                     
            np.savez(cache_name_input, data=cond_image)                                                                
            np.savez(cache_name_target, data=target_image) 
        
        # use all one mask and make a dict
        mask = np.zeros_like(target_image) + 1
        
        ret['gt_image'] = target_image
        ret['cond_image'] = cond_image
        ret['mask_image'] = cond_image
        ret['mask'] = mask
        if self.train:
            ret['path'] = "training_set_" + str(index)
        else:
            ret['path'] = "val_set_" + str(index)

        return ret

    def __len__(self):
        return len(self.imgs) * self.slice_len

    def get_mask(self):
        raise NotImplementedError(f'Mask mode {self.mask_mode} has not been implemented since we are targeting SR and modality transfer.')
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        # elif self.mask_mode == 'irregular':
        #     mask = get_irregular_mask(self.image_size)
        # elif self.mask_mode == 'free_form':
        #     mask = brush_stroke_mask(self.image_size)
        # elif self.mask_mode == 'hybrid':
        #     regular_mask = bbox2mask(self.image_size, random_bbox())
        #     irregular_mask = brush_stroke_mask(self.image_size, )
        #     mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)



class BRATSDataset(Dataset):
    def __init__(self, root_dir, imgtype='t1', augmentation=True, return_file_name = True, slice = True):
        print("imagetype:", imgtype)
        self.augmentation = augmentation
        self.imgtype = imgtype
        self.root = root_dir
        self.dataset = self.get_dataset()
        self.return_file_name = return_file_name
        self.slice = slice

    def get_dataset(self):
        image_dirs = sorted([d for d in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, d))])
        # if self.train == True:
        #     image_dirs=image_dirs[:int(0.8*length)]
        # else:
        #     image_dirs=image_dirs[int(0.8*length):]
        return image_dirs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img_dir = self.dataset[index]
        if self.imgtype != 'all':
            im_types = [self.imgtype]
        else:
            im_types = ['t1','t2','t1ce','flair']

        B = np.zeros((len(im_types), 240, 240, 155))
        for i,im_type in enumerate(im_types):
            img_name = os.path.join(self.root, img_dir, '*'+im_type+'.nii.gz')
            img_name = glob.glob(img_name)
            if len(img_name) != 1:
                print("no file or more than 1 files", img_name)
                print(os.path.join(self.root, img_dir, '*'+im_type+'.nii.gz'))
                raise NotImplementedError
            img_name = img_name[0]
            img = nib.load(img_name)
            B[i, :, :, :] = img.get_data()
        
        img = B
        if self.augmentation: # and self.train:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 1)
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 3)

        img = 1.0*img
        img = exposure.rescale_intensity(img)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # img = 2*img-1
        img = img[:, 20:220, 20:220, 1:153]
        if self.slice:# sample a random 2D slice
            img = img[:, :, :, np.random.randint(0, 152)]
        img = torch.from_numpy(img).float()

        if self.return_file_name:
            return {'data': img, 'file': img_name}
        else:
            return {'data': img}


class InpaintDataset_old(data.Dataset):
    def __init__(self, data_root, imgtype='t1', filp_aug=True, mask_config={}, data_len=-1,image_size = [200,200]):
        imgs = BRATSDataset(data_root, imgtype=imgtype, augmentation=filp_aug,return_file_name = True)
        if data_len > 0:
            print("!! using only {} data".format(data_len))
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        img_dict = self.imgs.__getitem__(index)
        img, name = img_dict["data"], img_dict["file"]
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = name 
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        # elif self.mask_mode == 'irregular':
        #     mask = get_irregular_mask(self.image_size)
        # elif self.mask_mode == 'free_form':
        #     mask = brush_stroke_mask(self.image_size)
        # elif self.mask_mode == 'hybrid':
        #     regular_mask = bbox2mask(self.image_size, random_bbox())
        #     irregular_mask = brush_stroke_mask(self.image_size, )
        #     mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


# class UncroppingDataset(data.Dataset):
#     def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
#         imgs = make_dataset(data_root)
#         if data_len > 0:
#             self.imgs = imgs[:int(data_len)]
#         else:
#             self.imgs = imgs
#         self.tfs = transforms.Compose([
#                 transforms.Resize((image_size[0], image_size[1])),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
#         ])
#         self.loader = loader
#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = image_size

#     def __getitem__(self, index):
#         ret = {}
#         path = self.imgs[index]
#         img = self.tfs(self.loader(path))
#         mask = self.get_mask()
#         cond_image = img*(1. - mask) + mask*torch.randn_like(img)
#         mask_img = img*(1. - mask) + mask

#         ret['gt_image'] = img
#         ret['cond_image'] = cond_image
#         ret['mask_image'] = mask_img
#         ret['mask'] = mask
#         ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
#         return ret

#     def __len__(self):
#         return len(self.imgs)

#     def get_mask(self):
#         if self.mask_mode == 'manual':
#             mask = bbox2mask(self.image_size, self.mask_config['shape'])
#         elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
#             mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
#         elif self.mask_mode == 'hybrid':
#             if np.random.randint(0,2)<1:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
#             else:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(
#                 f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2,0,1)


# class ColorizationDataset(data.Dataset):
#     def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
#         self.data_root = data_root
#         flist = make_dataset(data_flist)
#         if data_len > 0:
#             self.flist = flist[:int(data_len)]
#         else:
#             self.flist = flist
#         self.tfs = transforms.Compose([
#                 transforms.Resize((image_size[0], image_size[1])),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
#         ])
#         self.loader = loader
#         self.image_size = image_size

#     def __getitem__(self, index):
#         ret = {}
#         file_name = str(self.flist[index]).zfill(5) + '.png'

#         img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
#         cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

#         ret['gt_image'] = img
#         ret['cond_image'] = cond_image
#         ret['path'] = file_name
#         return ret

#     def __len__(self):
#         return len(self.flist)

