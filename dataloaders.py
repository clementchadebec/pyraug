import torch.utils.data as data
import pickle as pkl
import os
import torch
import random
import numpy as np
from glob import glob

class CBCTData(data.Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        assert phase in ['train', 'val', 'test'], "phase cannot recongnise..."
        self.config = config
        self.data_path = self.config.data_path
        self.data_pairs = self.__get_data_pairs__()
        self.PC = config.patient_cohort
        assert self.PC in ['inter', 'intra'], f"patient_cohort should be intra/inter, cannot be {self.PC}"

    def __getitem__(self, index):
        flag = (self.phase=='train') and (self.rand_prob()) and (self.config.patient_cohort=='inter')
        if flag:
            fx_img_path, _, fx_seg_path, _ = self.data_pairs[index]
            _, mv_img_path, _, mv_seg_path = self.__get_inter_pairs__(index)
        else:
            fx_img_path, mv_img_path, fx_seg_path, mv_seg_path = self.data_pairs[index]

        moving_image, moving_label = np.load(mv_img_path), np.load(mv_seg_path)
        fixed_image, fixed_label = np.load(fx_img_path), np.load(fx_seg_path)

        if self.phase in ['train'] and self.config.two_stage_sampling:
            random_label_index = random.randint(0, moving_label.shape[0]-1)
            moving_label, fixed_label = moving_label[random_label_index], fixed_label[random_label_index]
        else: pass

        if self.phase in ['train'] and self.config.crop_on_seg_aug and self.rand_prob():
            moving_label = self.random_crop_aug(moving_label)
            
        data_dict = {
            'mv_img': torch.FloatTensor(moving_image[None, ...]), 
            'mv_seg': torch.FloatTensor(moving_label[None, ...]), 
            'fx_img': torch.FloatTensor(fixed_image[None, ...]), 
            'fx_seg': torch.FloatTensor(fixed_label[None, ...]),
            'subject': os.path.basename(fx_img_path),
            'subject_mv': os.path.basename(mv_img_path),
            }

        return data_dict

    def __len__(self):
        return len(self.data_pairs)

    def __get_data_pairs__(self):
        '''split train val test data'''
        pid_lists = os.listdir(os.path.join(self.data_path, 'fixed_images'))
        pid_lists.sort()

        tmp = []
        for i in pid_lists:
            tmp.append([
                os.path.join(self.data_path, 'fixed_images', i),
                os.path.join(self.data_path, 'moving_images', i),
                os.path.join(self.data_path, 'fixed_labels', i),
                os.path.join(self.data_path, 'moving_labels', i),
            ])
        
        test_pairs = tmp[len(pid_lists)//8 * self.config.cv : len(pid_lists)//8 * (self.config.cv+1)]

        if self.config.cv == 7:
            val_pairs = tmp[:len(pid_lists)//8]
        else:
            val_pairs = tmp[len(pid_lists)//8 * (self.config.cv+1) : len(pid_lists)//8 * (self.config.cv+2)]
        
        train_pairs = [i for i in tmp if i not in val_pairs and i not in test_pairs]

        data_dict = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
        
        return data_dict[self.phase]

    def __get_inter_pairs__(self, index):
        indices = list(range(self.__len__()))
        del indices[index]
        return self.data_pairs[random.sample(indices, 1)[0]]

    @staticmethod
    def rand_prob(p=0.5):
        assert 0<=p<=1, "p should be a number in [0, 1]"
        return random.random() < p

    def random_crop_aug(self, seg_arr):
        '''A data augmentation method for conditional segmentation'''
        px, py, pz = np.where(seg_arr==1)
        grid_points = [i for i in zip(px, py, pz)]
        cx, cy, cz = random.sample(grid_points, 1)[0]  # select a point as center

        r_min, r_max = self.config.crop_on_seg_rad
        shape_x, shape_y, shape_z = self.config.input_shape

        rad = random.randint(r_min, r_max)
        Lx, Rx = max(0, cx-rad), min(shape_x, cx+rad)  # Left & Right x
        Ly, Ry = max(0, cy-rad), min(shape_y, cy+rad)  # Left & Right y
        Lz, Rz = max(0, cz-rad), min(shape_z, cz+rad)  # Left & Right z

        seg_arr[..., Lx:Rx, Ly:Ry, Lz:Rz] = 0
        return seg_arr
        