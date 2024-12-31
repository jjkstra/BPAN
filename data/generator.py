import torch
from torch.utils.data import Dataset

import os
import os.path as osp
from PIL import Image
from random import random
from data.transform import transform_manager
from utils import get_grid_location


class DatasetGenerator(Dataset):
    def __init__(self, mode, args):
        data_path = osp.join(args.data_root, args.dataset)

        if args.dataset in ['mini_imagenet', 'cub']:
            if args.dataset == 'mini_imagenet':
                IMAGE_PATH = osp.join(data_path, 'images')
            else:
                IMAGE_PATH = data_path

            SPLIT_PATH = osp.join(data_path, 'split')
            csv_path = osp.join(SPLIT_PATH, mode + '.csv')
            lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

            if args.dataset == "cub" and mode == 'train':
                lines.pop(5864)

            data = []
            wnids = []
            label = []
            label_idx = -1

            for line in lines:
                context = line.split(',')
                filename = context[0]
                wnid = context[1]
                path = osp.join(IMAGE_PATH, filename)
                if wnid not in wnids:
                    wnids.append(wnid)
                    label_idx += 1
                data.append(path)
                label.append(label_idx)

        elif args.dataset in ['tiered_imagenet', 'cifar_fs', 'fc100']:
            if args.dataset == 'cifar_fs':
                IMAGE_PATH = osp.join(data_path, '-'.join(['meta', mode]))
            else:
                IMAGE_PATH = osp.join(data_path, mode)

            data = []
            label = []
            folders = [osp.join(IMAGE_PATH, label) for label in os.listdir(IMAGE_PATH) if
                       osp.isdir(osp.join(IMAGE_PATH, label))]

            for idx in range(len(folders)):
                this_folder = folders[idx]
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(idx)

        else:
            raise ValueError('Unknown Dataset')

        self.mode = mode
        self.data = data
        self.label = label
        self.args = args
        self.transform = transform_manager(mode, args)

    def get_labels(self):
        return self.label

    def get_grid_patches(self, image):
        if self.mode == "train":
            grid_ratio = 1.0 + random()
        elif self.mode == 'val' or self.mode == 'test':
            grid_ratio = self.args.patch_ratio
        else:
            raise ValueError('Unknown set')

        w, h = image.size
        grid_locations_w = get_grid_location(w, grid_ratio, self.args.grid_size)
        grid_locations_h = get_grid_location(h, grid_ratio, self.args.grid_size)

        patches_list = []
        for i in range(self.args.grid_size):
            for j in range(self.args.grid_size):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = image.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform(patch)
                patches_list.append(patch)

        return torch.stack(patches_list)

    def get_random_patches(self, image):
        patch_list = []
        for _ in range(self.args.n_patch_views):
            patch_list.append(self.transform(image))
        return torch.stack(patch_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = Image.open(path).convert('RGB')
        if self.args.crop_mode == "none":
            return self.transform(image), label
        elif self.args.crop_mode == "random":
            return self.get_random_patches(image), label
        elif self.args.crop_mode == "grid":
            patches = self.get_grid_patches(image)
            patch_list = torch.cat([self.transform(image).unsqueeze(0), patches], dim=0)
            return patch_list, label
            # return self.get_grid_patches(image), label
        else:
            raise ValueError('Error mode')
