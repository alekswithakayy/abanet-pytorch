import skimage
import os
import numpy as np

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


class MakeCAMImageFolder(ImageFolder):

    def __init__(self, args):
        super(MakeCAMImageFolder, self).__init__(args.dataset_dir)
        self.image_size = args.image_size
        self.scales = [1., 0.5, 1.5]

    def __getitem__(self, index):
        path, target = self.samples[index]
        name, _ = os.path.splitext(os.path.basename(path))
        image = self.loader(path)
        image_list = []
        for s in self.scales:
            height, width = self.image_size
            size = (int(np.round(height*s)), int(np.round(width*s)))
            s_image = resize(np.array(image), size, order=3)
            s_image = transforms.ToTensor()(s_image)
            s_image = s_image.contiguous()
            image_list.append(s_image)
        return image_list, target, name


class MakeIRLabelImageFolder(ImageFolder):

    def __init__(self, args):
        super(MakeIRLabelImageFolder, self).__init__(args.dataset_dir)
        self.image_size = args.image_size
        self.cam_save_dir = args.cam_save_dir

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        name, _ = os.path.splitext(os.path.basename(image_path))
        image = self.loader(image_path)
        image = resize(np.array(image),  self.image_size, order=3)
        cam_path = os.path.join(self.cam_save_dir, name + '.npy')
        cam = np.load(cam_path, allow_pickle=True)
        return image, cam, target, name


class TrainIRNImageFolder(ImageFolder):

    def __init__(self, src_indices, dst_indices,  args):
        dataset_root = args.dataset_dir
        super(TrainIRNImageFolder, self).__init__(dataset_root)
        self.ir_label_save_dir = args.ir_label_save_dir
        self.src_indices = src_indices
        self.dst_indices = dst_indices
        self.transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor()])
        self.label_size = (int(args.image_size[0]*0.25), int(args.image_size[1]*0.25))
        self.num_classes = args.num_classes

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        name, _ = os.path.splitext(os.path.basename(image_path))
        label_path = os.path.join(self.ir_label_save_dir, name + '.png')

        image = self.loader(image_path)
        image = self.transform(image)
        label = self.loader(label_path)
        label = resize(np.array(label), self.label_size, order=1)

        bg_pos, fg_pos, neg = self.extract_affinity_label(label)

        return image, bg_pos, fg_pos, neg


    def extract_affinity_label(self, label_map):
        segm_map_flat = np.reshape(label_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.src_indices], axis=0)
        segm_label_to = segm_map_flat[self.dst_indices]

        valid_label = np.logical_and(np.less(segm_label_from, self.num_classes + 1),
                                     np.less(segm_label_to, self.num_classes + 1))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


def resize(image, size, order=1, anti_aliasing=True):
    image = skimage.transform.resize(image, size, order=order,
        anti_aliasing=anti_aliasing)
    return image
