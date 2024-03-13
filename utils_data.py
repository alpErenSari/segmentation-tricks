import glob
import os
import numpy as np
import torch
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.transforms.functional as TF
import utils
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from functools import partial
import voc_custom
import random


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)

    def __call__(self, image):
        b, ch, h, w = image.size()
        image2 = torch.clone(image)
        image2 = self.std.to(image.device) * image2 + self.mean.to(image.device)
        return image2


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).reshape(1, 3, 1, 1)

    def __call__(self, image):
        b, ch, h, w = image.size()
        image2 = torch.clone(image)
        image2 = (image2 - self.mean.to(image.device)) / self.std.to(image.device)
        return image2


class myRandomCropResize(object):
    def __init__(self, img_size, scale=(0.3, 1.0), ratio=(0.95, 1.05)):
        self.random_resized_crop = transforms.RandomResizedCrop(img_size, scale=scale, ratio=ratio)
        self.img_size = 2 * [img_size]
        self.scale = scale
        self.ratio = ratio
        self.interpolation = TF.InterpolationMode.BILINEAR

    def __call__(self, img, mask):
        batch_size = img.shape[0]
        assert batch_size == mask.shape[0], "batch size for images and masks should be the same"
        img_list, mask_list = [], []
        for b_ in range(batch_size):
            img_cur, mask_cur = img[b_].unsqueeze(0), mask[b_].unsqueeze(0)
            i, j, h, w = self.random_resized_crop.get_params(img_cur, self.scale, self.ratio)
            img_cur = TF.resized_crop(img_cur, i, j, h, w, self.img_size, self.interpolation)
            mask_cur = TF.resized_crop(mask_cur, i, j, h, w, self.img_size, self.interpolation)

            img_list.append(img_cur)
            mask_list.append(mask_cur)

        img_list, mask_list = torch.cat(img_list, 0), torch.cat(mask_list, 0)

        return img_list, mask_list

class VOCWrap(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0], 0


def crop_my_images(imgs, masks, crop_i, crop_j, patch_size, feat_upscale, eps=1e-6):
    h = imgs.shape[2] // 2
    w = imgs.shape[3] // 2
    crop_i = crop_i.cpu().tolist() * patch_size * feat_upscale
    crop_j = crop_j.cpu().tolist() * patch_size * feat_upscale

    img_s_list = []
    masks_s_list = []
    for img, mask, i, j in zip(imgs, masks, crop_i, crop_j):
        if i < h // 2: i = h // 2
        elif i > 3 * (h // 2): i = 3 * (h // 2)
        img_s = TF.crop(img, i, j, h, w)
        img_s_list.append(img_s)
        mask_s = TF.crop(mask, i, j, h, w)
        masks_s_list.append(mask_s)

    img_s_list = torch.stack(img_s_list, 0)
    masks_s_list = torch.stack(masks_s_list, 0)
    # masks_s_list = F.interpolate(masks_s_list, scale_factor=2, mode='bilinear').clamp(min=eps, max=1 - eps)
    if feat_upscale > 1:
        img_s_list = F.interpolate(img_s_list, scale_factor=feat_upscale, mode='bilinear')
        masks_s_list = F.interpolate(masks_s_list, scale_factor=feat_upscale, mode='bilinear')

    return img_s_list, masks_s_list



class myDUTS(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None, target_transform=None, dataset_suffix='TR', img_size=224, not_resize=False, center_crop=False):
        self.root = root
        self.dataset_suffix = dataset_suffix
        # dataset_suffix = 'TE'
        self.image_root = os.path.join(root, f'DUTS/DUTS-{dataset_suffix}/DUTS-{dataset_suffix}-Image')
        self.mask_root = os.path.join(root, f'DUTS/DUTS-{dataset_suffix}/DUTS-{dataset_suffix}-Mask')
        self.image_path_list = sorted(os.listdir(self.image_root))
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = [img_size, img_size]
        self.not_resize = not_resize
        self.center_crop = center_crop
        self.interpolation = TF.InterpolationMode.BILINEAR
        self.scale = (0.2, 1.0)
        self.ratio= (0.95, 1.05)
        self.random_resized_crop = transforms.RandomResizedCrop(self.img_size, scale=self.scale, ratio=self.ratio)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_name = os.path.splitext(self.image_path_list[idx])[0]
        img_path = os.path.join(self.image_root, self.image_path_list[idx])
        mask_path = os.path.join(self.mask_root, img_name + '.png')
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img = self.transform(img)
        mask = self.target_transform(mask)

        # Random crop
        if self.not_resize:
            return img, mask

        elif self.dataset_suffix == 'TE' or self.center_crop:
            img = TF.center_crop(img, self.img_size)
            mask = TF.center_crop(mask, self.img_size)

        elif self.dataset_suffix == 'TR':
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=self.img_size)

            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # i, j, h, w = self.random_resized_crop.get_params(img, self.scale, self.ratio)
            #
            # img = TF.resized_crop(img, i, j, h, w, self.img_size, self.interpolation)
            # mask = TF.resized_crop(mask, i, j, h, w, self.img_size, self.interpolation)

            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)

        else:
            img = TF.center_crop(img, self.img_size)
            mask = TF.center_crop(mask, self.img_size)

        return img, mask

class mySaliencyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None, target_transform=None, img_path='', mask_path=''):
        self.root = root
        self.image_root = os.path.join(root, img_path)
        self.mask_root = os.path.join(root, mask_path)
        self.image_path_list = sorted(os.listdir(self.image_root))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img_name = os.path.splitext(self.image_path_list[idx])[0]
        img_path = os.path.join(self.image_root, self.image_path_list[idx])
        mask_path = os.path.join(self.mask_root, img_name + '.png')
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img = self.transform(img)
        mask = self.target_transform(mask)

        return img, mask


class myCelebA(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transform=None):
        self.root = root
        self.img_root = os.path.join(root, 'CelebAMask-HQ/CelebA-HQ-img/data')
        self.img_path_list = glob.glob(os.path.join(self.img_root, '*'))
        # self.kmeans_pred = np.load(os.path.join(root, 'CelebAMask-HQ/CelebA-HQ-img/kmeans_pred.npy'), allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        mask = img

        return img, mask


def load_my_dataset_not_augment(args):
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet':
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        dataset_name = 'ImageNet100' if args.dataset == 'imagenet100' else 'ImageNet'

        dataset_root_train = os.path.join(args.root_dir, dataset_name, 'train')
        train_dataset = datasets.ImageFolder(dataset_root_train, transform_train)
        dataset_root_val = os.path.join(args.root_dir, dataset_name, 'val')
        val_dataset = datasets.ImageFolder(dataset_root_val, transform_val)

        return train_dataset, val_dataset

    elif args.dataset == 'imagenet-c':
        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        imagenet_c_dict = {
            'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
            'digital': ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate'],
            'extra': ['gaussian_blur', 'saturate', 'spatter', 'speckle_noise'],
            'noise': ['gaussian_noise', 'impulse_noise', 'shot_noise'],
            'weather': ['brightness', 'fog', 'frost', 'snow']
        }
        distortion_type_split = args.distortion_type.split(',')
        distortion_level = int(distortion_type_split[2])
        cur_key = distortion_type_split[0]
        cur_key_el = imagenet_c_dict[cur_key][int(distortion_type_split[1])-1]
        dataset_root_val = os.path.join(args.root_dir, 'ImageNet-C', cur_key, cur_key_el, str(distortion_level))
        val_dataset = datasets.ImageFolder(dataset_root_val, transform_val)
        print(f'ImageNet-C loaded with keys {cur_key}, {cur_key_el} and distortion level {distortion_level}')

        return val_dataset, val_dataset

    elif args.dataset == 'voc2012det':
        # since pascal voc 2021 segmentation dataset has 21 classes
        cat_ids = list(range(21))
        transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        dataset_root = os.path.join(args.root_dir, 'pascal_voc_2012')
        train_dataset = VOCWrap(voc_custom.VOCDetection(dataset_root, 'train', transform=transform_train, target_transform=None))
        val_dataset = VOCWrap(voc_custom.VOCDetection(dataset_root, 'val', transform=transform_val, target_transform=None))

        return train_dataset, val_dataset

    elif args.dataset == 'duts':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                transforms.Compose([
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]),
                # transforms.CenterCrop(args.img_size),
                # transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_val = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                # transforms.CenterCrop(args.img_size),
                # transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_target = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                # transforms.CenterCrop(args.img_size),
                transforms.ToTensor()
            ])

        # dataset_root_train = os.path.join(args.root_dir, 'DUTS/DUTS-TR')
        train_dataset = myDUTS(args.root_dir, transform_train, transform_target, dataset_suffix='TR', img_size=args.img_size, center_crop=False)
        # dataset_root_val = os.path.join(args.root_dir, 'DUTS/DUTS-TE')
        val_dataset = myDUTS(args.root_dir, transform_val, transform_target, dataset_suffix='TE', img_size=args.img_size, not_resize=False)

        return train_dataset, val_dataset

    elif args.dataset == 'dut-omron':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                transforms.CenterCrop(args.img_size),
                # transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_target = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor()
            ])

        img_path = 'DUT-OMRON/DUT-OMRON-image/DUT-OMRON-image'
        mask_path = 'DUT-OMRON/DUT-OMRON-gt-pixelwise.zip/pixelwiseGT-new-PNG'
        train_dataset = mySaliencyDataset(args.root_dir, transform_train, transform_target, img_path=img_path,
                                          mask_path=mask_path)
        val_dataset = mySaliencyDataset(args.root_dir, transform_train, transform_target, img_path=img_path,
                                        mask_path=mask_path)

        return train_dataset, val_dataset

    elif args.dataset == 'ecssd':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                transforms.CenterCrop(args.img_size),
                # transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_target = transforms.Compose(
            [
                transforms.Resize(args.img_size+15),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor()
            ])

        img_path = 'ECSSD/images/'
        mask_path = 'ECSSD/ground_truth_mask/'
        train_dataset = mySaliencyDataset(args.root_dir, transform_train, transform_target, img_path=img_path,
                                          mask_path=mask_path)
        val_dataset = mySaliencyDataset(args.root_dir, transform_train, transform_target, img_path=img_path,
                                        mask_path=mask_path)

        return train_dataset, val_dataset

    elif args.dataset == 'coco-stuff27':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                transforms.CenterCrop(args.img_size),
                # transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_val = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_target = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                transforms.CenterCrop(args.img_size)
            ])

        dataset_root_train = os.path.join(args.root_dir, 'COCO17/train2017')
        dataset_root_val = os.path.join(args.root_dir, 'COCO17/val2017')

        train_annotations_path = os.path.join(args.root_dir, 'COCO17/annotations/stuff_train2017.json')
        valid_annotations_path = os.path.join(args.root_dir, 'COCO17/annotations/stuff_val2017.json')
        print('COCO-Stuff annotations loaded')

        train_annotations = COCO(train_annotations_path)
        valid_annotations = COCO(valid_annotations_path)

        cat_ids = train_annotations.getCatIds()
        train_img_ids = []
        for cat in cat_ids:
            train_img_ids.extend(train_annotations.getImgIds(catIds=cat))

        train_img_ids = list(set(train_img_ids))
        print(f"Number of training images: {len(train_img_ids)}")

        valid_img_ids = []
        for cat in cat_ids:
            valid_img_ids.extend(valid_annotations.getImgIds(catIds=cat))

        valid_img_ids = list(set(valid_img_ids))
        print(f"Number of validation images: {len(valid_img_ids)}")

        train_dataset = utils.ImageDataCOCO(train_annotations, train_img_ids, cat_ids, dataset_root_train,
                                            transform_train, transform_target, use_coarse_label=True)
        val_dataset = utils.ImageDataCOCO(valid_annotations, valid_img_ids, cat_ids, dataset_root_val,
                                          transform_val, transform_target, use_coarse_label=True)
        return train_dataset, val_dataset


    elif args.dataset == 'imagenet100_subset':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                # transforms.CenterCrop(args.img_size),
                transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_val = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        transform_target = transforms.Compose(
            [
                transforms.Resize(args.img_size + 15),
                transforms.ToTensor()
            ])

        dataset_root_train = os.path.join(args.root_dir, 'ImageNet100/train')
        train_dataset = datasets.ImageFolder(dataset_root_train, transform_train)
        val_dataset = myDUTS(args.root_dir, transform_val, transform_target, dataset_suffix='TE', img_size=args.img_size, not_resize=args.not_resize)

        return train_dataset, val_dataset

    elif args.dataset == 'celeba':
        transform_train = transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        train_dataset = myCelebA(args.root_dir, transform_train)
        val_dataset = myCelebA(args.root_dir, transform_train)

        return train_dataset, val_dataset

    else:
        raise('Only ImageNet is implemented')