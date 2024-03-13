import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms.functional as TF

num_classes = 21
ignore_label = 255
# root = '/media/b3-542/LIBRARY/Datasets/VOC'
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = mask.convert('P')
    new_mask.putpalette(palette)

    return new_mask


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find('bndbox/ymin').text)
        xmin = int(boxes.find('bndbox/xmin').text)
        ymax = int(boxes.find('bndbox/ymax').text)
        xmax = int(boxes.find('bndbox/xmax').text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        # img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        # mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        # data_list = [l.strip('\n') for l in open(os.path.join(
        #     root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        annotation_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'Annotations')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'),
                    os.path.join(annotation_path, it + '.xml'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        annotation_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'Annotations')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'),
                    os.path.join(annotation_path, it + '.xml'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


def make_dataset_detection(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'train.txt')).readlines()]
        for it in data_list:
            item = os.path.join(img_path, it + '.jpg')
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'val.txt')).readlines()]
        for it in data_list:
            item = os.path.join(img_path, it + '.jpg')
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Main', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items


class VOC(data.Dataset):
    def __init__(self, root, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.kmeans_labels_path = os.path.join(root, 'extra_data', f'voc2012_{mode}_32_labels.npy')
        self.kmeans_labels = np.load(self.kmeans_labels_path)

    def __getitem__(self, index):
        if self.mode == 'test':
            img_path, img_name, annot_path = self.imgs[index]
            img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img

        img_path, mask_path, annot_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # bbox = torch.tensor(read_content(annot_path)[0])
        # if self.mode == 'train':
        #     mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
        #     mask = Image.fromarray(np.array(mask, dtype=np.uint8))
        # else:
        #     mask = Image.open(mask_path)

        mask = colorize_mask(Image.open(mask_path))
        # mask = torch.tensor(np.array(mask, dtype=np.uint8))
        # mask = TF.pil_to_tensor(mask)
        label = int(self.kmeans_labels[index])

        if self.joint_transform is not None:
            img1, mask1 = self.joint_transform(img, mask)
            img2, mask2 = self.joint_transform(img, mask)
        else:
            img1, img2 = img, img
            mask1, mask2 = mask, mask

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.target_transform is not None:
                mask1 = self.target_transform(mask1)
                mask2 = self.target_transform(mask2)
            return img1, mask1, img2, mask2, label

    def __len__(self):
        return len(self.imgs)


class VOCDetection(data.Dataset):
    def __init__(self, root, mode, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset_detection(root, mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.fromarray(np.zeros((img.size[0], img.size[1])).astype('uint8'))

        if self.joint_transform is not None:
            img1, mask1 = self.joint_transform(img, mask)
            img2, mask2 = self.joint_transform(img, mask)
        else:
            img1, img2 = img, img
            mask1, mask2 = mask, mask

        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            if self.target_transform is not None:
                mask1 = self.target_transform(mask1)
                mask2 = self.target_transform(mask2)
            return img1, mask1, img2, mask2, 0

    def __len__(self):
        return len(self.imgs)

