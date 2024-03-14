import os
import argparse
from tqdm import tqdm

import utils_data
import wandb

import torch
from torch import optim

from torch.utils.data import DataLoader
import torch.nn.functional as F

import vision_transformer as vits
import utils
from guided_filter import myGuidedFilter2d
from bilateral_solver import get_bf_optimized_masks


import token_metric
from segmenter_model import DINOWrapper, segmenterNetRGB

parser = argparse.ArgumentParser('Training of object clustering with DINO')
parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
       for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")

# training parameters
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--num_segments", default=1, type=int, help="Number of image segments")
parser.add_argument("--mask_threshold", default=0.5, type=float, help="Masking loss threshold to maximize")
# dataset settings
parser.add_argument('--img_size', default=320, type=int, help='Image Size to resize')
parser.add_argument('--patch_size', default=16, type=int, help='Patch size to DINO')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--root_dir', default='../../datasets', type=str)
parser.add_argument('--dataset', default='duts',
                    choices=['imagenet', 'imagenet100', 'ms-coco', 'coco-stuff', 'coco-stuff27',
                             'voc2012', 'voc2012det', 'duts', 'celeba', 'dut-omron', 'ecssd'
                             ]
                    )
# loss settings
parser.add_argument('--weight_entropy', default=1, type=float, help='Weight of the entropy loss')
parser.add_argument('--weight_reg_img', default=1, type=float, help='Weight of the img reg loss')
parser.add_argument('--weight_reg_feat', default=1, type=float, help='Weight of the feature reg loss')
parser.add_argument('--weight_const', default=0.01, type=float, help='Weight of the feature pca reg loss')
parser.add_argument('--weight_mask_s', default=0.01, type=float, help='Weight of the feature pca reg loss')
parser.add_argument('--weight_mask', default=1, type=float, help='Weight of the mask loss')
# DINO settings
parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
parser.add_argument('--ckpt_path', default='./pretrained_models/dino_deitsmall16_pretrain.pth',
                    type=str, help='Path to pretrained DINO checkpoint')
parser.add_argument('--ckpt_segmenter_path', default='', type=str, help='Path to pretrained segmentation head path')
# experiment settings
parser.add_argument('--experiment_name', default='dino_object_segment_t1', type=str)
parser.add_argument('--save_mask', action='store_true', help='Save the estimated masks')
parser.add_argument('--print_frequency', default=100, type=int, help='Frequency of loss prints')
parser.add_argument('--image_frequency', default=1000, type=int, help='Frequency of image and mask saves')
parser.add_argument('--save_frequency', default=1, type=int, help='Frequency of epoch to save state dict')
parser.add_argument('--disable_pos_embedding', default=True, type=bool, help='Disable positional embedding')
parser.add_argument('--wandb_off', action='store_true', help='Close wandb')
parser.add_argument('--evaluation_off', action='store_true', help='Do not evaluate at the end of each epoch')
parser.add_argument('--from_begin', action='store_true', help='Get DINO features from beginning')
parser.add_argument('--use_keys', action='store_true', help='Use keys from DINO instead of features')
parser.add_argument('--get_cls_token', action='store_true', help='Return the cls token from DINO')
parser.add_argument('--return_attention', action='store_true', help='Return the attention from DINO')
parser.add_argument('--just_evaluate', action='store_true', help='Just do the evaluation')
parser.add_argument('--use_bf', action='store_true', help='use bilateral filtering in the evaluation')
parser.add_argument('--use_gf', action='store_true', help='use guided filtering in the evaluation')
parser.add_argument('--use_fmax', action='store_true', help='use Fmax metric instead of Fbeta')
parser.add_argument('--use_multi_scale', action='store_true', help='use multiscale consistency loss')

# graph settings
parser.add_argument('--ncut_thr', default=0.4, type=float, help='Threshold for weight matrix threshold')
parser.add_argument('--ncut_reg_thr', default=0, type=float, help='Threshold for weight matrix threshold')
parser.add_argument('--epsilon', default=1e-6, type=float, help='Epsilon for weight matrix')
parser.add_argument('--feat_upscale', default=1, type=int, help='Amount of upscaling of features')

args = parser.parse_args()

args.num_patches = args.img_size // args.patch_size
args.lr_inpaint = float(args.lr)
args.h_featmap = args.img_size // args.patch_size
args.w_featmap = args.img_size // args.patch_size

save_folder = os.path.join('outputs', args.experiment_name)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

train_dataset, val_dataset = utils_data.load_my_dataset_not_augment(args)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
print(f'Train Dataset length is {len(train_dataset)}')
print(f'Val Dataset length is {len(val_dataset)}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# take random validation set to evaluate
img_val, label_val = next(iter(val_loader))
img_val, label_val = img_val.to(device), label_val.to(device)

reg_loss = utils.sempartRegLoss(3).to(device)
reg_feat_loss = utils.sempartFeatRegLoss(ch=384*args.n_last_blocks, threshold=args.ncut_thr).to(device)

my_loss_wrapper = utils.myLossWrapper(args)

def segment_from_feature(segment_net, guided_filter, feat_model, img, args, iters, mode='train'):
    if mode == 'eval':
        with torch.no_grad():
            feature = feat_model(img)
            segment_pred, _ = segment_net(feature, img)
            segment_large = torch.sigmoid(segment_pred / 1)

        return segment_large

    elif mode == 'pred':
        feature = feat_model(img)
        segment_pred, _ = segment_net(feature, img)
        segment_large = torch.sigmoid(segment_pred / 1)

        return segment_large

    else:
        feature = feat_model(img)
        segment_pred, segment_pred_small = segment_net(feature, img)
        segment_large = torch.sigmoid(segment_pred / 1)
        segment_small = torch.sigmoid(segment_pred_small / 1)
        segment_pred_soft = F.interpolate(segment_large, scale_factor=1/args.patch_size, mode='bilinear')

        loss_dict = {}
        loss_dict['mask_loss'] = utils.get_ncut_loss(feature, segment_small, args.ncut_thr, args.epsilon, iters)

        if args.use_multi_scale:
            b, d, h, w = feature.shape
            h_max_att = torch.randint(args.h_featmap // 4, 3 * (args.h_featmap // 4), (b,), device=img.device)
            w_max_att = torch.randint(args.w_featmap // 4, 3 * (args.w_featmap // 4), (b,), device=img.device)

            img_s, mask_s = utils_data.crop_my_images(img, segment_large, h_max_att, w_max_att, args.patch_size, 2)
            feature_s = feat_model(img_s)
            segment_pred_s, segment_pred_small_s = segment_net(feature_s, img_s)
            segment_large_s = torch.sigmoid(segment_pred_s / 1)
            segment_small_s = torch.sigmoid(segment_pred_small_s / 1)
            segment_large_s_gf = guided_filter(segment_large_s, img_s).clamp(min=args.epsilon, max=1 - args.epsilon).detach()

            loss_dict['mask_loss_s'] = utils.get_ncut_loss(feature_s, segment_small_s, args.ncut_thr, args.epsilon, iters)
            loss_dict['loss_consistency'] = (mask_s - segment_large_s_gf).pow(2).mean()

        loss_dict['mask_loss_img'] = reg_loss(img, segment_large)
        loss_dict['mask_loss_feat'] = reg_feat_loss(feature, segment_small)

        loss_dict['loss_entropy'] = (segment_small - segment_pred_soft).pow(2).mean()

        return loss_dict, segment_large


model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=1000)
model = utils.load_pretrained_weights(model, args.ckpt_path, None, args.arch, args.patch_size)

model = DINOWrapper(model, return_attention=args.return_attention, args=args).to(device)
args.z_chunk_size = model.model_dino.embed_dim * args.n_last_blocks

if args.wandb_off:
    os.environ['WANDB_MODE'] = 'offline'
# TODO: do not forget to remove entity
wandb.init(project='dino', name=args.experiment_name, config=args, entity='aesari')

segmenter = segmenterNetRGB(args.z_chunk_size, args.num_segments, 64, patch_size=args.patch_size)

if args.ckpt_segmenter_path != '':
    segmenter_state_dict = torch.load(args.ckpt_segmenter_path, map_location='cpu')['segmenter_state_dict']
    segmenter.load_state_dict(segmenter_state_dict)
    print('Pretrained SEGMENTER weights are loaded')

segmenter = segmenter.to(device)

guided_filter = myGuidedFilter2d(9, 1e-3).to(device)
optimizer = optim.AdamW(segmenter.parameters(), args.lr, betas=(0.9, 0.99), weight_decay=0)

dl_len = len(train_loader)
max_miou = 0
iters = 0
for epoch in range(args.epochs):
    if not args.just_evaluate:
        pbar = tqdm(enumerate(train_loader), total=dl_len)
        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)

            ## Train Segmenter Network
            optimizer.zero_grad()
            loss_dict, segment_large = segment_from_feature(segmenter, guided_filter, model, img, args, iters, mode='train')

            loss = my_loss_wrapper.get_loss_from_dict(loss_dict)

            loss.backward()
            optimizer.step()


            if (i % args.print_frequency) == 0:
                pbar.set_description(my_loss_wrapper.get_loss_print_str(loss_dict, epoch, args.epochs, i, dl_len))
                wandb.log(my_loss_wrapper.get_metric_log_dict(loss_dict), step=iters)

            if args.save_mask and (i % args.image_frequency) == 0:
                with torch.no_grad():
                    segment_large_val = segment_from_feature(segmenter, guided_filter, model, img_val, args, iters, mode='eval')

                utils.contruct_mask_and_log(segment_large_val, img_val, label_val, args, iters, phase='Val')
                utils.contruct_mask_and_log(segment_large, img, label, args, iters, phase='Train')

            iters += 1

    if (args.dataset in ['duts', 'dut-omron', 'ecssd']) and not args.evaluation_off:
        print(f'Evaluating the segmentation at epoch {epoch}')
        gt_list = []
        pred_list = []
        for (img, label) in tqdm(val_loader):
            img, label = img.to(device), label.to(device)

            with torch.no_grad():
                segment_large = segment_from_feature(segmenter, guided_filter, model, img, args, iters, mode='eval')
                if segment_large.mean() > 0.5:
                    segment_large = 1 - segment_large

                if args.use_gf:
                    segment_large = guided_filter(segment_large, img)

                if args.use_bf:
                    segment_large = get_bf_optimized_masks(img, segment_large)


            pred_list.append(segment_large.detach().cpu())
            gt_list.append(label.detach().cpu())

        pred_list = torch.cat(pred_list, 0)
        gt_list = torch.cat(gt_list, 0)
        pix_acc = token_metric.accuracy(pred_list, gt_list)
        miou = token_metric.IoU(pred_list, gt_list)
        if args.use_fmax:
            f_name = 'Fmax'
            fscore = token_metric.my_Fmax(pred_list, gt_list)
        else:
            f_name = 'F(not max)'
            precision, recall = token_metric.precision_recall(gt_list, pred_list)
            fscore = token_metric.F_max(precision, recall)
        print(f"Pix Acc: {pix_acc}, mIoU: {miou}, {f_name}: {fscore}")
        wandb.log({'Pix Acc': pix_acc, 'mIoU': miou, f_name: fscore}, step=iters)

    else:
        miou = -1

    if (epoch % args.save_frequency == 0 or miou > max_miou) and not args.just_evaluate:
        if miou > max_miou:
            max_miou = miou
            save_path = os.path.join(save_folder, f'dino_object_cluster_{epoch:04d}_best.pth')
        else:
            save_path = os.path.join(save_folder, f'dino_object_cluster_{epoch:04d}.pth')

        save_dict = {
            'segmenter_state_dict': segmenter.state_dict()
        }
        torch.save(save_dict, save_path)
        print(f'State dicts are saved at epoch {epoch}')

