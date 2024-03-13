import random

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torch import optim

from tqdm import tqdm

import utils
import utils_metric
import wandb
from segmenter_model import featureNet
from vit_pytorch_dis.vit import my_gradient_reg_func


def compute_segmentation_performance(val_loader, feat_model, segmenter, epoch, args, pred_func, device):
    print(f'Computing segmentation performance on epoch {epoch}')
    if args.dataset == 'coco-stuff27':
        n_classes = 27
        unsupervised_metric = utils_metric.UnsupervisedMetrics('', n_classes, -n_classes + args.num_segments, True, True).to(
            device)
    elif args.dataset.startswith('voc2012'):
        n_classes = 20
        args.num_segments = max(20, args.num_segments)
        unsupervised_metric = utils_metric.UnsupervisedMetrics(
            '', n_classes, -n_classes + args.num_segments,
            True, False, match_each_image=args.match_each_image
        ).to(device)
    else:
        n_classes = 91
        unsupervised_metric = utils_metric.UnsupervisedMetrics('', n_classes, -n_classes + args.num_segments, True, False).to(
            device)

    for data_batch in tqdm(val_loader):
        img = data_batch[0].to(device)
        gt_mask = data_batch[1].to(device).long()
        k_label = data_batch[4].to(device)
        # batch_size = gt_mask.size(0)
        # gt_mask_unique = []
        # for gt_mask_ in gt_mask:
        #     cur_values = torch.unique(gt_mask_)
        #     if cur_values.size(0) > 1:
        #         gt_mask_unique.append(cur_values[1])
        #     else:
        #         gt_mask_unique.append(cur_values[0])
        # gt_mask_unique = torch.stack(gt_mask_unique, 0)

        if args.dataset == 'voc2012':
            # eliminate background class to compute object segmentation performance
            gt_mask += -1

        _, segment_large, _, _ = pred_func(feat_model, segmenter, img, args)
        # segment_pred_soft = (segment_large.argmax(1) * k_label.reshape(-1, 1, 1)).long()
        segment_pred_soft = segment_large.argmax(1).long()
        unsupervised_metric.update(segment_pred_soft, gt_mask)

    return unsupervised_metric.compute()


def get_dino_output_tokens(model, img, args):

    with torch.no_grad():
        intermediate_output = model.get_intermediate_layers(img, args.n_last_blocks)[-2]
        b, n, d = intermediate_output.size()
        intermediate_output = intermediate_output[:, 1:, :].\
            reshape(b, args.h_featmap, args.w_featmap, d).permute(0, 3, 1, 2)

        return intermediate_output.detach()


def set_gradients(net, state):
    for param in net.parameters():
        param.requires_grad = state


class segmenterNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        # self.conv1 = nn.Conv2d(input_dim, hidden_dim, (1, 1), padding=0)
        # self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (1, 1), padding=0)
        self.conv3 = nn.Conv2d(input_dim, output_dim, (1, 1), padding=0)
        self.instance_norm = nn.InstanceNorm2d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.instance_norm(x)
        # x = F.relu(self.conv1(x))
        # x = F.interpolate(x, scale_factor=4, mode='bilinear')
        # x = F.relu(self.conv2(x))
        # x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=16, mode='bilinear')
        return x

class segmenterNetOld(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, (1, 1), padding=0)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (1, 1), padding=0)
        self.conv3 = nn.Conv2d(hidden_dim, output_dim, (1, 1), padding=0)
        self.instance_norm = nn.InstanceNorm2d(input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.instance_norm(x)
        x = F.relu(self.conv1(x))
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        return x

class segmenterNetInpaint(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


@torch.jit.script
def resize(classes: torch.Tensor, size: int):
    return F.interpolate(classes, (size, size), mode="bilinear", align_corners=False)


def get_crf_loss(img, code, cont_crf_loss):
    loss_crf = cont_crf_loss(resize(img, 56), norm(resize(code, 56))).mean()
    return loss_crf


def segment_from_feature(segment_net, feat_model, feat_model_seg, pred_func, cre_loss, cont_crf_loss,
                         img, args, is_eval=False, gt_mask=None):

    batch_size, ch, h, w = img.size()
    if gt_mask is None:
        if is_eval:
            feat_model_seg.eval()
            segment_net.eval()
        else:
            feat_model_seg.train()
            segment_net.train()

        loss_cluster, segment_large, intermedia_segment_feat, code = pred_func(feat_model_seg, segment_net, img, args)
        if is_eval:
            segment_pred_soft = F.gumbel_softmax(segment_large, hard=True, dim=1)
            return segment_pred_soft
        else:
            segment_pred_soft = F.gumbel_softmax(segment_large, hard=True, dim=1)
            # segment_pred_soft = torch.softmax(segment_large, dim=1)

        loss_crf = cont_crf_loss(resize(img, 56), norm(resize(code, 56))).mean()
        # loss_crf = utils.total_variation_loss(code, 1)
    else:
        segment_pred_soft = gt_mask
        loss_crf = 0
        loss_cluster = 0

    if segment_pred_soft.size(1) > 4:
        mask_element_ratio = segment_pred_soft[:(batch_size//2)].sum((2, 3)) / (h * w)
        _, token_indices = torch.topk(mask_element_ratio, 4, dim=1)
        token_indices = torch.cat(2 * [token_indices], 0)
        token_indices = token_indices.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)

        # noise = torch.rand_like(token_indices.float())
        # noise_indices = torch.argsort(noise, dim=1)[:, :4]
        # token_indices = token_indices.gather(1, noise_indices).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        # k_label = torch.stack([torch.zeros_like(k_label), k_label], 1)
        # token_indices = k_label.reshape(batch_size, 2, 1, 1).repeat(1, 1, h, w)
        segment_pred_soft = segment_pred_soft.gather(1, token_indices)

    cur_num_segments = segment_pred_soft.size(1)
    mask_element_ratio = segment_pred_soft.sum((2, 3)) / (h * w)

    # mask_element_ratio = segment_pred_soft[:, 1:].sum((1, 2, 3)) / (h * w)
    # mask_corr_matrix = torch.einsum("nkhw,nkhw->nhw", segment_pred_soft, segment_pred_soft)
    # mask_entropy_loss = -1 * (mask_element_ratio + 1e-6).log().mean()
    # mask_entropy_loss = segment_pred_soft[:(batch_size//2)].prod(0).mean()

    # mask_entropy_loss = -1 * ((mask_element_ratio[:, :1] + 1e-6).log() + (mask_element_ratio[:, 1:] / 0.5 + 1e-6).log()).mean()

    if args.use_discrete:
        img_segments = torch.stack(cur_num_segments * [img], 1).reshape(batch_size * cur_num_segments, ch, h, w)
        # intermediate_output_ = feat_model.get_intermediate_layers(img_segments, args.n_last_blocks,
        #                                                           mask=segment_pred_soft)
        # segment_pred_soft_large = F.interpolate(segment_pred_soft, scale_factor=args.patch_size, mode='nearest')
        # img_segments = img.unsqueeze(1) * segment_pred_soft_large.unsqueeze(2)
        # img_segments = img_segments.reshape(batch_size * cur_num_segments, ch, h, w)
        attn_cls = segment_pred_soft.reshape(batch_size * cur_num_segments, -1)
        intermediate_output_ = feat_model.get_intermediate_layers(img_segments, args.n_last_blocks,
                                                                  mask=None, attn_cls=attn_cls)
    else:
        img_segments = img.unsqueeze(1) * segment_pred_soft.unsqueeze(2)
        img_segments = img_segments.reshape(batch_size * cur_num_segments, ch, h, w)
        intermediate_output_ = feat_model.get_intermediate_layers(img_segments, args.n_last_blocks,
                                                                  mask=None)

    # compute image tokens for each image mask
    img_tokens = torch.cat([x[:, 0] for x in intermediate_output_], dim=-1)
    # reshape tokens so that they in the shape of batch_size x num_segments x feat_dim
    img_tokens = img_tokens.reshape(batch_size, cur_num_segments, -1)
    # normalize each feature token
    img_tokens = img_tokens / (img_tokens.norm(p=2, dim=2, keepdim=True) + 1e-6)

    img_token_cont, img_token_cont_n = img_tokens.split(batch_size // 2, dim=0)
    img_token_cont_mat_mask = torch.einsum("nak,nbk->nab", img_token_cont, img_token_cont_n) / args.softmax_temp_token
    mask_cross_label = torch.arange(0, cur_num_segments, device=img.device).unsqueeze(0).repeat(batch_size // 2, 1)
    token_loss = cre_loss(img_token_cont_mat_mask, mask_cross_label)

    img_token_cont_mat = torch.einsum("nsk,msk->snm", img_token_cont, img_token_cont_n) / args.softmax_temp_batch
    cross_label = torch.arange(0, (batch_size // 2), device=img.device).unsqueeze(0).repeat(cur_num_segments, 1)
    batch_loss = cre_loss(img_token_cont_mat, cross_label)

    mask_resize_flat = norm(resize(segment_pred_soft[:(batch_size // 2)], 28)).reshape(batch_size // 2, cur_num_segments, -1)
    mask_cor_mat = torch.einsum("nsk,msk->snm", mask_resize_flat, mask_resize_flat) / args.softmax_temp_batch
    mask_entropy_loss = cre_loss(mask_cor_mat, cross_label)

    if args.use_spatial_feedback:
        # _, token_indices = torch.topk(mask_element_ratio, cur_num_segments, dim=1)

        # img_token_norm = intermediate_output_[-2][:, 0].reshape(batch_size, cur_num_segments, -1)
        # # img_token_norm = img_token_norm.gather(1, token_indices[:, 1][:, None, None].repeat(1, 1, img_token_norm.size(2))).squeeze()
        # img_token_norm = F.normalize(img_token_norm, p=2, dim=-1)
        # normalized_intermediate_output = F.normalize(intermedia_segment_feat.permute(0, 2, 3, 1), p=2, dim=-1)
        # ## option 2
        # norm_inter_output_1, norm_inter_output_2 = normalized_intermediate_output.split(batch_size // 2, dim=0)
        # corr_matrix = torch.einsum('nabk, nhwk->nabhw', norm_inter_output_1, norm_inter_output_2)
        # norm_seg = F.normalize(resize(segment_large, normalized_intermediate_output.size(1)), p=2, dim=1)
        # seg_large_1, seg_large_2 = norm_seg.split(batch_size // 2, dim=0)
        # corr_matrix_seg = torch.einsum('nkab, nkhw->nabhw', seg_large_1, seg_large_2)
        # loss_spatial = -(corr_matrix * (corr_matrix_seg - args.spatial_bias)).mean()

        ## option 3
        # corr_matrix = torch.einsum('nabk,nsk->nsab', normalized_intermediate_output, img_token_norm).abs()
        # pseudo_mask = corr_matrix.argmax(1).long().detach()
        # loss_spatial = cre_loss(resize(segment_large, pseudo_mask.size(1)), pseudo_mask)

        ## option 1
        # selected_mask = segment_pred_soft.gather(1, token_indices[:, 1:2, None, None].repeat(1, 1, h, w))
        # selected_mask = resize(selected_mask, normalized_intermediate_output.size(1)).squeeze()
        # corr_matrix = torch.einsum('nabk,nk->nab', normalized_intermediate_output, img_token_norm).abs()
        # loss_spatial = -((corr_matrix * selected_mask).sum((1, 2)) / (selected_mask.sum((1, 2)) + 1e-6)).mean()

        ## option 4
        normalized_intermediate_output = F.normalize(intermedia_segment_feat, p=2, dim=1)
        norm_inter_output_1, norm_inter_output_2 = normalized_intermediate_output.permute(0, 2, 3, 1).split(batch_size // 2, dim=0)
        corr_matrix = torch.einsum('nabk, nhwk->nabhw', norm_inter_output_1, norm_inter_output_2)

        normalized_code = F.normalize(code.permute(0, 2, 3, 1), p=2, dim=-1)
        norm_code_1, norm_code_2 = normalized_code.split(batch_size // 2, dim=0)
        corr_matrix_code = torch.einsum('nabk, nhwk->nabhw', norm_code_1, norm_code_2)
        loss_spatial = -0.25 * ((corr_matrix - args.spatial_bias_aug) * corr_matrix_code).mean()

        corr_matrix_self = torch.einsum('nkab, nkhw->nabhw', normalized_intermediate_output, normalized_intermediate_output)
        corr_matrix_code_self = torch.einsum('nabk, nhwk->nabhw', normalized_code, normalized_code)
        loss_spatial += -0.67 * ((corr_matrix_self - args.spatial_bias_self) * corr_matrix_code_self).mean()

        norm_inter_output_1_shift = torch.roll(norm_inter_output_1, 1, dims=0)
        norm_inter_output_2_shift = torch.roll(norm_inter_output_2, 1, dims=0)
        corr_matrix_shift = torch.einsum('nabk, nhwk->nabhw', norm_inter_output_1_shift, norm_inter_output_2_shift)
        norm_code_1_shift = torch.roll(norm_code_1, 1, dims=0)
        norm_code_2_shift = torch.roll(norm_code_2, 1, dims=0)
        corr_matrix_code_shift = torch.einsum('nabk, nhwk->nabhw', norm_code_1_shift, norm_code_2_shift)
        loss_spatial += -0.63 * ((corr_matrix_shift - args.spatial_bias_random) * corr_matrix_code_shift).mean()

        # ## option 5
        # normalized_intermediate_output_2 = F.normalize(intermediate_output_[-2][:, 1:], p=2, dim=-1)
        # normalized_intermediate_output_2 = normalized_intermediate_output_2.reshape(batch_size, cur_num_segments, normalized_intermediate_output_2.size(1), normalized_intermediate_output_2.size(2))
        # segment_pred_small = F.gumbel_softmax(resize(segment_large, 14) / 0.01, hard=True, dim=1).reshape(batch_size, cur_num_segments, -1)
        # segment_features = segment_pred_small.unsqueeze(3) * normalized_intermediate_output_2
        # segment_features = segment_features.sum(2) / (segment_pred_small.sum(2).unsqueeze(-1) + 1e-6)
        #
        # # token loss
        # spat_token_mat = torch.einsum("nak,nbk->nab", segment_features[:(batch_size//2)],
        #                               segment_features[(batch_size//2):]) / args.softmax_temp_token
        # # mask_cross_label = torch.arange(0, cur_num_segments, device=img.device).unsqueeze(0).repeat(batch_size // 2, 1)
        # loss_spatial = cre_loss(spat_token_mat, mask_cross_label)
        #
        # # batch loss
        # spat_token_cont_mat = torch.einsum("nsk,msk->snm", segment_features[:(batch_size//2)], segment_features[(batch_size//2):]) / args.softmax_temp_batch
        # spat_cross_label = torch.arange(0, (batch_size // 2), device=img.device).unsqueeze(0).repeat(cur_num_segments, 1)
        # spat_batch_loss = cre_loss(spat_token_cont_mat, spat_cross_label)
        # loss_spatial += spat_batch_loss

    else:
        loss_spatial = 0





    # batch_loss = -1 * (torch.softmax(img_token_cont / args.softmax_temp_batch, -1) *
    #                    torch.log_softmax((img_token_cont_n - ema_mean) / args.softmax_temp_batch, -1)).sum(-1).mean()
    # ema_mean = 0.99 * ema_mean + (1 - 0.99) * img_tokens.detach().mean(0, keepdim=True)

    # img_token_cont_mat = torch.einsum("nsk,msk->snm", img_tokens, img_tokens) / args.softmax_temp_batch
    # cross_label = torch.arange(0, batch_size, device=img.device).unsqueeze(0).repeat(cur_num_segments, 1)
    # batch_loss = cre_loss(img_token_cont_mat, cross_label)

    return token_loss, loss_crf, batch_loss, mask_entropy_loss, loss_cluster, segment_pred_soft, loss_spatial


def split_and_combine(input_tensor, num_display):
    num_display_half = num_display // 2
    input_tensor_1, input_tensor_2 = input_tensor.split(input_tensor.size(0) // 2, dim=0)
    return torch.cat([input_tensor_1[:num_display_half], input_tensor_2[:num_display_half]], 0)


def compute_val_masks(segmenter, model, segment_func, cre_loss, cont_crf_loss, img_val, args, iters,
                      image_name='Val', gt_mask=None):
    NUM_DISPLAY = min(args.batch_size, 8)

    if image_name[:3] != 'Val':
        img_val = split_and_combine(img_val, NUM_DISPLAY)
        if gt_mask is not None:
            gt_mask = split_and_combine(gt_mask, NUM_DISPLAY)

    if gt_mask is not None:
        segment_large_label = gt_mask
        if args.dataset == 'coco-stuff27':
            class_labels = {i: f'C-{i}' for i in range(1, 28)}
        elif args.dataset == 'voc2012':
            class_labels = {i: f'C-{i}' for i in range(1, 20)}
        else:
            class_labels = {i: f'C-{i}' for i in range(1, 91)}
    else:
        with torch.no_grad():
            segment_large = segment_from_feature(
                segmenter, model, model, segment_func, cre_loss, cont_crf_loss, img_val, args, is_eval=True)

            if args.use_discrete:
                segment_large = F.interpolate(segment_large, scale_factor=args.patch_size, mode='bilinear')

        class_labels = {i: f'Class {i}' for i in range(1, args.num_segments+1)}
        segment_large_label = segment_large[:NUM_DISPLAY, ...].argmax(1, keepdim=True)

    grid_mask = torch.cat([segment_large_label[i, 0] for i in range(NUM_DISPLAY)], 1)
    grid_img = vutils.make_grid(img_val[:NUM_DISPLAY], normalize=True, nrow=NUM_DISPLAY, padding=0)
    # grid = torch.cat([grid_img, grid_mask], dim=-2)
    images = wandb.Image(grid_img,
                         masks={'predictions': {'mask_data': grid_mask.detach().cpu().numpy(),
                                                'class_labels': class_labels}}
                         )
    wandb.log({f'Image and Masks {image_name}': images}, step=iters)


def compute_val_masks_4_times(segmenter, model, segment_func, cre_loss, cont_crf_loss, img, label, img_val, label_val, args, iters):
    compute_val_masks(segmenter, model, segment_func, cre_loss, cont_crf_loss,
                                  img_val, args, iters, image_name='Val')
    compute_val_masks(segmenter, model, segment_func, cre_loss, cont_crf_loss,
                                  img_val, args, iters, image_name='Val GT', gt_mask=label_val)
    compute_val_masks(segmenter, model, segment_func, cre_loss, cont_crf_loss, img,
                                  args, iters, image_name='Train')
    compute_val_masks(segmenter, model, segment_func, cre_loss, cont_crf_loss, img,
                                  args, iters, image_name='Train GT', gt_mask=label)


def compute_segmentation_loss(segment_large, label, cre_loss, args):
    segment_flat = segment_large.permute(0, 2, 3, 1).reshape(-1, args.num_segments)
    label_flat = label.long().reshape(-1)
    mask = (label_flat >= 0) & (label_flat < args.num_segments)
    segment_flat = segment_flat[mask, :]
    label_flat = label_flat[mask]
    loss = cre_loss(segment_flat, label_flat)

    return loss


def refine_segment_w_diffusion(model, diffusion, x_0, mask, last_time=5, num_iters=10):

    device = x_0.device
    batch_size = x_0.shape[0]

    kernel = np.ones((3, 3), np.uint8)
    mask_ = np.stack([cv2.dilate(m_cur[0].cpu().numpy(), kernel, iterations=2) for m_cur in mask], axis=0)
    mask_ = torch.from_numpy(mask_).unsqueeze(1).ge(0.1).float().to(device)
    # mask_ = mask

    for j in range(num_iters):
        mask_erosion = np.stack([cv2.erode(m_cur[0].cpu().numpy(), kernel, iterations=3) for m_cur in mask_], axis=0)
        mask_erosion = torch.from_numpy(mask_erosion).unsqueeze(1).ge(0.1).float().to(device)
        mask_delta = mask_ - mask_erosion


        t = torch.tensor(batch_size * [last_time], device=x_0.device)
        img = diffusion.q_sample(x_0, t, noise=None)

        indices = list(range(last_time))[::-1]

        for i in indices:
            t = torch.tensor(batch_size * [i], device=x_0.device)

            out = diffusion.p_sample(
                model,
                img,
                t + 1,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )
            img = out["sample"]

            x_t = diffusion.q_sample(x_0, t, noise=None)
            img = img * mask_ + x_t * (1 - mask_)

            inp_err = (img - x_0).abs().mean(1, keepdim=True)
            inp_err = inp_err.ge(0.2).float()
            mask_delta = mask_delta * inp_err
            mask_ = mask_erosion + mask_delta
            mask_ = TF.gaussian_blur(mask_, 11, [2., 2.]).ge(0.5).float()

    return mask_

def refine_segment_w_diffusion_grad(model, diffusion, x_0, mask, last_time=5, num_iters=10):

    device = x_0.device
    batch_size = x_0.shape[0]

    kernel = np.ones((3, 3), np.uint8)
    mask_ = np.stack([cv2.dilate(m_cur[0].cpu().numpy(), kernel, iterations=5) for m_cur in mask], axis=0)
    mask_ = torch.from_numpy(mask_).unsqueeze(1).ge(0.1).float().to(device)
    # mask_ = mask

    for j in range(num_iters):
        mask_erosion = np.stack([cv2.erode(m_cur[0].cpu().numpy(), kernel, iterations=3) for m_cur in mask_], axis=0)
        mask_erosion = torch.from_numpy(mask_erosion).unsqueeze(1).ge(0.1).float().to(device)
        mask_dilated = np.stack([cv2.dilate(m_cur[0].cpu().numpy(), kernel, iterations=3) for m_cur in mask_], axis=0)
        mask_dilated = torch.from_numpy(mask_dilated).unsqueeze(1).ge(0.1).float().to(device)

        mask_delta = mask_dilated - mask_erosion

        t = torch.tensor(batch_size * [last_time], device=x_0.device)
        img = diffusion.q_sample(x_0, t, noise=None)
        img_noise = img.clone()

        noise = 0.1 * torch.randn_like(mask)
        mask_noise = mask_ + noise

        indices = list(range(last_time))[::-1]

        for i in indices:
            t = torch.tensor(batch_size * [i], device=x_0.device)
            sample_noise = torch.randn_like(x_0)

            out = diffusion.p_sample(
                model,
                img,
                t + 1,
                clip_denoised=True,
                noise=sample_noise,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )
            img = out["sample"]

            x_t = diffusion.q_sample(x_0, t, noise=None)
            img = img * mask_ + x_t * (1 - mask_)

            out_n = diffusion.p_sample(
                model,
                img_noise,
                t + 1,
                clip_denoised=True,
                noise=sample_noise,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )
            img_noise = out_n["sample"]
            img_noise = img_noise * mask_noise + x_t * (1 - mask_noise)

        inp_err = (img - x_0).abs().mean(1, keepdim=True)
        inp_err_noise = (img_noise - x_0).abs().mean(1, keepdim=True)

        mean_error_mask = (inp_err * mask_).sum((1,2,3), keepdim=True) / mask_.sum((1,2,3), keepdim=True)
        mean_error_mask_noise = (inp_err_noise * mask_noise).sum((1,2,3), keepdim=True) / mask_noise.sum((1,2,3), keepdim=True)

        inp_err = (inp_err - mean_error_mask) / mask_.sum((1,2,3), keepdim=True)
        inp_err_noise = (inp_err_noise - mean_error_mask_noise) / mask_noise.sum((1,2,3), keepdim=True)
        grad_est = (inp_err_noise - inp_err) / noise
        grad_est = grad_est * mask_delta

        mask_ += 1 * grad_est
        mask_ = mask_.clamp(0, 1)
        mask_ = TF.gaussian_blur(mask_, 11, [2., 2.]).ge(0.5).float()

        # inp_err = (img - x_0).abs().mean(1, keepdim=True)
        # inp_err = inp_err.ge(0.2).float()
        # mask_delta = mask_delta * inp_err
        # mask_ = mask_erosion + mask_delta
        # mask_ = TF.gaussian_blur(mask_, 11, [2., 2.]).ge(0.5).float()

    return mask_


@torch.no_grad()
def mae_inpaint_image(img, model, mask):
    # x = torch.tensor(img)
    x = img.clone().detach()

    # run MAE
    loss, y, mask, _, _ = model(x.float(), mask=mask)
    y = model.unpatchify(y).detach()
    # y = torch.einsum('nchw->nhwc', y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    # mask = torch.einsum('nchw->nhwc', mask).detach()

    # x = torch.einsum('nchw->nhwc', x)

    # # masked image
    # im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    return im_paste

class segmenterParallel(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.module_list = nn.ModuleList(batch_size * [featureNet(3, 1, 128)])

    def forward(self, x):
        out = []
        for inp, module in zip(x, self.module_list):
            out.append(module(inp.unsqueeze(0)))

        return torch.cat(out, 0)

def update_mask_with_convnet(img_batch, initial_mask, num_iters=30):
    segment_pred_list = []
    initial_mask = initial_mask.ge(0.1).float()
    for img, mask in zip(img_batch, initial_mask):
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)

        feature_net = featureNet(3, 1, 64).to(img.device)
        # feature_net = segmenterParallel(img_batch.shape[0]).to(img_batch.device)

        optimizer = optim.AdamW(feature_net.parameters(), 2e-3, betas=(0.9, 0.95), weight_decay=1e-4)

        # img_inp = torch.cat([img, inp_small], 0)
        # mask_org_target = torch.cat([mask_org_patch.detach(), torch.zeros_like(mask_org_patch)], 0)

        bce_loss = nn.BCEWithLogitsLoss()

        for _ in range(num_iters):
            optimizer.zero_grad()

            segment_pred = feature_net(img)
            loss_reg = my_gradient_reg_func(segment_pred).mean()
            loss = bce_loss(segment_pred, mask.detach()) + 0.1 * loss_reg

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            segment_pred = feature_net(img)
            segment_pred_list.append(segment_pred.detach())

    segment_pred_list = torch.cat(segment_pred_list, 0)

    return segment_pred_list

def segment_from_feature_sempart(segment_net, feat_model, img, args, eval=False, segment_input=None, guided_filter=None):
    if eval:
        with torch.no_grad():
            feature = feat_model(img)
            segment_pred, _ = segment_net(feature, img)
            segment_large = torch.sigmoid(segment_pred / 1)

        return segment_large
    else:
        feature = feat_model(img)
        if guided_filter is not None:
            img_small = F.interpolate(img, scale_factor=1 / args.patch_size, mode='bilinear').mean(1, keepdims=True)
            feature = guided_filter(feature, img_small)

        if segment_input is None:
            segment_pred, code = segment_net(feature, img)
            segment_large = torch.sigmoid(segment_pred / 1)
        else:
            segment_large = segment_input

        # segment_pred_soft = TF.gaussian_blur(segment_large, [15, 15], [3, 3])
        # segment_pred_soft = F.interpolate(segment_large, scale_factor=1/args.patch_size, mode='bilinear')
        segment_pred_soft = F.max_pool2d(segment_large, args.patch_size)

        b, _, h, w = segment_pred_soft.shape
        segment_flat = segment_pred_soft.reshape(b, 1, h * w).clamp(min=1e-6, max=1-1e-6)
        feature = feature.reshape(b, -1, h*w)

        # get weight matrix
        # feat_norm = F.normalize(feature, p=2, dim=1).reshape(b, d, h*w)
        # segment_flat = segment_pred_soft.reshape(b, 1, h*w)
        # feat_cross_correlation = torch.einsum('nka, nkb -> nab', feat_norm, feat_norm) # B x (HW) x (HW)

        # code_cross_correlation = utils.flatten_and_get_correlation(code)
        feat_cross_correlation = utils.flatten_and_get_correlation(feature)
        # # feat_cross_correlation = (feat_cross_correlation + code_cross_correlation) / 2
        #
        weight_ones = torch.ones_like(feat_cross_correlation)
        weight_matrix = torch.where(feat_cross_correlation > float(args.ncut_thr),
                                    weight_ones, args.epsilon * weight_ones
                                    )

        # weight_matrix = 0.7 * weight_matrix + 0.3 * (0.5 * code_cross_correlation + 0.5)

        ncut_loss1 = (segment_flat @ weight_matrix @ (1 - segment_flat).transpose(1, 2)) / (
                    segment_flat @ weight_matrix).sum((1,2), keepdim=True).clamp(min=1e-6)
        ncut_loss2 = ((1 - segment_flat) @ weight_matrix @ segment_flat.transpose(1, 2)) / (
            (1 - segment_flat) @ weight_matrix).sum((1,2), keepdim=True).clamp(min=1e-6)

        mask_loss = torch.mean(0.5 * (ncut_loss1 + ncut_loss2))

        # corr_loss = utils.get_corr_loss(code_cross_correlation, feature, shift=float(args.ncut_thr))
        # corr_loss = (feat_cross_correlation - code_cross_correlation).pow(2).mean()
        corr_loss = 0
        loss_entropy = 0


        return loss_entropy, mask_loss, corr_loss, segment_large