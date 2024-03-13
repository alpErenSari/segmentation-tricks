import math
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms.functional as TF

import utils_data
import utils_dis
from vit_pytorch_dis.vit import Transformer


class segmenterNetRGB(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, patch_size=16, img_size=224):
        super().__init__()
        upscale_layer_num = int(math.log2(patch_size))
        self.input_dim = input_dim

        self.inp_ch = [hidden_dim + 3 , 195] + [2 * hidden_dim + 3] * (upscale_layer_num - 2)
        self.out_ch = [192] + [2 * hidden_dim] * (upscale_layer_num - 1)

        self.conv_input = nn.Conv2d(input_dim, hidden_dim, 1, padding=0)
        self.inp_transformer = Transformer(hidden_dim, 2, 2, hidden_dim, hidden_dim)
        self.conv_coarse = nn.Conv2d(hidden_dim, output_dim, 1, padding=0)
        # self.conv_input = myCluster(input_dim, 2 * input_dim, proj_type='nonlinear')
        self.resize_ratios = [(2 ** i) for i in range(upscale_layer_num-1, -1, -1)]

        self.layers = nn.ModuleList()

        for i in range(upscale_layer_num):
            inp_dim = self.inp_ch[i]
            out_dim = self.out_ch[i]
            if i == upscale_layer_num - 1:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(inp_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Conv2d(out_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.01, inplace=True)
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(inp_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(0.01, inplace=True)
                ))

        self.conv_final = nn.Conv2d(self.out_ch[-1] + 3, output_dim, 1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    @staticmethod
    def downscale_helper(img, down_ratio):
        if down_ratio == 1:
            return img
        else:
            return F.interpolate(img, scale_factor=1/down_ratio, mode='bilinear')

    def forward(self, x, img):
        x = self.conv_input(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        x = self.inp_transformer(x)
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        code = self.conv_coarse(x)

        for layer, resize_ratio in zip(self.layers, self.resize_ratios):
            cur_img = self.downscale_helper(img, resize_ratio)
            x = self.upsample(x)
            x = torch.cat([x, cur_img], 1)
            x = layer(x)

        x = torch.cat([x, img], 1)

        x = self.conv_final(x)
        return x, code


class DINOWrapper(nn.Module):
    def __init__(self, model_dino, return_attention=False, args=None):
        super().__init__()
        self.model_dino = model_dino
        self.return_attention = return_attention
        self.h_featmap = args.h_featmap
        self.w_featmap = args.w_featmap
        self.patch_size = args.patch_size
        self.from_begin = args.from_begin
        self.n_last_blocks = args.n_last_blocks
        self.use_keys = args.use_keys
        self.get_cls_token = args.get_cls_token

    def set_hw_feat_ratio(self, ratio):
        self.h_featmap = int(self.h_featmap * ratio)
        self.w_featmap = int(self.w_featmap * ratio)

    def set_args(self, h_featmap, w_featmap):
        self.h_featmap = int(h_featmap)
        self.w_featmap = int(w_featmap)

    @staticmethod
    def attention_transform(att, w_featmap, h_featmap, patch_size):
        b, nh = att.shape[:2]  # number of head
        # we keep only the output patch attention
        att = att[:, :, 0, 1:].reshape(b, nh, w_featmap, h_featmap)
        return att

    @torch.no_grad()
    def forward(self, samples):

        if self.from_begin:
            feat, _, qkvs = self.model_dino.get_intermediate_feat_from_start(samples, self.n_last_blocks)
        else:
            feat, _, qkvs = self.model_dino.get_intermediate_feat(samples, self.n_last_blocks)

        if self.use_keys:
            _, B, H, N, C = qkvs[0].size()
            intermediate_output = [x[1].permute(0, 2, 1, 3).reshape(B, N, H*C) for x in qkvs]
        else:
            intermediate_output = feat

        if self.get_cls_token:
            intermediate_output = torch.cat([x[:, 0, :] for x in feat], dim=-1)
        else:
            intermediate_output = torch.cat([x[:, 1:, :] for x in intermediate_output], dim=-1)
            b, n, d = intermediate_output.size()
            intermediate_output = intermediate_output.permute(0, 2, 1).reshape(b, d, self.w_featmap, self.h_featmap)

        if self.return_attention:
            attentions = self.model_dino.get_last_selfattention(samples)
            attentions = self.attention_transform(attentions[0], self.w_featmap, self.h_featmap, self.patch_size)
            return intermediate_output.detach(), attentions
        else:
            return intermediate_output.detach()