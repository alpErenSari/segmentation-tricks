import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch_dis.pos_embed import get_1d_sincos_pos_embed_from_grid

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, inp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.to_new_embedding = nn.Linear(dim, dim)
        self.to_new_embedding = nn.Sequential(
            Rearrange('b d h w -> b (h w) d'),
            nn.LayerNorm(inp_dim),
            nn.Linear(inp_dim, dim),
        )

        self.embedding_to_patch = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim * 32),
            # nn.LayerNorm(patch_dim),
            # nn.Tanh(),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=(image_height // patch_height)),
            nn.Conv2d(32, channels, 3, padding=1)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, return_latent=False, input_img=False):
        if input_img:
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = self.to_new_embedding(img)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        if return_latent:
            return x
        elif not input_img:
            return self.embedding_to_patch(x[:, 1:])


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class predTransformer(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, mlp_dim, pool = 'cls',
                 dim_head = 64, dropout = 0., emb_dropout = 0., softmask=False, temp=0.01, norm_pix_loss=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Linear(dim, dim)
        self.norm_pix_loss = norm_pix_loss
        self.softmask = softmask
        self.loss_ema = 10
        self.loss_ema_lambda = 0.99
        self.temp = temp
        # self.use_boundry_mask = use_boundry_mask

        self.to_patch_norm = nn.Sequential(
            Rearrange('b d h w -> b (h w) d'),
            # nn.LayerNorm(dim, elementwise_affine=True)
        )
        # self.to_patch_embedding = nn.Linear(dim, dim)

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches ** 2, dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, img, mask, use_boundry_mask=False):
        img = self.to_patch_norm(img)
        if self.norm_pix_loss:
            mean = img.mean(dim=-1, keepdim=True)
            var = img.var(dim=-1, keepdim=True)
            img = (img - mean) / (var + 1.e-6)**.5



        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        num_segments = mask.size(1)

        mask_tokens = repeat(self.mask_token, '1 1 d -> b n d', b=b, n=n)

        if use_boundry_mask:
            # mask_hard_loss = 0.7 * my_gradient_func(mask) + 0.3 * mask
            mask_hard_loss = my_gradient_func(mask)
            mask_hard_loss = mask_hard_loss.reshape(b, n)
        else:
            mask_hard_loss = mask.reshape(b, n)

        mask = mask.reshape(b, num_segments, -1).permute(0, 2, 1)
        if not self.softmask:
            mask_hard = mask.detach().ge(0.5).float()
            mask = mask_hard + mask - mask.detach()
        else:
            mask_hard = mask.detach()

        x = x * (1 - mask) + mask_tokens * mask

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        x = self.mlp_head(x)
#         mse_loss = self.mse_loss(img, x)
#         mse_loss = (img - x).pow(2).mean()
#         mask_one_hot = mask.detach().ge(0.5).float()

        mask_hard_loss = mask_hard.squeeze()
        img_soft = F.softmax(img / self.temp, dim=-1)
        entropy_loss_spatial = torch.sum(-img_soft * F.log_softmax(x / self.temp, dim=-1), dim=-1)
        entropy_loss = (entropy_loss_spatial * mask_hard_loss).sum() / mask_hard_loss.sum().clamp(1e-6)
        # entropy_loss = torch.mean(entropy_loss_spatial)

        min_val, max_val = img_soft.min(), img_soft.max()
        # mse_loss_spatial = (img - x).pow(2).mean(-1)
        # # mask_hard_loss = mask_hard_loss * mse_loss_spatial.detach().le(self.loss_ema).float()
        # mse_loss = (mse_loss_spatial * mask_hard_loss).sum() / (mask_hard_loss.sum() + 1e-6)

        # self.loss_ema =  self.loss_ema_lambda * self.loss_ema + (1 - self.loss_ema_lambda) * float(mse_loss_spatial.mean())
        # mask_token_norm = torch.norm(self.mask_token.detach().reshape(-1), p=2)

        return x, entropy_loss, min_val, max_val, entropy_loss_spatial


def my_gradient_func(img):
    B, C, H, W = img.size()
    dx = img[:, :, :, :-1] - img[:, :, :, 1:]
    dy = img[:, :, :-1, :] - img[:, :, 1:, :]

    dx1_app = torch.cat([dx, torch.zeros(B, C, H, 1, device=img.device)], dim=-1)
    dx2_app = torch.cat([torch.zeros(B, C, H, 1, device=img.device), -dx], dim=-1)
    dy1_app = torch.cat([dy, torch.zeros(B, C, 1, W, device=img.device)], dim=-2)
    dy2_app = torch.cat([torch.zeros(B, C, 1, W, device=img.device), -dy], dim=-2)

    derivative_estimate = torch.cat([dx1_app, dx2_app, dy1_app, dy2_app], dim=1)
    # derivative_estimate = derivative_estimate.ge(0.5).sum(1, keepdim=True) > 0
    derivative_estimate = torch.sigmoid((F.relu(derivative_estimate - 0.5).sum(1, keepdim=True) - 1e-2) / 0.001)

    return derivative_estimate.float()

def my_gradient_reg_func(img):
    B, C, H, W = img.size()

    grad_est = torch.zeros(B, H, W, device=img.device)
    dx = (img[:, :, :-1, :-1] - img[:, :, :-1, 1:]).pow(2).mean(1)
    dy = (img[:, :, :-1, :-1] - img[:, :, 1:, :-1]).pow(2).mean(1)

    grad_est[:, :-1, :-1] = dx + dy
    return grad_est

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class predTransformer2Img(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, mlp_dim, pool='cls',
                 dim_head=64, dropout=0., emb_dropout=0., softmask=False):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Linear(dim, dim)
        self.softmask = softmask
        self.to_patch_embedding = nn.Linear(dim, dim)

        # self.embedding_to_patch = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     Rearrange('b (p p) d -> b d p p')
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches ** 2, dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, img, mask):
        # img = img
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        num_segments = mask.size(1)

        mask = mask.reshape(b, num_segments, -1).permute(0, 2, 1)
        # mask = mask + 0.1 * torch.randn_like(mask).detach()
        if not self.softmask:
            mask_hard = mask.detach().ge(0.5).float()
            mask = mask_hard + mask - mask.detach()


        # x = x * mask + mask_tokens * (1 - mask)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, mlp_dim, args,
                 pool='cls', channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., norm_pix_loss=False):
        super().__init__()

        self.z_chunk_size = args.z_chunk_size
        self.layer_norm = nn.LayerNorm([num_patches, dim], elementwise_affine=False)
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.z_chunk_size))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.decoder_embed = nn.Linear(self.z_chunk_size, dim, bias=True)
        if args.use_softmax_prediction_loss:
            self.decoder_pred = nn.Linear(dim, args.num_latents_book, bias=True)
        else:
            self.decoder_pred = nn.Linear(dim, self.z_chunk_size, bias=True)
        self.dropout = nn.Dropout(emb_dropout)
        self.norm_pix_loss = norm_pix_loss
        self.normalize_pred_vectors = args.normalize_pred_vectors
        self.use_softmax_prediction_loss = args.use_softmax_prediction_loss
        self.num_latents_book = args.num_latents_book
        self.criterion_cross = nn.CrossEntropyLoss(reduction='none')
        self.criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
        self.criterion_entropy = HLoss()
        self.use_same_mask_batch = args.use_same_mask_batch
        self.disable_pos_embedding = args.disable_pos_embedding

        self.initialize_weights()

    def initialize_weights(self):
        embed_positions = np.arange(self.pos_embedding.shape[-2], dtype=np.float32)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embedding.shape[-1], embed_positions)
        self.pos_embedding.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # x = self.layer_norm(x)
        if self.normalize_pred_vectors:
            x = x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-6)
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.use_same_mask_batch:
            noise = torch.rand(1, L, device=x.device).repeat(N, 1)
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        b, n, _ = x.shape

        x = self.decoder_embed(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        if not self.disable_pos_embedding:
            x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, target, pred, mask, phase):
        """
        target: [N, L, D]
        pred: [N, L, D]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        if phase == 'trans':
            loss = (pred - target) ** 2
        elif phase == 'auto':
            loss = (pred + target) ** 2

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss_mask = (loss * mask).sum() / (mask.sum() + 1e-6)  # mean loss on removed patches
        loss_nonmask = (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)  # mean loss on removed patches
        # loss_nonmask = torch.zeros_like(loss_mask)
        return loss_mask, loss_nonmask

    def forward_loss_softmax(self, target, pred, mask, phase):
        """
               target: [N, L, D]
               pred: [N, L, D ]
               mask: [N, L], 0 is keep, 1 is remove,
               """
        B, L, D = pred.size()

        target = target.reshape(-1)
        pred = pred.reshape(-1, D)
        mask = mask.reshape(-1)

        if phase == 'trans':
            loss = self.criterion_cross(pred, target) # [N x L, D]
        elif phase == 'auto':
            # target = (target * (1 - mask[:, 0]) + (D - 1) * torch.ones_like(target) * mask[:, 0]).long()
            target_one_hot = (1 - F.one_hot(target, D)) / (D - 1)
            loss = self.criterion_bce(pred, target_one_hot).mean(1)
            # loss = self.criterion_entropy(pred)
        # loss = self.criterion_cross(pred, target)  # [N x L, D]
        loss_mask = (loss * mask).sum() / (mask.sum() + 1e-6)
        loss_nonmask = 0 * (loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)

        return loss_mask, loss_nonmask

    def forward(self, x, mask_ratio=0.75, indices=None, phase='auto'):
        x = self.layer_norm(x)
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        pred = self.forward_decoder(x_masked, ids_restore)
        if self.use_softmax_prediction_loss:
            loss_mask, loss_nonmask = self.forward_loss_softmax(indices, pred, mask, phase)
        else:
            loss_mask, loss_nonmask = self.forward_loss(x, pred, mask, phase)
        return loss_mask, loss_nonmask, pred, mask


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(1)
        return b