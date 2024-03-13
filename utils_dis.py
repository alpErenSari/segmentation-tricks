import argparse
import itertools

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob

import warnings
from torch.utils.data import Dataset, Subset
import h5py
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
import torchvision.utils as vutils
import random
import scipy.io as sio
from sklearn.utils import extmath

import pickle
import string
import io

PerceptualLoss = None

# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


def normalize_chunks(encoded, z_chunk_size):
    batch_size, z_dim = encoded.size()
    assert z_dim % z_chunk_size == 0, "z_chunk_size({}) should be dividable by latent dimension({})".format(
        z_chunk_size, z_dim)
    encoded = encoded.view(batch_size, z_dim // z_chunk_size, z_chunk_size)
    encoded = F.normalize(encoded, dim=2)
    # encoded = encoded.view(batch_size, z_dim)
    return encoded


def get_vec_sim_loss(outputs1, outputs2, z_chunk_size):
    encoded1 = normalize_chunks(outputs1['encoded1'], z_chunk_size)[:, 1:, :]
    encoded2 = normalize_chunks(outputs2['encoded1'], z_chunk_size)[:, 1:, :]
    return F.mse_loss(encoded1, encoded2)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort





def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


class imgLossModule(nn.Module):
    def __init__(self, use_lpips=True):
        super(imgLossModule, self).__init__()
        self.criterion_l1 = nn.L1Loss()
        self.use_lpips = use_lpips
        self.criterion_lpips = PerceptualLoss()

    def forward(self, outputs, data):
        loss_rec = self.criterion_l1(outputs['rec4'], data['x1']) + \
                   self.criterion_l1(outputs['rec1'], data['x1'])
        if self.use_lpips:
            loss_rec += self.criterion_lpips(outputs['rec4'], data['x1']).mean() + \
                    self.criterion_lpips(outputs['rec1'], data['x1']).mean()
        return loss_rec


class imgLossModule2(nn.Module):
    def __init__(self):
        super(imgLossModule2, self).__init__()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_lpips = PerceptualLoss()

    def forward(self, outputs, inputs):
        loss_rec = self.criterion_l1(outputs, inputs) + \
            self.criterion_lpips(outputs, inputs).mean()
        return loss_rec






