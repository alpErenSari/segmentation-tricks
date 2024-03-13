# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import argparse
import os
import sys
import time

import PIL.JpegImagePlugin
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps, Image
import wandb

from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from typing import Any, Callable, List, Optional, Tuple
import torchvision.utils as vutils


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f'Pretrained DINO loaded with {msg}')
        else:
            print("There is no reference weights available for this model => We use random weights.")

    return model


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)["state_dict"]
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t", metric_mode='train'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.metric_mode = metric_mode

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def get_my_dict(self):
        loss_dict = {}
        for name, meter in self.meters.items():
            loss_dict[name + f'_{self.metric_mode}'] = float(str(meter).split()[0])
        return loss_dict

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, epoch, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                iters = epoch * len(iterable) + i
                # wandb.log(self.get_my_dict(), step=iters)
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.2'
        os.environ['MASTER_PORT'] = args.port_dist
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class sempartRegLoss(nn.Module):
    def __init__(self, ch=3, sigma=1):
        super(sempartRegLoss, self).__init__()
        self.diff1 = differenceFilter(1)
        self.diff3 = differenceFilter(ch)
        self.sigma = sigma

    def forward(self, img, segment):
        img_diff = self.diff3(img)
        img_diff = torch.exp(-img_diff / self.sigma)
        segment_diff = self.diff1(segment)
        loss = (img_diff * segment_diff).mean()
        return loss

class sempartFeatRegLoss(nn.Module):
    def __init__(self, ch=3, sigma=1, threshold=0.2):
        super(sempartFeatRegLoss, self).__init__()
        self.diff1 = differenceFilter(1)
        self.diff3 = differenceFilter(ch)
        self.sigma = sigma
        self.threshold = threshold
        self.eps = 1e-6

    def forward(self, feat, segment):
        feat = F.normalize(feat, p=2, dim=1)
        feat_diff = self.diff3(feat)
        feat_diff = 1 - feat_diff / 2
        weight_ones = torch.ones_like(feat_diff)
        weight_matrix = torch.where(feat_diff > float(self.threshold),
                                    weight_ones, self.eps * weight_ones
                                    )
        segment_diff = self.diff1(segment)
        loss = (weight_matrix * segment_diff).mean()
        return loss


class differenceFilter(nn.Module):
    def __init__(self, ch):
        super(differenceFilter, self).__init__()
        self.ch = ch
        diff_filter = self.prepare_diff_filter(ch)
        self.conv_layer = nn.Conv2d(in_channels=ch, out_channels=ch * 8, kernel_size=3, bias=False, padding=1,
                                    padding_mode='reflect', groups=ch)
        self.conv_layer.weight.data = diff_filter

    def prepare_diff_filter(self, ch):
        diff_filter = torch.zeros((8, 9))
        for i in range(8):
            diff_filter[i, 4] = 1
            if i > 3:
                diff_filter[i, i + 1] = -1
            else:
                diff_filter[i, i] = -1
        #         self.diff_filter = self.diff_filter.repeat()
        diff_filter = diff_filter.reshape(8, 1, 3, 3).repeat(ch, 1, 1, 1)
        return diff_filter

    def forward(self, x):
        b, ch, h, w = x.shape
        x = self.conv_layer(x)
        x = x.reshape(b, ch, 8, h, w)
        x = x.pow(2).sum(1)
        return x




def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class ImageDataCOCO(Dataset):
    def __init__(
            self,
            annotations: COCO,
            img_ids: List[int],
            cat_ids: List[int],
            root_path: str,
            transform: Optional[Callable] = None,
            transform_target: Optional[Callable] = None,
            use_coarse_label: bool = True
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [os.path.join(root_path, img["file_name"]) for img in self.img_data]
        self.transform = transform
        self.transform_target = transform_target
        self.use_coarse_label = use_coarse_label

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"]
                                                 for ann in anns]), axis=0)).unsqueeze(0)

        if self.use_coarse_label:
            coarse_label = torch.zeros_like(mask)
            for fine, coarse in self.fine_to_coarse.items():
                coarse_label[mask == fine] = coarse
            coarse_label[mask == -1] = -1

            mask = coarse_label

        img = Image.open(self.files[i]).convert('RGB')

        img = self.transform(img)
        mask = self.transform_target(mask)

        return img, mask


def get_new_bbox(bbox, params):
    new_boxes = [max(bbox[0][0] - params[0], 0), max(bbox[0][1] - params[1], 0)]
    new_boxes += [min(new_boxes[0] + bbox[0][2], params[2]) - new_boxes[0],
                  min(new_boxes[1] + bbox[0][3], params[3]) - new_boxes[1]]
    return new_boxes

def train_transform_coco(
        img1: PIL.JpegImagePlugin.JpegImageFile,
        img2: PIL.JpegImagePlugin.JpegImageFile,
        image_size: int,
        transform_augment: transforms.Compose
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    params = transforms.RandomResizedCrop.get_params(img1, scale=(0.8, 1.0), ratio=(0.9, 1.1))

    img1 = TF.resized_crop(img1, *params, size=(image_size, image_size))
    img2 = TF.resized_crop(img2, *params, size=(image_size, image_size))

    # Random horizontal flipping
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)

    img1 = transform_augment(img1)

    return img1, img2

def contruct_mask_and_log(segment_pred, img, label, args, iters, phase='Val', show_soft_mask=True):
    b = segment_pred.size(0)
    segment_pred_resized = segment_pred.permute(1, 0, 2, 3).reshape(b * args.num_segments, 1, args.img_size, args.img_size)

    grid_img = vutils.make_grid(img, normalize=True, nrow=b, padding=0)

    if args.num_segments > 1:
        segment_pred_resized_bin = F.one_hot(segment_pred.argmax(1), num_classes=args.num_segments).permute(3, 0, 1, 2)
        segment_pred_resized_bin = segment_pred_resized_bin.reshape(b * args.num_segments, 1, args.img_size, args.img_size)
    else:
        segment_pred_resized_bin = segment_pred_resized.ge(0.5).float()

    grid_mask_bin = vutils.make_grid(segment_pred_resized_bin, normalize=False, nrow=b, padding=0)

    if show_soft_mask:
        grid_mask = vutils.make_grid(segment_pred_resized, normalize=True, nrow=b, padding=0)
        grid = torch.cat([grid_img, grid_mask, grid_mask_bin], dim=-2)
    else:
        grid = torch.cat([grid_img, grid_mask_bin], dim=-2)

    if args.dataset in ['duts', 'duts-neig'] or (args.dataset == 'imagenet100_subset' and phase == 'Val'):
        grid_label = vutils.make_grid(label, normalize=True, nrow=b, padding=0)
        grid = torch.cat([grid, grid_label], dim=-2)

    images = wandb.Image(grid)
    wandb.log({f'{phase} Image and Masks': images}, step=iters)

def contruct_mask_and_log_softmax(segment_pred, img, iters, phase='Val'):
    b = segment_pred.size(0)
    class_labels = {i+1: f'C {i+1}' for i in range(segment_pred.size(1))}
    segment_pred = segment_pred.argmax(1, keepdim=True)
    grid_img = vutils.make_grid(img.cpu(), normalize=True, nrow=b, padding=0)
    grid_img = grid_img.permute(1, 2, 0).numpy()
    grid_mask = torch.cat([x[0] for x in segment_pred.detach().cpu()], -1).numpy()

    images = wandb.Image(grid_img, masks={'predictions': {'mask_data': grid_mask, 'class_labels': class_labels}})
    wandb.log({f'{phase} Image and Masks': images}, step=iters)

class stegoCFG:
    def __init__(self, patch_size=16, dino_feat_type='feat', model_type='vit_small', dropout=False,
                 pretrained_weights=None, projection_type='nonlinear', feature_type='dino'):

        self.dino_patch_size = patch_size
        self.dino_feat_type = dino_feat_type
        self.model_type = model_type
        self.dropout = dropout
        self.pretrained_weights = pretrained_weights
        self.projection_type = projection_type
        self.feature_type = feature_type

def min_mask_loss_multi(mask, min_mask_area):
    return torch.relu(min_mask_area - mask.mean((-1, -2))).sum(1).mean()


def max_mask_loss_multi(mask, max_mask_area):
    return torch.relu(mask.mean((-1, -2)) - max_mask_area).sum(1).mean()

def flatten_and_get_correlation(feat):
    feat_norm = F.normalize(feat, p=2, dim=1)
    feat_cross_correlation = torch.einsum('nka, nkb -> nab', feat_norm, feat_norm)  # B x (HW) x (HW)

    return feat_cross_correlation.clamp(1e-9)

def flatten_and_get_correlation2feat(feat1, feat2):
    feat1_norm = F.normalize(feat1, p=2, dim=1)
    feat2_norm = F.normalize(feat2, p=2, dim=1)
    feat_cross_correlation = torch.einsum('nka, nkb -> nab', feat1_norm, feat2_norm)  # B x (HW) x (HW)

    return feat_cross_correlation.clamp(1e-9)

def get_cross_ncut_loss(feat, feat_n, segment, ncut_thr, epsilon, segment_n=None):
    feat_cross_correlation = flatten_and_get_correlation2feat(feat, feat_n)

    weight_ones = torch.ones_like(feat_cross_correlation)
    weight_matrix = torch.where(feat_cross_correlation > float(ncut_thr),
                                weight_ones, epsilon * weight_ones
                                )

    if segment_n is not None:
        if segment.mean() > 0.5:
            segment = 1 - (1 - segment) * (1 - segment_n)
        else:
            segment = segment * segment_n

        segment = segment.clamp(min=epsilon, max=1-epsilon)

    w_a_b = segment @ weight_matrix @ (1 - segment).transpose(1, 2)
    w_a_v = (segment @ weight_matrix).sum((1, 2), keepdim=True)
    w_b_v = ((1 - segment) @ weight_matrix).sum((1, 2), keepdim=True)

    ncut_loss1 = w_a_b / w_a_v.clamp(min=1e-6)
    ncut_loss2 = w_a_b / w_b_v.clamp(min=1e-6)
    mask_loss = torch.mean(ncut_loss1 + ncut_loss2)

    return mask_loss


def get_ncut_loss(feat, segment, ncut_thr, epsilon, iters):
    b, d, h, w = feat.shape
    segment_flat = segment.reshape(b, 1, h * w).clamp(min=epsilon, max=1 - epsilon)
    feature_flat = feat.reshape(b, d, h * w)

    feat_cross_correlation = flatten_and_get_correlation(feature_flat)

    weight_ones = torch.ones_like(feat_cross_correlation)
    weight_matrix = torch.where(feat_cross_correlation > float(ncut_thr),
                                weight_ones, epsilon * weight_ones
                                )

    w_a_b = segment_flat @ weight_matrix @ (1 - segment_flat).transpose(1, 2)
    w_a_v = (segment_flat @ weight_matrix).sum((1, 2), keepdim=True)
    w_b_v = ((1 - segment_flat) @ weight_matrix).sum((1, 2), keepdim=True)

    ncut_loss1 = w_a_b / w_a_v.clamp(min=1e-6)
    ncut_loss2 = w_a_b / w_b_v.clamp(min=1e-6)
    mask_loss = torch.mean(ncut_loss1 + ncut_loss2)

    return mask_loss

def get_ncut_loss_instance(feat, segment, ncut_thr, num_segments, epsilon, iters):
    b, d, h, w = feat.shape
    segment_flat = segment.reshape(b, num_segments, 1, h * w).clamp(min=epsilon, max=1 - epsilon)
    feature_flat = feat.reshape(b, d, h * w)

    feat_cross_correlation = flatten_and_get_correlation(feature_flat)
    mask_inv_cumprod = torch.ones(b, 2, 1, h * w, device=segment_flat.device)
    # mask_inv_cumprod = torch.cumprod(1 - segment_flat[:, :-1], dim=1)
    if num_segments > 2:
        mask_inv_cumprod = torch.cat(
            [mask_inv_cumprod, torch.cumprod(1 - segment_flat[:, :-2], dim=1)], dim=1)
    mask_inv_cumprod_corr = torch.einsum('bkld, bklc -> bkdc', mask_inv_cumprod, mask_inv_cumprod)
    feat_cross_correlation_extended = feat_cross_correlation.unsqueeze(1) * mask_inv_cumprod_corr

    weight_ones = torch.ones_like(feat_cross_correlation_extended)
    weight_matrix = torch.where(feat_cross_correlation_extended > float(ncut_thr),
                                weight_ones, epsilon * weight_ones
                                )

    w_a_b = segment_flat @ weight_matrix @ (1 - segment_flat).transpose(-1, -2)
    w_a_v = (segment_flat @ weight_matrix).sum((-1, -2), keepdim=True)
    w_b_v = ((1 - segment_flat) @ weight_matrix).sum((-1, -2), keepdim=True)

    ncut_loss1 = w_a_b / w_a_v.clamp(min=1e-6)
    ncut_loss2 = w_a_b / w_b_v.clamp(min=1e-6)
    mask_loss = torch.mean(ncut_loss1 + ncut_loss2)

    return mask_loss

def create_movement_matrix(size=5, device='cuda'):
    center = (size - 1) // 2

    # Create a 5x5 matrix with relative movement values
    movement_matrix = torch.zeros((size, size, 2), device=device)

    for i in range(size):
        for j in range(size):
            movement_matrix[i, j, 0] = i - center
            movement_matrix[i, j, 1] = j - center

    movement_matrix = movement_matrix.reshape(1, 1, size**2, 2)

    return movement_matrix

MOVEMENT_MATRIX5 = create_movement_matrix(5)

def get_mincut_loss(img, segment, coords, args, sigma_coord=0.012, sigma_w=1.0):
    num_ch = img.shape[1]

    # Convert 1D indices to 2D spatial coordinates
    row_coordinates = torch.div(coords, args.img_size, rounding_mode='trunc')
    col_coordinates = torch.remainder(coords, args.img_size)
    segment = torch.gather(segment, -1, coords)
    img_sampled = torch.gather(img, -1, coords.repeat(1, num_ch, 1)).permute(0, 2, 1)
    spatial_coordinates = torch.cat([row_coordinates, col_coordinates], dim=1).permute(0, 2, 1) / (args.img_size - 1)
    spatial_coordinates = 2 * spatial_coordinates - 1
    pairwise_distances = (spatial_coordinates.unsqueeze(1) - spatial_coordinates.unsqueeze(2)).pow(2).sum(-1)
    weight_matrix_coord = torch.exp(-pairwise_distances / sigma_coord) # sigma is 5 / 224

    pairwise_img_distances = (img_sampled.unsqueeze(1) - img_sampled.unsqueeze(2)).pow(2).sum(-1)
    weight_matrix_img = torch.exp(-pairwise_img_distances / sigma_w)  # sigma is 5 / 224

    weight_matrix = weight_matrix_img * weight_matrix_coord

    w_a_b = segment @ weight_matrix @ (1 - segment).transpose(1, 2)
    w_a_v = (segment @ weight_matrix).sum((1, 2), keepdim=True)
    w_b_v = ((1 - segment) @ weight_matrix).sum((1, 2), keepdim=True)

    ncut_loss1 = w_a_b / w_a_v.clamp(min=1e-6)
    ncut_loss2 = w_a_b / w_b_v.clamp(min=1e-6)
    mask_loss = torch.mean(ncut_loss1 + ncut_loss2)

    return mask_loss

def get_img_ncut_loss(img, segment_pred):
    img_dist1 = (img[:, :, :, :-1] - img[:, :, :, 1:]).pow(2).sum(1, keepdims=True)
    img_dist2 = (img[:, :, :-1, :] - img[:, :, 1:, :]).pow(2).sum(1, keepdims=True)
    img_dist3 = (img[:, :, :-1, :-1] - img[:, :, 1:, 1:]).pow(2).sum(1, keepdims=True)
    img_dist3 = (img[:, :, :-1, :-1] - img[:, :, 1:, 1:]).pow(2).sum(1, keepdims=True)


    return img.pow(2).mean()

def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)

def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)

def get_corr_loss(code_corr, feat, shift=0):
    with torch.no_grad():
        feat_correlation = flatten_and_get_correlation(feat)
    assert code_corr.shape == feat_correlation.shape
    # loss = - code_corr.clamp(0, .8) * (feat_correlation - shift)
    loss = - code_corr * (feat_correlation - shift)
    loss_mask = (code_corr.ge(0) | (feat_correlation - shift).ge(0)).float()
    loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-6)

    return loss.mean()

def sample_coords(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

def get_mask_adj(height, width, device):
    mask_adj = torch.zeros(1, height * width, height * width, device=device)
    for i in range(height * width):
        row, col = divmod(i, height)
        adj_indices = [(row + dr, col + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if
                       0 <= row + dr < height and 0 <= col + dc < width]
        adj_indices = [r * height + c for r, c in adj_indices]
        mask_adj[:, i, adj_indices] = 1

    return mask_adj

def decide_foreground(segment_pred, patch_size):
    segment_small = F.interpolate(segment_pred, scale_factor=1/patch_size, mode='bilinear')
    corners_mean = (segment_small[:, :, 0, 0] + segment_small[:, :, 0, -1] +
                    segment_small[:, :, -1, 0] + segment_small[:, :, -1, -1]) / 4

    segment_pred_new = torch.zeros_like(segment_pred)

    for i in range(segment_pred.shape[0]):
        if corners_mean[i].mean().item() > 0.5:
            segment_pred_new[i] = 1 - segment_pred[i]
        else:
            segment_pred_new[i] = segment_pred[i]

    return segment_pred_new


class myLossWrapper:
    def __init__(self, args):
        self.loss_weight_dict = {
            'mask_loss': args.weight_mask, 'mask_loss_img': args.weight_reg_img,
            'mask_loss_feat': args.weight_reg_feat, 'loss_entropy': args.weight_entropy,
            'mask_loss_s': args.weight_mask_s, 'loss_consistency': args.weight_const
        }

        self.loss_logging_dict = {
            'loss_entropy': 'Entropy Loss', 'mask_loss': 'NCut Loss',
            'mask_loss_img': 'Reg Img Loss', 'mask_loss_feat': 'Reg Feat Loss',
            'mask_loss_s': 'NCut-S Loss', 'loss_consistency': 'Const Loss'
        }

    def get_loss_print_str(self, input_dict, cur_epoch, total_epochs, cur_i, loader_len):
        print_string_list = [f'Epoch: [{cur_epoch}]/[{total_epochs}], Iter: [{cur_i}]/[{loader_len}]']
        print_string_list += [f'{self.loss_logging_dict[k]}: {float(v): .04f}' for k, v in input_dict.items()]
        return ', '.join(print_string_list)

    def get_loss_from_dict(self, input_dict):
        return sum([self.loss_weight_dict[k] * v for k, v in input_dict.items()])

    def get_metric_log_dict(self, input_dict):
        return {self.loss_logging_dict[k]: float(v) for k, v in input_dict.items()}






