import math
from scipy.ndimage import gaussian_filter

import numpy as np

import torch
from torch.nn import functional as F


def cal_anomaly_maps(ft_list, fs_list, out_size=[256, 256], uni_am=True, use_cos=True, amap_mode='add', gaussian_sigma=4, weights=None):
    # ft_list = [f.cpu() for f in ft_list]
    # fs_list = [f.cpu() for f in fs_list]
    bs = ft_list[0].shape[0]
    weights = weights if weights else [1] * len(ft_list)
    anomaly_map = np.ones(
        [bs] + out_size) if amap_mode == 'mul' else np.zeros([bs] + out_size)
    a_map_list = []
    if uni_am:
        size = (ft_list[0].shape[2], ft_list[0].shape[3])
        for i in range(len(ft_list)):
            ft_list[i] = F.interpolate(F.normalize(
                ft_list[i], p=2), size=size, mode='bilinear', align_corners=True)
            fs_list[i] = F.interpolate(F.normalize(
                fs_list[i], p=2), size=size, mode='bilinear', align_corners=True)
        ft_map, fs_map = torch.cat(ft_list, dim=1), torch.cat(fs_list, dim=1)
        if use_cos:
            a_map = 1 - F.cosine_similarity(ft_map, fs_map, dim=1)
            a_map = a_map.unsqueeze(dim=1)
        else:
            a_map = torch.sqrt(
                torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True))
        a_map = F.interpolate(a_map, size=out_size,
                              mode='bilinear', align_corners=True)
        a_map = a_map.squeeze(dim=1).cpu().detach().numpy()
        anomaly_map = a_map
        a_map_list.append(a_map)
    else:
        for i in range(len(ft_list)):
            ft = ft_list[i]
            fs = fs_list[i]
            # fs_norm = F.normalize(fs, p=2)
            # ft_norm = F.normalize(ft, p=2)
            if use_cos:
                a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
                a_map = a_map.unsqueeze(dim=1)
            else:
                a_map = torch.sqrt(
                    torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
            a_map = F.interpolate(a_map, size=out_size,
                                  mode='bilinear', align_corners=True)
            a_map = a_map.squeeze(dim=1)
            a_map = a_map.cpu().detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'add':
                anomaly_map += a_map * weights[i]
            else:
                anomaly_map *= a_map
        if amap_mode == 'add':
            anomaly_map /= (len(ft_list) * sum(weights))
    if gaussian_sigma > 0:
        for idx in range(anomaly_map.shape[0]):
            anomaly_map[idx] = gaussian_filter(
                anomaly_map[idx], sigma=gaussian_sigma)
    return anomaly_map, a_map_list


def cal_anomaly_maps_dino(fs_list, ft_list, out_size=256):
    # _ = cal_anomaly_maps(fs_list, ft_list)
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size,
                              mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(
        dim=1, keepdim=True)
    # anomaly_map = anomaly_map.squeeze().cpu().detach().numpy()
    # for idx in range(anomaly_map.shape[0]):
    #     anomaly_map[idx] = gaussian_filter(
    #         anomaly_map[idx], sigma=4)
    return anomaly_map, a_map_list[0]


def cal_anomaly_maps_dino_2(fs_list, ft_list, out_size=256, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size,
                              mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * \
                (a_map - norm_factor[0][i]) / \
                (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def cal_anomaly_map_single(fs_list, ft_list, out_size=256, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size,
                              mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * \
                (a_map - norm_factor[0][i]) / \
                (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    # convert it to tensor
    anomaly_map = torch.tensor(anomaly_map).float().unsqueeze(dim=0).to('cuda')
    return anomaly_map, a_map_list


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
        torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) /
        (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
