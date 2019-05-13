import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class InterPixelRelationLoss(nn.Module):
    def __init__(self, pixel_radius=5):
        super(InterPixelRelationLoss, self).__init__()
        self.pixel_radius = pixel_radius

    def forward(self, inputs, targets):
        # Displacement field, boundary detection
        df, bd = inputs

        # Background, foreground and negative target labels
        # bg_label = targets[0].cuda(non_blocking=True)
        # bg_count = torch.sum(bg_label) + 1e-5
        # fg_label = targets[1].cuda(non_blocking=True)
        # fg_count = torch.sum(fg_label) + 1e-5
        # neg_label = targets[2].cuda(non_blocking=True)
        # neg_count = torch.sum(neg_label) + 1e-5

        ###########################
        # Displacement field loss #
        ###########################

        # Get indices of pixel pairs limited by
        size = (df.size(2), df.size(3))
        ind_from, ind_to = get_indices_of_pairs(self.pixel_radius, size)
        ind_to = torch.from_numpy(ind_to).cuda(non_blocking=True)
        ind_from = torch.from_numpy(ind_from).cuda(non_blocking=True)

        pl = get_pixel_locations(size)
        pl = torch.from_numpy(pl).cuda(non_blocking=True)
        pl = pl.view(2, -1)
        lf = torch.index_select(pl, dim=1, index=ind_from)
        lf = torch.unsqueeze(lf, dim=1)
        lt = torch.index_select(pl, dim=1, index=ind_to)
        lt = lt.view(2, -1, lf.size(2))
        delta_hat = torch.unsqueeze(lt - lf, 0)

        df = df.view(df.size(0), df.size(1), -1)
        ff = torch.index_select(df, dim=2, index=ind_from)
        ff = torch.unsqueeze(ff, dim=2)
        ft = torch.index_select(df, dim=2, index=ind_to)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
        delta = ft - ff

        L_D_fg = torch.sum(fg_label * torch.abs(delta - delta_hat)) / fg_count
        L_D_bg = torch.sum(bg_label * delta) / bg_count

        ###########################
        # Boundary detection loss #
        ###########################



def get_indices_of_pairs(radius, size):
    deltas = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            if x * x + y * y < radius * radius and x + y != 0:
                deltas.append((x, y))

    h, w = size
    indices = np.reshape(np.arange(0, h*w, dtype=np.int64), (h, w))

    indices_from = indices[radius:-radius, radius:-radius].flatten()

    indices_to = []
    for dx, dy in deltas:
        start_x = radius + dx
        end_x = radius + (w - 2 * radius) + dx
        start_y = radius + dy
        end_y = radius + (h - 2 * radius) + dy
        indices_to.append(indices[start_y:end_y, start_x:end_x].flatten())
    indices_to = np.concatenate(indices_to, axis=0)

    return indices_from, indices_to


def get_pixel_locations(size):
    pixel_loc = np.zeros((2, size[0], size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            pixel_loc[0,i,j] = i
            pixel_loc[1,i,j] = j
    return pixel_loc
