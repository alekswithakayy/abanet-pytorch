import torch
import torch.nn.functional as F
import numpy as np


class PathIndex:

    def __init__(self, radius=5, size=None):
        self.radius = radius
        self.radius_floor = int(np.ceil(radius) - 1)

        self.paths_by_len, self.path_dst = self.get_paths_by_len(self.radius)
        self.path_dst2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.path_dst).transpose(1, 0), 0), -1).float()

        if size:
            self.paths_by_len_indices, self.src_indices, self.dst_indices \
                = self.get_path_indices(size)


    def get_paths_by_len(self, radius):
        # Collect points lying within a specified radius
        radial_points = []
        for y in range(-radius + 1, radius):
            for x in range(-radius + 1, radius):
                len_sq = x**2 + y**2
                if  len_sq < radius ** 2 and len_sq != 0:
                    radial_points.append((y, x))

        # Collect all points along path to each radial point
        paths_by_len = [[] for _ in range(radius * 4)]
        for (p_y, p_x) in radial_points:
            points_on_path = []
            len_sq = p_y**2 + p_x**2

            min_y, max_y = sorted((0, p_y))
            min_x, max_x = sorted((0, p_x))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Determine distance of point from path line
                    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                    dist_sq = (p_y * x - p_x * y) ** 2 / len_sq
                    if dist_sq < 1:
                        points_on_path.append([y, x])

            points_on_path.sort(key=lambda p: -abs(p[0]) - abs(p[1]))
            path_len = len(points_on_path)
            paths_by_len[path_len].append(points_on_path)

        paths_by_len = [np.asarray(path) for path in paths_by_len if path]
        paths_dst = np.concatenate([path[:, 0] for path in paths_by_len], axis=0)

        return paths_by_len, paths_dst


    def get_path_indices(self, size):
        r = self.radius_floor
        h, w = size
        indices = np.reshape(np.arange(0, h * w, dtype=np.int64), (h, w))

        h_cropped = h - 2 * r
        w_cropped = w - 2 * r

        # Get index for each point along each path, across all source points
        paths_by_len_indices = []
        for paths_of_len_n in self.paths_by_len:
            paths_of_len_n_indices = []
            for path in paths_of_len_n:
                path_indices = []
                for dy, dx in path:
                    # Get index for path point relative to every src point at once
                    point_indices = indices[r + dy:r + dy + h_cropped,
                                            r + dx:r + dx + w_cropped]
                    point_indices = np.reshape(point_indices, [-1])
                    path_indices.append(point_indices)
                paths_of_len_n_indices.append(np.stack(path_indices))
            paths_by_len_indices.append(np.stack(paths_of_len_n_indices))
        paths_by_len_indices = np.array(paths_by_len_indices)
        src_indices = np.reshape(indices[r:r + h_cropped, r:r + w_cropped], -1)
        dest_indices = np.concatenate([path[:,0] for path in paths_by_len_indices], axis=0)
        return paths_by_len_indices, src_indices, dest_indices


    def to_displacement(self, x):
        r = self.radius_floor
        h, w = x.size(2), x.size(3)
        h_cropped = h - 2 * r
        w_cropped = w - 2 * r

        feat_src = x[:, :, r:r + h_cropped, r:r + w_cropped]
        feat_dst = []
        for dy, dx in self.path_dst:
            feat_dst.append(x[:, :, r + dy:r + dy + h_cropped, r + dx:r + dx + w_cropped])
        feat_dst = torch.stack(feat_dst, 2)
        disp = torch.unsqueeze(feat_src, 2) - feat_dst
        disp = disp.view(disp.size(0), disp.size(1), disp.size(2), -1)
        return disp


    def to_displacement_loss(self, x):
        return torch.abs(x - self.path_dst2.cuda())


def edge_to_affinity(edge, paths_indices):
    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(len(paths_indices)):
        if isinstance(paths_indices[i], np.ndarray):
            paths_indices[i] = torch.from_numpy(paths_indices[i])
        paths_indices[i] = paths_indices[i].cuda(non_blocking=True)

    for ind in paths_indices:
        ind_flat = ind.view(-1)
        dist = torch.index_select(edge, dim=-1, index=ind_flat)
        dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
        aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
        aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)

    return aff_cat


def feature_to_affinity(x, ind_from, ind_to):
    x = x.view(x.size(0), x.size(1), -1)

    if isinstance(ind_from, np.ndarray):
        ind_from = torch.from_numpy(ind_from)
        ind_to = torch.from_numpy(ind_to)

    ind_from = torch.unsqueeze(ind_from, dim=0)

    ff = x[..., ind_from.cuda(non_blocking=True)]
    ft = x[..., ind_to.cuda(non_blocking=True)]

    aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

    return aff


def affinity_sparse2dense(affinity_sparse, ind_from, ind_to, n_vertices):
    ind_from = torch.from_numpy(ind_from)
    ind_to = torch.from_numpy(ind_to)

    affinity_sparse = affinity_sparse.view(-1).cpu()
    ind_from = ind_from.repeat(ind_to.size(0)).view(-1)
    ind_to = ind_to.view(-1)

    indices = torch.stack([ind_from, ind_to])
    indices_tp = torch.stack([ind_to, ind_from])

    indices_id = torch.stack([torch.arange(0, n_vertices).long(), torch.arange(0, n_vertices).long()])

    affinity_dense = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                       torch.cat([affinity_sparse, torch.ones([n_vertices]), affinity_sparse])).to_dense().cuda()

    return affinity_dense


def to_transition_matrix(affinity_dense, beta, times):
    scaled_affinity = torch.pow(affinity_dense, beta)

    trans_mat = scaled_affinity / torch.sum(scaled_affinity, dim=0, keepdim=True)
    for _ in range(times):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    return trans_mat


def propagate_to_edge(x, edge, radius=5, beta=10, exp_times=8):
    height, width = x.shape[-2:]

    hor_padded = width+radius*2
    ver_padded = height+radius

    path_index = PathIndex(radius=radius, size=(ver_padded, hor_padded))

    edge_padded = F.pad(edge, (radius, radius, 0, radius), mode='constant', value=1.0)
    sparse_aff = edge_to_affinity(torch.unsqueeze(edge_padded, 0),
                                               path_index.default_path_indices)

    dense_aff = affinity_sparse2dense(sparse_aff, path_index.default_src_indices,
                                      path_index.default_dst_indices, ver_padded*hor_padded)
    dense_aff = dense_aff.view(ver_padded, hor_padded, ver_padded, hor_padded)
    dense_aff = dense_aff[:-radius, radius:-radius, :-radius, radius:-radius]
    dense_aff = dense_aff.reshape(height * width, height * width)

    trans_mat = to_transition_matrix(dense_aff, beta=beta, times=exp_times)

    x = x.view(-1, height, width) * (1 - edge)


    rw = torch.matmul(x.view(-1, height * width), trans_mat)
    rw = rw.view(rw.size(0), 1, height, width)

    return rw
