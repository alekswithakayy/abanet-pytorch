import torch
import torch.nn as nn
import torch.nn.functional as F


class IRN(nn.Module):

    def __init__(self, backbone, args):
        super(IRN, self).__init__()

        self.stage1 = backbone.stage1
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3
        self.stage4 = backbone.stage4
        self.stage5 = backbone.stage5

        # branch: class boundary detection
        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        # branch: displacement field
        self.fc_df1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.fc_df2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_df3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
        )
        self.fc_df4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_df5 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )

        self.fc_df6 = nn.Sequential(
            nn.Conv2d(768, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_df7 = nn.Sequential(
            nn.Conv2d(448, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 1, bias=False),
            IRN.MeanShift(2)
        )

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.df_layers = nn.ModuleList([self.fc_df1, self.fc_df2, self.fc_df3, self.fc_df4, self.fc_df5, self.fc_df6, self.fc_df7])

    class MeanShift(nn.BatchNorm2d):

        def __init__(self, num_features, momentum=0.1):
            super(IRN.MeanShift, self).__init__(num_features, affine=False, momentum=momentum)

        def forward(self, input):
            if self.training:
                super(IRN.MeanShift, self).forward(input).detach()
                return input
            return input - self.running_mean.view(1, 2, 1, 1)


    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()

        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]
        edge_up = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        edge_out = edge_up

        df1 = self.fc_df1(x1)
        df2 = self.fc_df2(x2)
        df3 = self.fc_df3(x3)
        df4 = self.fc_df4(x4)[..., :df3.size(2), :df3.size(3)]
        df5 = self.fc_df5(x5)[..., :df3.size(2), :df3.size(3)]

        df_up3 = self.fc_df6(torch.cat([df3, df4, df5], dim=1))[..., :df2.size(2), :df2.size(3)]
        df_out = self.fc_df7(torch.cat([df1, df2, df_up3], dim=1))

        return edge_out, df_out

    def trainable_parameters(self):
        return (tuple(self.edge_layers.parameters()), tuple(self.df_layers.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


class AffinityDisplacement(IRN):

    def __init__(self, backbone, paths_by_len_indices, ind_from, ind_to, args):
        super(AffinityDisplacement, self).__init__(backbone, args)
        self.num_path_lengths = len(paths_by_len_indices)
        for i in range(self.num_path_lengths):
            param = torch.nn.Parameter(torch.from_numpy(paths_by_len_indices[i]),
                                       requires_grad=False)
            self.register_parameter("paths_of_len_n" + str(i), param)
        self.register_parameter("ind_from",
            torch.nn.Parameter(torch.unsqueeze(ind_from, dim=0), requires_grad=False))
        self.register_parameter("ind_to",
            torch.nn.Parameter(ind_to, requires_grad=False))


    def edge_to_affinity(self, edge):
        edge = edge.view(edge.size(0), -1)

        aff_list = []
        for i in range(self.num_path_lengths):
            ind = self._parameters['paths_of_len_n' + str(i)]
            ind_flat = ind.view(-1)
            # Select values using indices along every path for every point
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            # Find max value along every path
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)

        return torch.cat(aff_list, dim=1)


    def forward(self, x):
        edge_out, df_out = super().forward(x)
        edge_out = torch.sigmoid(edge_out)

        aff_out = self.edge_to_affinity(edge_out)

        return aff_out, df_out


class EdgeDisplacement(IRN):

    def __init__(self, backbone, args):
        super(EdgeDisplacement, self).__init__(backbone, args)

    def forward(self, x, out_settings=None):
        edge_out, df_out = super().forward(x)
        edge_out = torch.sigmoid(edge_out)

        return edge_out, df_out
