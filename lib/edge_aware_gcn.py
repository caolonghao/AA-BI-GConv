import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv


class GCN_layer(nn.Module):
    def __init__(self, num_state):
        super(GCN_layer, self).__init__()
        self.num_state = num_state
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()
        seg_similar = torch.bmm(seg, aj)
        out = self.relu(self.conv2(seg_similar))
        output = out + seg

        return output


class APPNP(nn.Module):
    def __init__(self, num_s, hidden_dim=3, alpha=0.3, depth=3) -> None:
        super(APPNP, self).__init__()
        # self. = num_state
        self.num_s = num_s
        self.mlp = nn.Sequential(
            nn.Linear(num_s, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_s),
        )
        self.relu = nn.ReLU()
        self.alpha = alpha
        self.depth = depth

    def forward(self, seg, adj):
        n, c, h, w = seg.size()
        seg = seg.view(n, -1, self.num_s).contiguous()

        transformed_seg = self.relu(self.mlp(seg))
        output = transformed_seg.contiguous()
        for _ in range(self.depth):
            output = (1 - self.alpha) * torch.bmm(
                adj, output
            ) + self.alpha * transformed_seg

        # 需要残差吗？
        return output


class GAT(nn.Module):
    def __init__(
        self, num_s, hidden_dim=3, num_head=3, depth=2, activation=nn.ELU()
    ) -> None:
        super(GAT, self).__init__()
        self.num_s = num_s
        self.gat_layers = nn.ModuleList()
        self.depth = depth
        self.activation = activation

        self.gat_layers.append(
            GATConv(
                in_feats=num_s,
                out_feats=hidden_dim,
                num_heads=num_head,
                feat_drop=0.3,
                attn_drop=0.3,
                residual=True,
                activation=self.activation,
            )
        )
        for _ in range(1, depth - 1):
            self.gat_layers.append(
                GATConv(
                    in_feats=hidden_dim * num_head,
                    out_feats=hidden_dim,
                    num_heads=num_head,
                    feat_drop=0.3,
                    attn_drop=0.3,
                    residual=True,
                    activation=self.activation,
                )
            )

        self.gat_layers.append(
            GATConv(
                in_feats=hidden_dim * num_head,
                out_feats=num_s,
                num_heads=num_head,
                feat_drop=0.3,
                attn_drop=0.3,
                residual=True,
            )
        )

    def forward(self, seg, adj):
        n, c, h, w = seg.size()
        seg = seg.view(n, -1, self.num_s).contiguous()

        # kNN稀疏化，否则对于GAT而言边数过多，且无强度区分
        a, _ = adj.topk(k=h * w // 6, dim=2)
        adj_min = torch.min(a, dim=-1).values
        adj_min = adj_min.unsqueeze(-1).repeat(1, 1, adj.shape[-1])
        ge = torch.ge(adj, adj_min)
        zeros = torch.zeros_like(adj)
        adj = torch.where(ge, adj, zeros)

        output_list = []
        for index in range(n):
            seg_now, adj_now = seg[index], adj[index]
            # print("\nseg_now.shape:", seg_now.shape, "adj_now.shape:", adj_now.shape)
            u, v = torch.where(adj_now > 0)
            g = dgl.graph((u, v))
            # 需要实验一下这里是否需要添加自环
            # g = dgl.add_self_loop(g)
            fea = seg_now
            for l in range(self.depth - 1):
                fea = self.gat_layers[l](g, fea).flatten(1)

            fea = self.gat_layers[-1](g, fea).mean(1)

            output_list.append(fea)

        output = torch.stack(output_list, dim=0)
        return output.view(n, -1, self.num_s)


# 思路来自 SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks
class Adj_Process(nn.Module):
    def __init__(self, F=nn.ReLU()) -> None:
        super(Adj_Process, self).__init__()
        self.F = F

    def forward(self, adj, epsilon):
        # 数值非负化
        # adj = self.F(adj)

        # 稀疏化
        adj[adj < epsilon] = 0

        # 对称化
        adj = (adj + torch.transpose(adj, 1, 2)) / 2

        # 归一化, adj 为加入了自环的邻接矩阵
        adj = (
            adj + torch.diag_embed(torch.ones(adj.shape[0], adj.shape[1])).cuda()
        )  # 加入自环，可能可以不要
        degree_mat = torch.sum(adj, dim=2)
        degree_mat = torch.diag_embed(torch.pow(degree_mat, -0.5))
        degree_mat[torch.isinf(degree_mat)] = 0
        adj = torch.bmm(degree_mat, adj)
        adj = torch.bmm(adj, degree_mat)

        return adj


"""
num_s 1x1 卷积后每个点的维度
"""


class EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(EAGCN, self).__init__()
        self.num_in = num_in
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids, mids))

        self.gcn = GCN_layer(num_state=num_in)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)
        self.adj_process = Adj_Process()

    def forward(self, seg_ori, edge_ori):
        #
        epsilon = 0.2

        seg = seg_ori
        edge = edge_ori
        n, c, h, w = seg.size()

        seg_s = self.conv_s1(seg)
        theta_T = seg_s.view(n, self.num_s, -1).contiguous()
        theta = seg_s.view(n, -1, self.num_s).contiguous()

        # print("seg.shape:", seg.shape)
        # print("theta.shape:", theta.shape)

        channel_att = torch.relu(
            self.mlp(self.maxpool_c(seg).squeeze(3).squeeze(2))
        ).view(n, self.num_s, -1)
        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s))

        similarity_c = torch.bmm(theta, diag_channel_att)
        similarity_c = torch.bmm(similarity_c, theta_T)

        similarity_c = self.adj_process(similarity_c, epsilon=0.15)
        # 每个节点可以去到多个点，softmax不合理
        # similarity_c = self.softmax(torch.bmm(similarity_c, theta_T))

        seg_c = self.conv_s11(seg)
        sigma = seg_c.view(n, self.num_s, -1).contiguous()
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)

        edge_m = seg * edge

        maxpool_s, _ = torch.max(seg, dim=1)
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        seg_ss = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)
        edge_mm = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)

        diag_spatial_att = torch.bmm(edge_mm, seg_ss)
        similarity_s = sigma_out * diag_spatial_att

        similarity_s = self.adj_process(similarity_s, epsilon=0.15)
        # 同之前对similarity_c的处理
        # similarity_s = self.softmax(diag_spatial_att)
        similarity = similarity_c + similarity_s

        # 这里直接相加，边权有些可能比较极端，例如大于1或者非常接近0，可以归一化然后阈值卡一下
        similarity = self.adj_process(similarity, epsilon=0.0)

        seg_gcn = self.gcn(seg, similarity).view(n, self.num_in, self.mids, self.mids)

        # 注意原版 bi-gcn 这里是有残差项的
        # ext_up_seg_gcn = seg_gcn + seg_ori
        ext_up_seg_gcn = seg_gcn
        return ext_up_seg_gcn, similarity


class AG_EAGCN(nn.Module):
    def __init__(
        self,
        num_in,
        plane_mid,
        mids,
        alpha=0.5,
        epsilon=0.2,
        postgnn="APPNP",
        postgnn_depth=3,
        aggregation_mode="mean",
        prop_nums=3,
    ) -> None:
        super(AG_EAGCN, self).__init__()

        self.eagcn = EAGCN(num_in, plane_mid, mids)
        self.adj_process = Adj_Process()
        self.prop_nums = prop_nums
        self.aggregation_mode = aggregation_mode

        if self.aggregation_mode == "attention":
            self.agg_conv = nn.Conv1d(self.prop_nums, self.prop_nums, kernel_size=1)

        if postgnn == "APPNP":
            self.post_gnn = APPNP(num_s=num_in, depth=postgnn_depth, alpha=alpha)
        elif postgnn == "GAT":
            self.post_gnn = GAT(num_s=num_in, depth=postgnn_depth)
        elif postgnn == "GCN":
            self.post_gnn = GCN(
                num_state=num_in,
            )

    def forward(self, seg, edge):
        sample_num, c, h, w = seg.size()

        adj_list = []
        ori_seg = seg
        for i in range(self.prop_nums):
            seg, adj = self.eagcn(seg, edge)
            adj_list.append(adj)

        if self.aggregation_mode == "mean":
            adj = sum(adj_list) / len(adj_list)
        elif self.aggregation_mode == "sum":
            adj = sum(adj_list)
        elif self.aggregation_mode == "attention":
            # NOTE: not completed yet
            adj = torch.stack(adj_list, dim=1)

        # 聚合后的邻接矩阵如用GAT必须一定程度稀疏化，否则无法反映重复出现的边的可信度

        # 稀疏化阈值
        post_epsilon = 0.0
        adj = self.adj_process(adj, post_epsilon)

        # Graph Regulation - Idea from: Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings
        _seg = seg.reshape(-1, h * w, c)
        degree_mat = torch.sum(adj, dim=2)
        degree_mat = torch.diag_embed(degree_mat)
        # print("_seg.shape:", _seg.shape, "adj.shape:", adj.shape, "degree_mat.shape:", degree_mat.shape)
        Laplacian_mat = degree_mat - adj
        _seg_T = torch.transpose(_seg, 2, 1)
        smoothness_mat = torch.bmm(_seg_T, Laplacian_mat)
        smoothness_mat = torch.bmm(smoothness_mat, _seg)
        dirichlet_energy = torch.trace(adj[0])
        for i in range(1, adj.shape[0]):
            dirichlet_energy += torch.trace(adj[i])

        gamma = 0.1
        beta = 0.1
        alpha = 0.1
        n = adj.shape[1]
        ones = torch.ones((sample_num, n, 1)).cuda()
        ones_T = torch.ones((sample_num, 1, n)).cuda()
        # print("adj.shape:", adj.shape, "ones.shape:", ones.shape)
        f_A = -beta * torch.bmm(
            ones_T, torch.log(torch.bmm(adj, ones))
        ) / n + gamma * torch.norm(adj) / (n**2)

        # dirichlet_energy 控制同质性的满足，f_A 控制不出现 A=0
        graph_regulation = alpha * dirichlet_energy + f_A

        output = self.post_gnn(ori_seg, adj)

        output = output.reshape(-1, c, h, w)
        return output, graph_regulation


class GRU_EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids):
        super(GRU_EAGCN, self).__init__()

        self.eagcn = EAGCN(num_in, plane_mid, mids)
        self.rnn = torch.nn.GRU(input_size=1024, hidden_size=1024, num_layers=1)

    def forward(self, seg, edge):
        _, c, h, w = seg.size()
        # ------------t0-------#
        updated_seg, _ = self.eagcn(seg, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_0, h_0 = self.rnn(updated_seg)
        # ------------t1-------#
        output = output_0.view(-1, c, h, w)
        updated_seg, _ = self.eagcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_1, h_1 = self.rnn(updated_seg, h_0)
        # -------------t2----------#
        output = output_1.view(-1, c, h, w)
        updated_seg, _ = self.eagcn(output, edge)
        updated_seg = updated_seg.view(c, -1, h * w)
        output_2, h_2 = self.rnn(updated_seg, h_1)

        # reshape back
        output_2 = output_2.view(-1, c, h, w)

        return output_2
