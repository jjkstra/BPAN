import torch
import torch.nn.functional as F
from torch import nn
from modules import FewShotClassifier
from utils import l2_distance, cosine_distance
from models.attention import MultiHeadAttention
from torch.autograd import Variable


class BLAN(FewShotClassifier):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.metric = cosine_distance if args.metric == 'cos' else l2_distance
        self.aggregate = nn.Linear(args.n_patch_views, 1)
        self.self_attention = MultiHeadAttention(8, 640, 640, 640)
        self.cross_attention = MultiHeadAttention(8, 640, 640, 640)
        self.weight = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)

    def forward(self, images):
        features = self.extract_features(images)
        k = self.args.way * self.args.shot
        data_shot, data_query = features[:k], features[k:]
        # if self.args.shot > 1:
        #     data_shot = self.get_sfc(data_shot)
        return self.forward_1shot(data_shot, data_query)

    def forward_1shot(self, support_patch_prototypes, query_patch_features):
        support_patch_prototypes = self.self_attention(
            support_patch_prototypes, support_patch_prototypes, support_patch_prototypes)
        support_aggregated_prototypes = self.gather_features(support_patch_prototypes)

        query_patch_features = self.self_attention(
            query_patch_features, query_patch_features, query_patch_features)
        query_aggregated_features = self.gather_features(query_patch_features)

        # batch_size = self.way * self.args.query
        batch_size = query_patch_features.shape[0]

        support_patch_prototypes = support_patch_prototypes.unsqueeze(0).expand(batch_size, -1, -1, -1)
        support_to_query_features = self.gather_features(self.cross_attention(
            support_patch_prototypes.reshape(batch_size, -1, self.emb_dim),
            query_patch_features,
            query_patch_features).reshape(
            batch_size,
            self.way * self.args.shot,
            self.args.n_patch_views,
            self.emb_dim))

        query_patch_features = query_patch_features.unsqueeze(1).expand(-1, self.way * self.args.shot, -1, -1)
        query_to_support_features = self.gather_features(self.cross_attention(
            query_patch_features.reshape(-1, self.args.n_patch_views, self.emb_dim),
            support_patch_prototypes.reshape(-1, self.args.n_patch_views, self.emb_dim),
            support_patch_prototypes.reshape(-1, self.args.n_patch_views, self.emb_dim)).reshape(
            batch_size,
            self.way * self.args.shot,
            self.args.n_patch_views,
            self.emb_dim))

        support_to_query_prototypes = self.compute_prototypes(support_to_query_features)
        query_to_support_prototypes = self.compute_prototypes(query_to_support_features)

        support_aggregated_prototypes = support_aggregated_prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        sim_query_to_support = self.compute_distance(query_to_support_prototypes, support_aggregated_prototypes)

        query_aggregated_features = query_aggregated_features.unsqueeze(1)
        sim_support_to_query = self.metric(query_aggregated_features, support_to_query_prototypes).squeeze()
        if self.args.metric == 'cos':
            sim_support_to_query /= self.args.tau

        return self.weight * sim_support_to_query + (1 - self.weight) * sim_query_to_support


    def compute_distance(self, a, b):
        distance = torch.stack([_.diag() for _ in self.metric(a, b)])
        if self.args.metric == 'cos':
            distance /= self.args.tau
        return distance

    def gather_features(self, features):
        features = features.transpose(-2, -1)
        features = self.aggregate(features).squeeze(-1)
        return features

    def get_sfc(self, support):
        SFC = support.view(self.args.shot, -1, self.args.n_patch_views, 640).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)
        optimizer = torch.optim.Adam([SFC], lr=0.001)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot).to(self.device)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot)
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC
