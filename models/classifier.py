import torch.nn as nn
from models.attention import MultiHeadAttention
from modules import FewShotClassifier
from utils import l2_distance, cosine_distance
from setup import setup_num_class


class Classifier(FewShotClassifier):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = MultiHeadAttention(8, 640, 640, 640)
        self.aggregate = nn.Linear(args.n_patch_views, 1)
        self.distance = cosine_distance if args.distance == 'cos' else l2_distance
        self.fc = nn.Linear(640, setup_num_class(args.dataset))

    def forward(self, images):
        features = self.extract_features(images)
        if self.training:
            return self.fc(self.gather_features(self.attention(features, features, features))) \
                if self.args.do_random_crop else self.fc(features)
        else:
            k = self.args.way * self.args.shot
            data_shot, data_query = features[:k], features[k:]
            return self.forward_1shot(data_shot, data_query)

    def forward_1shot(self, proto, query):
        if self.args.do_random_crop:
            proto = self.gather_features(self.attention(proto, proto, proto))
            query = self.gather_features(self.attention(query, query, query))
        return self.distance(query, proto)

    def gather_features(self, features):
        features = features.transpose(-2, -1)
        features = self.aggregate(features).squeeze(-1)
        return features
