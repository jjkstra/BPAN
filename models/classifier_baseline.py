import torch.nn as nn
from modules import FewShotClassifier
from utils import l2_distance, cosine_distance
from setup import setup_num_class


class Classifier(FewShotClassifier):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distance = cosine_distance if args.metric == 'cos' else l2_distance
        self.fc = nn.Linear(640, setup_num_class(args.dataset))

    def forward(self, images):
        features = self.extract_features(images)
        if self.training:
            return self.fc(features)
        else:
            k = self.args.way * self.args.shot
            data_shot, data_query = features[:k], features[k:]
            return self.forward_1shot(data_shot, data_query)

    def forward_1shot(self, proto, query):
        return self.distance(query, proto)
