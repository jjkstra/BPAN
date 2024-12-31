from modules import FewShotClassifier
from utils import l2_distance, cosine_distance


class PrototypicalNetwork(FewShotClassifier):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.distance = cosine_distance if args.distance == 'cos' else l2_distance

    def forward(self, images):
        features = self.extract_features(images)
        k = self.args.way * self.args.shot
        data_shot, data_query = features[:k], features[k:]
        prototypes = self.compute_prototypes(data_shot)
        return self.distance(data_query, prototypes)
