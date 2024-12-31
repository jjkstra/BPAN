import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from abc import abstractmethod
from typing import Optional

from modules.resnet import resnet12
from setup import setup_device


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
    ):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: the feature extractor used by the method. Must output a tensor of the
                appropriate shape (depending on the method).
                If None is passed, the backbone will be initialized as nn.Identity().
            use_softmax: whether to return predictions as soft probabilities
            feature_centering: a features vector on which to center all computed features.
                If None is passed, no centering is performed.
            feature_normalization: a value by which to normalize all computed features after centering.
                It is used as the p argument in torch.nn.functional.normalize().
                If None is passed, no normalization is performed.
        """
        super().__init__()
        self.backbone = backbone if backbone else resnet12()
        self.way = 5
        self.emb_dim = 640
        self.device = setup_device()

    @abstractmethod
    def forward(
        self,
        images,
    ) -> Tensor:
        """
        Predict classification labels.
        Args:
            images: images of an episode
        Returns:
            a prediction of classification scores
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    def extract_features(self, images):
        if images.shape.__len__() == 5:
            batch_size, n_patch = images.shape[:2]
            images = images.view(-1, images.shape[2], images.shape[3], images.shape[4])
            features = self.backbone(images)
            features = F.adaptive_avg_pool2d(features, 1).squeeze()
            features = features.view(-1, n_patch, features.shape[-1])

        else:
            features = self.backbone(images)
            features = F.adaptive_avg_pool2d(features, 1).squeeze()

        return features

    def compute_prototypes(self, support_features):
        return support_features.view(-1, self.way, self.emb_dim).mean(dim=0)
