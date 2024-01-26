import torch
import torch.nn as nn
import torchvision.models as models

class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor, positive, negative):
        ap_distance = torch.sum((anchor - positive) ** 2, dim=-1)
        an_distance = torch.sum((anchor - negative) ** 2, dim=-1)
        return ap_distance, an_distance
    

class L2Normalize(nn.Module):
    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.distance_layer = DistanceLayer()

    def forward(self, anchor, positive, negative):
        encoded_a = self.encoder(anchor)
        encoded_p = self.encoder(positive)
        encoded_n = self.encoder(negative)
        ap_distance, an_distance = self.distance_layer(encoded_a, encoded_p, encoded_n)
        return ap_distance, an_distance


class SiameseModel(nn.Module):
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin

    def forward(self, anchor, positive, negative):
        
        ap_distance, an_distance = self.siamese_network(anchor, positive, negative)
        
        return ap_distance, an_distance
    

def load_pretrained_model(input_shape):
    """
    This function load resnet50 CNN pretrained model and replace the mlp by our custom mlp
    """
    pretrained_model = models.resnet50(pretrained=True)

    
    for param in list(pretrained_model.parameters())[:-27]:
        param.requires_grad = False

    
    features_dim = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(features_dim, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        L2Normalize()
    )
    
    return pretrained_model


def triplet_loss(ap_distance, an_distance):
    """
    Compute the triplet loss based on ap_distance and an_distance
    """
    loss = ap_distance - an_distance + 0.5
    loss = torch.max(loss, torch.zeros_like(loss))
    return loss.mean()
