import torch.nn as nn
import torchvision.models as models

def build_resnet50(num_classes=10):
    """
    Builds ResNet50 with a classification head for STL10.
    """
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)  # Adjust for 10 classes
    return resnet
