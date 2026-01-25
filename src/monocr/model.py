import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ResNetFeatureExtractor(nn.Module):
    """
    Modified ResNet-18 for OCR feature extraction.
    Handles grayscale input and preserves width for sequence modeling.
    """
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:] = backbone.conv1.weight.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Preserve width in later layers
        for layer in [self.layer2, self.layer3, self.layer4]:
            layer[0].conv1.stride = (2, 1)
            layer[0].downsample[0].stride = (2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class MonOCRModel(nn.Module):
    """
    CRNN architecture: ResNet + Bi-LSTM + FC.
    """
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2):
        super(MonOCRModel, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0
        )
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.avg_pool(features).squeeze(2).permute(0, 2, 1)
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(features)
        return self.fc(recurrent)
