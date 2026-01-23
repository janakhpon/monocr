import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for OCR.
    Modified ResNet-18 to handle grayscale input and maintain spatial width.
    """
    def __init__(self, output_channels=512):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pretrained resnet18
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify first layer to accept 1 channel (Grayscale)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        # Initialize with average of pretrained weights
        with torch.no_grad():
            self.conv1.weight[:] = backbone.conv1.weight.sum(dim=1, keepdim=True)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Modify strides to preserve width for sequence modeling
        self.layer2[0].conv1.stride = (2, 1)
        self.layer2[0].downsample[0].stride = (2, 1)
        
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)
        
        self.out_channels = 512

    def forward(self, x):
        """
        Extract spatial features from image tensor.
        Args:
            x: Input tensor [B, 1, H, W]
        Returns:
            Feature map [B, 512, H', W']
        """
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
    CRNN architecture for Mon OCR.
    Combines ResNet feature extraction with Bi-LSTM sequence modeling.
    """
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_layers=2, dropout=0.1):
        super(MonOCRModel, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor()
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        self.lstm_input_size = 512 
        
        self.rnn = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Forward pass for OCR.
        Args:
            x: Input image tensor [B, 1, 64, W]
        Returns:
            Logits tensor [B, T, C]
        """
        # x: [B, 1, H, W]
        features = self.feature_extractor(x)
        
        features = self.avg_pool(features)
        features = features.squeeze(2) 
        features = features.permute(0, 2, 1) # [B, T, C]
        
        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(features)
        
        logits = self.fc(rnn_out)
        
        return logits
