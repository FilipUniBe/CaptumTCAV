import torchvision
import torch.nn as nn
class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#change model to 1 channel
        self.densenet121.classifier = nn.Linear(num_ftrs, num_classes)#match model to desired output class number
    def forward(self, x):
        x = self.densenet121(x)
        return x