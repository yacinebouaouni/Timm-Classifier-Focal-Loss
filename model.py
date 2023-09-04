import timm
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, model_name, num_classes=1):
        super(Classifier, self).__init__()

        # Load the specified model from timm and ensure it's not pretrained
        model = timm.create_model(model_name, pretrained=False)

        # The input images are grayscale 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final classification layer for binary classification
        num_ftrs = model.fc.in_features if hasattr(model, "fc") else model.num_features
        setattr(model, "fc", nn.Linear(num_ftrs, num_classes))

        self.model = model
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax()

    def forward(self, x):
        return self.activation(self.model(x))
