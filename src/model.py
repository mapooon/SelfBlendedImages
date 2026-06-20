import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from utils.sam import SAM

class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x, return_features=False):
        # Feature: Added toggle to extract raw embedding features before the classifier head
        if return_features:
            return self.net.extract_features(x)
        return self.net(x)
    
    def training_step(self, x, target):
        for i in range(2):
            pred_cls = self(x)
            if i == 0:
                pred_first = pred_cls
            
            loss = self.cel(pred_cls, target)
            self.optimizer.zero_grad()
            loss.backward()
            
            # Dynamically select the SAM step based on the loop index
            step_fn = self.optimizer.first_step if i == 0 else self.optimizer.second_step
            step_fn(zero_grad=True)
        
        # Feature: Now calculates and returns batch accuracy alongside the predictions
        acc = (pred_first.argmax(dim=-1) == target).float().mean()
        return pred_first, loss, acc
        
        return pred_first
    
