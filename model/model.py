import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet



def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


############################################ Define Net Class
class BengaliaiNet(nn.Module):
    def __init__(self, model_type="seresnext50", n_classes=[168, 11, 7]):
        super(BengaliaiNet, self).__init__()
        self.model_type = model_type
        self.n_classes = n_classes
        
        if model_type == "ResNet34":
            self.basemodel = ptcv_get_model("resnet34", pretrained=True)
            self.basemodel.features.final_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        elif model_type == "seresnext50":
            self.basemodel = ptcv_get_model("seresnext50_32x4d", pretrained=True)
            self.basemodel.features.final_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        elif model_type == "seresnext101":
            self.basemodel = ptcv_get_model("seresnext101_32x4d", pretrained=True)
            self.basemodel.features.final_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        elif model_type == "senet154":
            self.basemodel = ptcv_get_model("senet154", pretrained=True)
            self.basemodel.features.final_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        elif model_type == 'efficientnet-b0':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b0')
        elif model_type == 'efficientnet-b1':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b1')
        elif model_type == 'efficientnet-b2':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b2')
        elif model_type == 'efficientnet-b3':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b3')
        elif model_type == 'efficientnet-b4':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b4')
        elif model_type == 'efficientnet-b5':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b5')
        elif model_type == 'efficientnet-b6':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b6')
        elif model_type == 'efficientnet-b7':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b7')
        elif model_type == 'efficientnet-b8':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b8')
        else:
            raise NotImplementedError
        
        # modules = {}
        # for name, module in self.basemodel.named_modules():
        #     if(isinstance(module, nn.MaxPool2d)):
        #         module.kernel_size = 2
        #         module.padding = 0
        #         modules[name] = module
            
            
        self.avg_poolings = nn.ModuleList([
            GeM() for _ in range(len(self.n_classes))
        ])
    
        # self.dropout = nn.Dropout(0.2)
    
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        
        self.logits = nn.ModuleList(
            [ nn.Linear(2048, c) for c in self.n_classes ]
        )
        
    def forward(self, x):
        
        bs = x.shape[0]
        
        if self.model_type == "ResNet34":
            x = self.basemodel.features(x)
        elif self.model_type == "seresnext50":
            x = self.basemodel.features(x)
        elif self.model_type == "seresnext101":
            x = self.basemodel.features(x)
        elif self.model_type == "senet154":
            x = self.basemodel.features(x)
        elif self.model_type == 'efficientnet-b0':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b1':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b2':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b3':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b4':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b5':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b6':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b7':
            x = self.basemodel.extract_features(x)
        elif self.model_type == 'efficientnet-b8':
            x = self.basemodel.extract_features(x)
        else:
            raise NotImplementedError
        
        # 4 tasks
        logits = []
        
        for i, avg_pooling in enumerate(self.avg_poolings):
            logit = avg_pooling(x)
            logit = logit.view(bs, -1)
            
            for j, dropout in enumerate(self.dropouts):
                
                if j == 0:
                    logit = dropout(logit) / len(self.dropouts)
                else:
                    logit += dropout(logit) / len(self.dropouts)
        
            # logit = self.dropout(logit)
            
            logits.append(self.logits[i](logit))
        
        # logits = [ l(x) for l in self.logits ]
        
        return logits
    
    
    
############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor(np.random.random((96, 3, 137, 236)).astype(np.float32)).cuda()
    model = BengaliaiNet().cuda()

    logits = model(x)
    print(logits[0], logits[1], logits[2])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()