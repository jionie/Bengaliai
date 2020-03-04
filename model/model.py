import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet

from apex import amp



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
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * F.sigmoid(x)
    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


############################################ Define Net Class
class BengaliaiNet(nn.Module):
    def __init__(self, model_type="efficientnet-b1", n_classes=[168, 11, 7, 1295]):
        super(BengaliaiNet, self).__init__()
        self.model_type = model_type
        self.n_classes = n_classes
        
        if model_type == "ResNet34":
            self.basemodel = ptcv_get_model("resnet34", pretrained=True)
            self.basemodel.features.final_pool = Identity()
            self.feature_size = 2048
        elif model_type == "seresnext50":
            self.basemodel = ptcv_get_model("seresnext50_32x4d", pretrained=True)
            self.basemodel.features.final_pool = Identity()
            self.feature_size = 2048
        elif model_type == "seresnext101":
            self.basemodel = ptcv_get_model("seresnext101_32x4d", pretrained=True)
            self.basemodel.features.final_pool = Identity()
            self.feature_size = 2048
        elif model_type == "senet154":
            self.basemodel = ptcv_get_model("senet154", pretrained=True)
            self.basemodel.features.final_pool = Identity()
            self.feature_size = 2048
        elif model_type == 'efficientnet-b0':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b0')
            self.feature_size = 2048
        elif model_type == 'efficientnet-b1':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b1', advprop=True)
            self.feature_size = 1280
        elif model_type == 'efficientnet-b2':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b2')
            self.feature_size = 2048
        elif model_type == 'efficientnet-b3':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b3', advprop=True)
            self.feature_size = 1536
        elif model_type == 'efficientnet-b4':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b4')
            self.feature_size = 2048
        elif model_type == 'efficientnet-b5':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b5', advprop=True)
            self.feature_size = 2048
        elif model_type == 'efficientnet-b6':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b6')
            self.feature_size = 2048
        elif model_type == 'efficientnet-b7':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b7')
            self.feature_size = 2048
        elif model_type == 'efficientnet-b8':
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b8')
            self.feature_size = 2048
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
    
        # self.dropout = nn.Dropout(0.5)
    
        # self.logits = nn.ModuleList(
        #     [ nn.Sequential(nn.Dropout(0.25), nn.BatchNorm1d(self.feature_size), nn.Linear(self.feature_size, 512), Mish(), \
        #       nn.Dropout(0.25), nn.BatchNorm1d(512), nn.Linear(512, c)) for c in self.n_classes ]
        # )
        
        self.tail = nn.ModuleList([
             nn.Sequential(Mish(), nn.Conv2d(self.feature_size, 512, 1), nn.Dropout(0.2), nn.BatchNorm2d(512)) for _ in self.n_classes 
        ])
        
        self.logits = nn.ModuleList(
            [ nn.Linear(512, c) for c in self.n_classes ]
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
        
        for i in range(len(self.n_classes)):
            
            logit = self.tail[i](x)
            logit = self.avg_poolings[i](logit)
            logit = logit.view(bs, -1)
        
            logits.append(self.logits[i](logit))
        
        
        # logits = [ l(x) for l in self.logits ]
        
        return logits
    
    
    
############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor(np.random.random((144, 3, 224, 224)).astype(np.float32)).cuda()
    model = BengaliaiNet().cuda()
    model = amp.initialize(model, opt_level="O1")

    logits = model(x)
    print(logits[0], logits[1], logits[2], logits[3])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()