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
    
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
class MishConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MishConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.mish(x)
    
class Conv2dBN(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
    
class InceptionA(nn.Module):
    
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


############################################ Define Net Class
class BengaliaiNet(nn.Module):
    def __init__(self, model_type="seresnext50", n_classes=[168, 11, 7, 1295]):
        super(BengaliaiNet, self).__init__()
        self.model_type = model_type
        self.n_classes = n_classes
        
        if model_type == "ResNet34":
            self.basemodel = ptcv_get_model("resnet34", pretrained=True)
            self.basemodel.features.final_pool = Identity()
            self.feature_size = 512
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
        
        # self.head  = nn.Sequential(
        #     BasicConv2d(3, 64, kernel_size=5, stride=1, padding=2), #bias=0
        #     BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1), #bias=0
        #     BasicConv2d(64, 3, kernel_size=3, stride=1, padding=1), #bias=0
        # )
        
        self.head = nn.Sequential(
            InceptionA(3, 32), # (bs, 64+64+96+32, H, W)
            BasicConv2d(64+64+96+32, 3, kernel_size=1, stride=1, padding=0), #bias=0
        )
        
        # self.tail = nn.ModuleList([
        #      nn.Sequential(Mish(), Conv2dBN(self.feature_size, 512, kernel_size=1)) for _ in self.n_classes 
        # ])
        
        self.tail = nn.ModuleList([
            nn.Sequential(Mish(), Conv2dBN(self.feature_size, 512, kernel_size=1)), \
            nn.Sequential(Mish(), Conv2dBN(self.feature_size, 512, kernel_size=1)), \
            nn.Sequential(Mish(), Conv2dBN(self.feature_size, 512, kernel_size=1)), \
            nn.Sequential(Mish(), Conv2dBN(self.feature_size, 512, kernel_size=1))
        ])
        
        # self.logits = nn.ModuleList(
        #     [ nn.Linear(4096, c) for c in self.n_classes ]
        # )
        
        self.logits = nn.ModuleList(
            [ nn.Linear(512, c) for c in self.n_classes ]
        )
        
    def forward(self, x):
        
        bs = x.shape[0]
        
        # x = self.head(x)
        
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
            # print(logit.shape)
            logit = self.avg_poolings[i](logit)
            logit = logit.view(bs, -1)
        
            logits.append(self.logits[i](logit))
        
        
        # logits = [ l(x) for l in self.logits ]
        
        return logits
    
    
    
############################################ Define test Net function
def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor(np.random.random((800, 3, 64, 112)).astype(np.float32)).cuda()
    model = BengaliaiNet().cuda()
    model = amp.initialize(model, opt_level="O1")
    print(model.state_dict().keys())
    logits = model(x)
    print(logits[0], logits[1], logits[2], logits[3])
    print("------------------------testing Net finished----------------------")

    return

def test_Net_eval():
    print("------------------------testing Net----------------------")
    with torch.no_grad():
        x = torch.tensor(np.random.random((64, 3, 137, 236)).astype(np.float32)).cuda()
        model = BengaliaiNet().cuda().eval()
        model = amp.initialize(model, opt_level="O1")

        logits = model(x)
        print(logits[0], logits[1], logits[2], logits[3])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == "__main__":
    print("------------------------testing Net----------------------")
    test_Net()
    # test_Net_eval()