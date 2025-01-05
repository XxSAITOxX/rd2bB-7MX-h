from torchsummary import summary
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from resnet18 import *

#pthファイルからモデルの構築&ロード
#model = torch.load("./weight/FractalDB-10000_res18.pth")
#model = torch.load("./weight/resnet18-5c106cde.pth")
#print(model)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet18(pretrained=False, num_classes=1000).to(device)

#torchsummaryによるモデルの構成確認
summary(model,(3,224,224))
x = torch.rand(4,3,224,224).to(device)
print(model(x).shape)