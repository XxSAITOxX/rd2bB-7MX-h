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
from train import *
from dataset import Data_Loader
import glob
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
from inference import *
from PIL import Image
import glob
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet18(pretrained=False).to(device)
#モデルの読み込み
model.load_state_dict(torch.load("../weight/train2/best.pth"))

#torchsummaryによるモデルの構成確認
summary(model,(3,224,224))
x = torch.rand(1,3,224,224).to(device)
print(model(x).shape)