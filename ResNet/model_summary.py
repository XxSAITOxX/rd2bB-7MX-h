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
from dataset import Data_Loader
import glob
import torch.utils.data

#pthファイルからモデルの構築&ロード
#model = torch.load("./weight/FractalDB-10000_res18.pth")
#model = torch.load("./weight/resnet18-5c106cde.pth")
#print(model)
"""
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet18(pretrained=False, num_classes=1000).to(device)

#torchsummaryによるモデルの構成確認
summary(model,(3,224,224))
x = torch.rand(4,3,224,224).to(device)
print(model(x).shape)
"""

#Dataset確認
img_size = 224
b_size = 1
crop_size1 = (224,224)
crop_size2 = (56,56)
path = glob.glob("./dataset/train/*")
train_hr_transform = transforms.Compose([
                                        transforms.Resize(img_size, interpolation=2),
                                        transforms.RandomCrop(crop_size1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
train_lr_transform = transforms.Compose([
                                        transforms.Resize(int(img_size/4), interpolation=2),
                                        transforms.RandomCrop(crop_size2),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
train_dataset = Data_Loader(path, train_hr_transform, train_lr_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=None)
print(next(iter(train_loader)))

