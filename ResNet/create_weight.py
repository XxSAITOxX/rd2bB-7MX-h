from torchsummary import summary
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from resnet18 import *
from resnet import resnet18 as resnet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model1 = resnet18(pretrained=False).to(device)
model2 = resnet(pretrained=False, num_classes=10000).to(device)
#モデルの読み込み
model1.load_state_dict(torch.load("./weight/init_weight.pth"))
model2.load_state_dict(torch.load("./weight/FractalDB-10000_res18.pth"))

#print(model1.state_dict().keys())
keys1 = list(model1.state_dict().keys())
keys2 = list(model2.state_dict().keys())

list_and = list(set(keys1) & set(keys2))
#print(list_and)

#i = "conv1.weight"
#print(model1.conv1.weight)
#st = "model1."+i+" = nn.Parameter(torch.tensor(model2."+i+"))"
#exec(st)
#model1.conv1.weight = nn.Parameter(torch.tensor(model2.conv1.weight))
#print(model1.layer1[0].conv1.weight)
#model1.layer1 = model1.layer1.replace(model2.layer2)
#print(model1.layer1)

ll = []
with open('./list.txt') as f:
    for line in f:
        ll.append(line.rstrip("\n"))

for i in ll:
    st = "model1."+i+" = nn.Parameter(torch.tensor(model2."+i+"))"
    print(st)
    if "batches_tracked" in st:
        st = "model1."+i+" = model2."+i
        exec(st)
    else:
        exec(st)

path_saved_model = "./weight/pretrain_weight.pth"
torch.save(model1.state_dict(), path_saved_model)