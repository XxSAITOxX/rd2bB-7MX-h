import torch
import numpy as np
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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet18(pretrained=False).to(device)
#モデルの読み込み
model.load_state_dict(torch.load("../weight/train3/best.pth"))

#Dataset確認
img_size = 224
b_size = 1
#crop_size = (224,224)

#train_path = glob.glob("../ImageNet/ILSVRC/Data/DET/train/**/*.JPEG",recursive=True)
#test_path = glob.glob("../ImageNet/ILSVRC/Data/DET/val/*.JPEG")
test_path = glob.glob("../ImageNet/ILSVRC2017_DET_test_new/ILSVRC/Data/DET/test/*")

#データセット
"""
train_txt = "train.txt"
val_txt = "val.txt"
train_path = []
test_path = []

with open(train_txt) as f:
    for line in f:
        if line[-4:].find("\n"):
            train_path.append(line[:-1])
        else:
            train_path.append(line)

with open(val_txt) as f:
    for line in f:
        if line[-4:].find("\n"):
            test_path.append(line[:-1])
        else:
            test_path.append(line)
"""

hr_transform = transforms.Compose([
                                        transforms.Resize(img_size, interpolation=2),
                                        #transforms.RandomCrop(crop_size),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
lr_transform = transforms.Compose([
                                        transforms.Resize(int(img_size/4), interpolation=2),
                                        transforms.Resize(img_size, interpolation=2),
                                        #transforms.RandomCrop(crop_size),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
#train_dataset = Data_Loader(train_path, hr_transform, lr_transform)
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, worker_init_fn=None)
#print(next(iter(train_loader)))

test_dataset = Data_Loader(test_path, hr_transform, lr_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, worker_init_fn=None)

epoch = 50
train_loss = []
test_loss = []
save_name = "train3/"

best_loss = 10000
"""
#train
for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train(model, train_loader, test_loader, device)
    train_loss.append(train_l)
    test_loss.append(test_l)
    #重み保存
    if epoch % 1 == 0:
        path_saved_model = "../weight/"+save_name+str(epoch)+".pth"
        torch.save(model.state_dict(), path_saved_model)
    #ベストエポック
    if best_loss>test_loss[-1]:
        best_loss = test_loss[-1]
        path_saved_model = "../weight/"+save_name+"best.pth"
        torch.save(model.state_dict(), path_saved_model)
    #エポックごとに表示
    if epoch % 1 == 0:
        print("Epoch:"+str(epoch),"Train loss: {a:.3f}, Test loss: {b:.3f}".format(a=train_loss[-1], b = test_loss[-1]))

#モデルの保存
path_saved_model = "../weight/"+save_name+"last.pth"
torch.save(model.state_dict(), path_saved_model)

plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.savefig("Epoch_Error2.png")
"""


#eval
name = 0
preds, labels = inference(model, test_loader, device)
preds = torch.cat(preds, axis=0)
labels = torch.cat(labels, axis=0)
for i in preds:
    #print(i.shape)
    img = (i*255).to(torch.uint8)
    img = img.cpu().detach().numpy()
    img = np.array(img)
    img = np.stack([img[0],img[1],img[2]], axis=2)
    img = Image.fromarray(img,mode="RGB")
    img.save("./run/inference3/"+str(name)+".png")
    name += 1
name = 0

for i in labels:
    #print(i.shape)
    img = (i*255).to(torch.uint8)
    img = img.cpu().detach().numpy()
    img = np.array(img)
    img = np.stack([img[0],img[1],img[2]], axis=2)
    img = Image.fromarray(img,mode="RGB")
    img.save("./run/GT3/"+str(name)+".png")
    name += 1
name = 0
