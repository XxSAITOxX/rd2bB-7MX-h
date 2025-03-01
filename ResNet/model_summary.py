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

#pthファイルからモデルの構築&ロード
#model = torch.load("./weight/FractalDB-10000_res18.pth")
#model = torch.load("./weight/resnet18-5c106cde.pth")
#print(model)

#データセット
dataset_path = glob.glob("../PASCAL VOC/VOC2012/JPEGImages/*.jpg")
#print(dataset_path)

size = len(dataset_path)
train_size = int(size*0.8)
val_size = size - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset_path, [train_size, val_size])

print(train_dataset[0])

with open("train.txt","w") as f:
    for i in train_dataset:
        f.write(str(i)+"\n")
with open("val.txt","w") as f:
    for i in val_dataset:
        f.write(str(i)+"\n")

"""
t_path = glob.glob("../ImageNet/ILSVRC/Data/DET/train/**/*.JPEG",recursive=True)
v_path = glob.glob("../ImageNet/ILSVRC/Data/DET/val/*.JPEG")

print(len(t_path))
print(len(v_path))
count = 0
for i in t_path:
    if not(os.path.isfile(i)):
        print(i)
    print(count)
    count+=1
count = 0
for i in v_path:
    if not(os.path.isfile(i)):
        print(i)
    print(count)
    count+=1
"""
"""
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = resnet18(pretrained=False).to(device)
#モデルの読み込み
model.load_state_dict(torch.load("./weight/pretrain_weight.pth"))

#torchsummaryによるモデルの構成確認
summary(model,(3,224,224))
x = torch.rand(500,3,224,224).to(device)
print(model(x).shape)
"""
"""
#Dataset確認
img_size = 224
b_size = 10
crop_size = (224,224)
train_path = glob.glob("./dataset/train/*")
test_path = glob.glob("./dataset/val/*")
hr_transform = transforms.Compose([
                                        transforms.Resize(img_size, interpolation=2),
                                        transforms.RandomCrop(crop_size),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
lr_transform = transforms.Compose([
                                        transforms.Resize(int(img_size/4), interpolation=2),
                                        transforms.Resize(int(img_size), interpolation=2),
                                        transforms.RandomCrop(crop_size),
                                        transforms.ToTensor(),
                                        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
train_dataset = Data_Loader(train_path, hr_transform, lr_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=None)
#print(next(iter(train_loader)))

test_dataset = Data_Loader(test_path, hr_transform, lr_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, worker_init_fn=None)

epoch = 100
train_loss = []
test_loss = []
save_name = "train1/"

best_loss = 10000
#train
for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train(model, train_loader, test_loader, device)
    train_loss.append(train_l)
    test_loss.append(test_l)
    #重み保存
    if epoch % 10 == 0:
        path_saved_model = "./weight/"+save_name+str(epoch)+".pth"
        torch.save(model.state_dict(), path_saved_model)
    #ベストエポック
    if best_loss>test_loss[-1]:
        best_loss = test_loss[-1]
        path_saved_model = "./weight/"+save_name+"best.pth"
        torch.save(model.state_dict(), path_saved_model)
    #エポックごとに表示
    if epoch % 10 == 0:
        print("Epoch:"+str(epoch),"Train loss: {a:.3f}, Test loss: {b:.3f}".format(a=train_loss[-1], b = test_loss[-1]))

#モデルの保存
path_saved_model = "./weight/"+save_name+"last.pth"
torch.save(model.state_dict(), path_saved_model)
"""
"""
plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
"""

"""
#eval
name = 0
preds, labels = inference(model, train_loader, device)
preds = torch.cat(preds, axis=0)
labels = torch.cat(labels, axis=0)
for i in preds:
    #print(i.shape)
    img = (i*255).to(torch.uint8)
    img = img.cpu().detach().numpy()
    img = np.array(img)
    img = np.stack([img[0],img[1],img[2]], axis=2)
    img = Image.fromarray(img,mode="RGB")
    img.save("./run/preds/"+str(name)+".png")
    name += 1
name = 0

for i in labels:
    #print(i.shape)
    img = (i*255).to(torch.uint8)
    img = img.cpu().detach().numpy()
    img = np.array(img)
    img = np.stack([img[0],img[1],img[2]], axis=2)
    img = Image.fromarray(img,mode="RGB")
    img.save("./run/labels/"+str(name)+".png")
    name += 1
name = 0
"""

