import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Down2Up2(nn.Module):

    #初期化
    def __init__(self, num_channels=3):
        super(Down2Up2, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2,stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=9, padding=9//2)
        self.pixsh = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(16, num_channels, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)


    #順伝播の処理
    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.leakyRelu(self.pixsh(out))
        out = self.conv3(out)

        return out

def down2up2():
    model = Down2Up2()
    return model

def train(model, train_loader, test_loader, device):
    #モデル
    model.train()
    #train損失
    train_batch_loss = []
    #平均二乗誤差
    criterion = nn.MSELoss()
    #optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #train
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch)
        #GPUへ
        data, label = sample_batched["image"].to(device), sample_batched["label"].to(device)
        #勾配ゼロ
        optimizer.zero_grad()
        #推論
        output = model(data)
        #誤差
        loss = criterion(output, label)
        #誤差伝播
        loss.backward()
        #パラメータ更新
        optimizer.step()
        #train損失の取得
        train_batch_loss.append(loss.item())

    #test
    model.eval()
    test_batch_loss = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            data, label = sample_batched["image"].to(device), sample_batched["label"].to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())
    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)

def inference(model, dataloader, device):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            #GPUへ
            data, label = sample_batched["image"].to(device), sample_batched["label"].to(device)
            #推論
            output = model(data)
            #Collect data
            preds.append(output)
            labels.append(label)

    return preds, labels