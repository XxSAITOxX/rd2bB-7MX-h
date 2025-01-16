import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#Train
def train(model, train_loader, test_loader, device):
    #モデル
    model.train()
    #train損失
    train_batch_loss = []
    #平均二乗誤差
    criterion = nn.MSELoss()
    #optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i_batch, sample_batched in enumerate(train_loader):
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
