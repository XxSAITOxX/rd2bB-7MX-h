import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    """
    #Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    #return as numpy
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    """
    return preds, labels