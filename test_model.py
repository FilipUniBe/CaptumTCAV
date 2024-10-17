import os
import pandas as pd
import random
import torch
import pandas as pd

from main_on_CheXpert import preprocess_excel, load_and_prepare_model
from model import DenseNet121
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from data_loading import Load_from_path_Dataset
from model import DenseNet121
from sampler import StratifiedSampler
from tqdm import tqdm

def test(test_dataloader, n_classes, model, device):
    model.eval()

    test_loss = []
    labels_list = []
    preds_list = []

    with torch.no_grad():

        for batch_idx, data_batch in enumerate(tqdm(test_dataloader)):
            image, labels = data_batch[0].float(), data_batch[1].float()

            # calculating class weight
            class_proportion = 1 - (torch.sum(labels, 0) / labels.shape[0])
            class_weights = class_proportion / class_proportion.sum()

            if device!='cpu':
                image, labels, class_weights = image.to(device), labels.to(device), class_weights.to(device)

            logits = model(image)
            preds = torch.sigmoid(logits)

            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights.data)
            # criterion = torch.nn.BCELoss()
            loss = criterion(logits, target=labels)

            labels_list.append(labels.cpu().data.numpy())
            preds_list.append(preds.cpu().data.numpy())
            test_loss.append(loss.cpu().data.numpy())

        test_loss_mean = np.mean(test_loss)
        print("test_loss", test_loss_mean)

        targets = np.concatenate(labels_list)
        outputs = np.concatenate(preds_list)

        roc_auc = roc_auc_score(targets, outputs)
        ap = average_precision_score(targets, outputs)

        return roc_auc, ap, test_loss_mean


if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # if multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    homepath = "/home/fkraehenbuehl/projects/"

    usemarker=False #todo change for marker model
    if usemarker == True:
        dspath = "CheXpert-v1.0-marker"
    else:
        dspath = "CheXpert-v1.0"
    print("dspath", dspath)

    test_labels_meta = pd.read_csv(homepath + f"{dspath}/test.csv")
    exist_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    test_labels_meta = preprocess_excel(test_labels_meta, exist_labels)
    y_test = test_labels_meta[exist_labels].values
    x_test_path = test_labels_meta.Path
    imgw=320
    imgh=320
    bs=16
    test_dataset = Load_from_path_Dataset(x_test_path, homepath + f"{dspath}/", y_test, imgw, imgh,
                                          mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=20)

    path_load_model='/home/fkraehenbuehl/projects/SalCon/model/models/model_0/densenet pretrain unweighted bce with class weight wd0.0001_model_gc_lr0.0001_epoches5.pt'#todo
    device=0#todo
    model = load_and_prepare_model(path_load_model, 5, device)

    print('Start Testing')
    roc_auc, ap, test_loss = test(test_dataloader=test_dataloader, n_classes=5, model=model,
                                  device=device)
    print("test roc_auc:", roc_auc, np.mean(roc_auc), "average precision:", ap, "test_loss", test_loss)
































