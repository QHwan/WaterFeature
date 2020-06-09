from __future__ import division
from __future__ import print_function

import random
import numpy as np
from collections import Counter
import sparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

class BulkDataset(Dataset):
    def __init__(self):
        w_filename = "data.npz"
        w_file = np.load(w_filename, allow_pickle=True)

        n_fea = 12
        x_w = w_file['feature'].reshape(-1, n_fea)
        y_w = w_file['jump'].flatten()

        data = np.concatenate((x_w, y_w.reshape(-1,1)), axis=1)

        
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data_norm = self.scaler.transform(data)
        self.Xs = data_norm[:,:-1]
        self.Ys = data_norm[:,-1]
        

        #self.Xs = x_w
        #self.Ys = y_w

    def __len__(self):
        return(len(self.Ys))

    def __getitem__(self, idx):
        return((torch.from_numpy(self.Xs[idx]).float(),
            self.Ys[idx]))



def collate_data(dataset):
    batch_Xs = []
    batch_Ys = []
    for i, (Xs, Ys) in enumerate(dataset):
        batch_Xs.append(Xs)
        batch_Ys.append(Ys)

    batch_Ys = np.array(batch_Ys)
    
    return(torch.stack(batch_Xs, dim=0),
        torch.from_numpy(batch_Ys).float())



def load_data(params):
    dataset = BulkDataset()
    n_data = len(dataset)
    n_train = int(n_data*params['train_ratio'])
    n_val = int(n_data*params['val_ratio'])
    n_test = n_data - n_train - n_val
    
    trainset, valset, testset = random_split(dataset, [n_train, n_val, n_test])
    #trainset = Subset(dataset, list(range(n_train)))
    #valset = Subset(dataset, list(range(n_train, n_train+n_val)))
    #testset = Subset(dataset, list(range(n_train+n_val, n_train+n_val+n_test)))

    dataloader_args = {'batch_size': params['n_batch'],
                       'shuffle': True,
                       'pin_memory': False,
                       'drop_last': True,
                       'collate_fn': collate_data,}

    train_loader = DataLoader(trainset, **dataloader_args)
    val_loader = DataLoader(valset, **dataloader_args)
    test_loader = DataLoader(testset, **dataloader_args)

    return(dataset, train_loader, val_loader, test_loader)



def load_predict_data(params, file):
    dataset = TestDataset(file)
    n_data = len(dataset)

    dataloader_args = {'batch_size': params['n_batch'],
                       'shuffle': False,
                       'pin_memory': False,
                       'drop_last': True,
                       'collate_fn': collate_data,}

    dataloader = DataLoader(dataset, **dataloader_args)

    return(dataset, dataloader)