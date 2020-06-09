from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
from data_prepare import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--best_model', type=str, default='pretrained/best_model.pth.tar')
parser.add_argument('--resume', type=str)
parser.add_argument('--n_epoch', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--loss_fn', type=str, default='mse')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--train_ratio', type=float, default=.6)
parser.add_argument('--val_ratio', type=float, default=.2)
parser.add_argument('--n_batch', type=int, default=128,
                    help='number of mini-batch')
parser.add_argument('--test', type=bool, default=True)

args = parser.parse_args()
params = vars(args)

def save_checkpoint(state, is_best, filename=params['best_model']):
    if is_best:
        torch.save(state, filename)



def train(model, optimizer, criterion, train_loader):
    model.train()
    loss_train = 0.
    for i, batch in enumerate(train_loader, 1):
        X, Y = batch
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        net_out = model(X)

        loss = criterion(net_out.squeeze(), Y)
        loss.backward()
        optimizer.step()
        
        loss_train += loss.cpu().data.numpy()

    return(loss_train/len(train_loader))


def validate(model, optimizer, criterion, val_loader):
    model.train()
    loss_val = 0.
    for i, batch in enumerate(val_loader,):
        X, Y = batch
        X, Y = X.to(device), Y.to(device)

        net_out = model(X)
        
        loss_val += criterion(net_out.squeeze(), Y)

    return(loss_val/len(val_loader))


def test(model, optimizer, criterion, test_loader):
    loss_test = 0.
    pred_test = []
    for i, batch in enumerate(test_loader,):
        X, Y = batch
        X, Y = X.to(device), Y.to(device)

        net_out = model(X)
        
        loss_test += criterion(net_out.squeeze(), Y)

        # output probability
        net_out = model(X)
        Y_pred = net_out.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        
        for j in range(len(Y)):
            pred_test.append([Y[j], Y_pred[j]])

    return(loss_test/len(test_loader), np.array(pred_test))

    



# Train model
np.random.seed(int(time.time()))
torch.manual_seed(int(time.time()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

t_total = time.time()

dataset, train_loader, val_loader, test_loader = load_data(params)
params['n_fea'] = len(dataset.Xs[0])

n_train = len(train_loader)
n_val = len(val_loader)

model = NN(params['n_fea'], 1).to(device)

optimizer = optim.Adam(model.parameters(), lr=params['lr'])
criterion = nn.MSELoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    'min',
    patience=10,
    factor=0.9,
    verbose=True)

if params['resume']:
    if os.path.isfile(params['resume']):
        print("=> loading pretrained model '{}'".format(params['resume']))
        checkpoint = torch.load(params['resume'])
        args.start_epoch = checkpoint['epoch']
        best_error = checkpoint['best_error']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(params['resume'], checkpoint['epoch']))
    else:
        print("=> no pretrained file at '{}'".format(params['resume']))

best_error = 1e8
for i in range(params['n_epoch']):
    loss_train  = train(model,
                        optimizer,
                        criterion,
                        train_loader,
                        )
    loss_val = validate(model,
                        optimizer,
                        criterion,
                        val_loader,
                        )

    error = loss_val
    is_best = error < best_error
    best_error = min(error, best_error)
    save_checkpoint({
        'epoch': i + 1,
        'state_dict': model.state_dict(),
        'best_error': best_error,
        'optimizer': optimizer.state_dict(),
        'params': params
    }, is_best)

    print('Epoch: {} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f} '.format(
        i, loss_train, loss_val))

if params['test']:
    loss_test, pred_test = test(model,
                                optimizer,
                                criterion,
                                test_loader,
                                )
    print("Test Loss: {:.4f}".format(loss_test))
    print(scipy.stats.pearsonr(pred_test[:,0], pred_test[:,1]))
    np.savetxt("test2.xvg", pred_test)