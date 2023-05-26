"""
    Auxiliar function file: data import, build model and saving data
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo Miniéri

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""

import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *

import json
from tqdm.auto import tqdm
import random

import os
import numpy as np
from skimage import io, color

from skimage.io import imsave
from skimage.color import lab2rgb

'''
def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
'''
    
def crop_center(X,cropx,cropy):
    ret = []
    for img in tqdm(X, desc="Adjusting Images"):
        y = img.shape[0]
        x = img.shape[1]
        if x > cropx and y > cropy:
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            img = img[starty:starty + cropy, startx:startx + cropx, :]
            ret.append(img)
    return ret

def get_data_model(path, split = 0.95, train = True):

    X = []
    for filename in os.listdir(path):
        X.append(io.imread(path + filename)[:,:,0:3])

    split = int(split * len(X))
    if not train:
        split = 1-split
        X = X[split:]
    else:
        X = X[:split]

    size = X[0].shape

    if size[0] != size[1]:
        size = 256
        X = crop_center(X,256,256)
    else:
        size = size[0]

    transform = transforms.ToTensor()

    Xtest = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 0]

    Xtest = Xtest.reshape(Xtest.shape + (1,))

    aux = []
    for i in Xtest:
        aux.append(transform(i).float().reshape(1, 1, size, size))
    Xtest = aux

    Ytest = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 1:]
    Ytest = Ytest / 128
    aux = []
    for i in Ytest:
        aux.append(transform(i).float().reshape(1, 2, size, size))
    Ytest = aux

    return [Xtest, Ytest]


def get_data(config):
    if config["data_set"] == "default":
        if config["model"] == "Model 1":
            train = get_data_model("data/data_1/", split = 0.5)
            test = get_data_model("data/data_1/", split = 0.5)

        elif config["model"] == "Model 2":
            train = get_data_model("data/data_2/Train/", split = 1)
            test = get_data_model("data/data_2/Test/", split = 1)

        else: # Model 3
            train = get_data_model("data/data_2/Train/", split = 1)
            test = get_data_model("data/data_2/Train/", split = 1)

    else:
        train = get_data_model(config["data_set"], split = config["split"])
        test = get_data_model(config["data_set"], split = 0.01, train = False)

    return train, test


def built_model(config, device="cuda"):
    if config["model"] == "Model 1":
        model = Model1().to(device)
    elif config["model"] == "Model 2":
        model = Model2().to(device)
    elif config["model"] == "Model 3": # Model 3
        model = Model3().to(device) 
    elif config["model"] == "ConVAE":
        model = ConvAE().to(device)
    else:
        model = ColorizationNet().to(device)

    return model

def shuffle(loader):
    p = np.random.permutation(len(loader[0]))
    train = [loader[0][i] for i in p]
    test = [loader[1][i] for i in p]
    return [train, test]


def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def set_criterion(config):
    if config["criterion"] == "MSE":
        criterion = torch.nn.MSELoss()

    elif config["criterion"] == "MAE":
        criterion = torch.nn.L1Loss()

    else:
        criterion = RMSELoss
    
    return criterion


def set_optimizer(config, model):
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) # <-- ñe
    
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], momentum=0.9) # <-- una mierda tremenda
    
    elif config["optimizer"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"]) # <-- kk
    
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    
    elif config["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    
    return optimizer
    

def make(config, device = "cuda"):

    model = built_model(config)
    train, test = get_data(config)
    criterion = set_criterion(config)
    optimizer = set_optimizer(config, model)

    return model, train, test, criterion, optimizer#, scheduler
 

def save_image(output_AB, output_L, size, path, name = "/img_"):
    output = tqdm(zip(output_AB, output_L, range(len(output_AB))), desc="Saving images")
    for AB, L, i in output:
        save_1_image(AB, L, size, path, name+str(i+1))
        '''
        cur = np.zeros((size, size, 3))
        cur[:,:,0] = np.array(L[0][0,:,:].cpu())
        cur[:,:,1:] = np.array(128*AB[0].cpu().permute(1,2,0))
        imsave(path+name+str(i+1)+".png", (lab2rgb(cur)*255).astype(np.uint8))
        '''

def save_1_image(AB, X, size, path, name):
    cur = np.zeros((size, size, 3))
    cur[:, :, 0] = np.array(X[0][0, :, :].cpu())
    cur[:, :, 1:] = np.array(128 * AB[0].cpu().permute(1, 2, 0))
    imsave(path + name + ".png", (lab2rgb(cur) * 255).astype(np.uint8))

def save_model(model):
    path = "weights/Weights "+model.get_name()+".pt"
    torch.save(model.state_dict(), path)


def import_model(model):
    path = "weights/Weights "+model.get_name()+".pt"
    model.load_state_dict(torch.load(path))
    return model

def delete_files(path = "image_log/"):
    for f in os.listdir(path):
        os.remove(os.path.join(dir, f))
