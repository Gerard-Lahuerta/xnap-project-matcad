"""
    Auxiliar function file: data import, build model and saving data
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo -- 1533327

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""


###### IMPORTS ########################################################################################

# model / dataset manipulations libraries
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *
import random
import numpy as np

# documentative libraries
from tqdm.auto import tqdm

# image manipulation /saving libraries
from skimage import io, color
from skimage.io import imsave
from skimage.color import lab2rgb
import os


##### ADJUSTING DATASET ###############################################################################
  
def crop_center(X,cropx,cropy):
    ret = []
    for img in tqdm(X, desc="Adjusting Images"):
        y = img.shape[0]
        x = img.shape[1]
        if x > cropx and y > cropy: # if some image hase less dimension than need we discard it
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)
            img = img[starty:starty + cropy, startx:startx + cropx, :]
            ret.append(img)
    return ret

def shuffle(loader):
    p = np.random.permutation(len(loader[0])) # random index list to permutate randomly the loader
    train = [loader[0][i] for i in p]
    test = [loader[1][i] for i in p]
    return [train, test]

def get_data_model(path, split = 0.95, train = True):

    # FRAGMENTATION OF THE DATASET
    X = []
    for filename in os.listdir(path):
        X.append(io.imread(path + filename)[:,:,0:3]) 

    split = int(split * len(X)) # select a fragment of the hole dataset

    if not train: # select the last fragment of the dataset
        split = 1-split
        X = X[split:]
    else: #select the first fragment of the dataset
        X = X[:split]

    # DETERMINING SIZE OF IMAGE
    size = X[0].shape 
    if size[0] != size[1]: # not simetric image --> ajust image to make it simetric (crop it)
        size = 256
        X = crop_center(X,256,256)
    else: 
        size = size[0]

    # TRANSFORMING DATASET AND SEPARATION DATA (X) AND LABEL(Y)
    transform = transforms.ToTensor()

    ## data X separation (L image)
    L = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 0]
    L = L.reshape(L.shape + (1,))

    aux = []
    for i in L:
        aux.append(transform(i).float().reshape(1, 1, size, size))
    L = aux

    ## data Y separation (AB image)
    AB = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 1:]
    AB = AB / 128

    aux = []
    for i in AB:
        aux.append(transform(i).float().reshape(1, 2, size, size))
    AB = aux

    return [L, AB]

def get_data(config):
    if config["data_set"] == "default": # default mode --> original dataset of each model
        if config["model"] == "Model 1":
            train = get_data_model("data/data_1/", split = 0.5)
            test = get_data_model("data/data_1/", split = 0.5, train = False)

        elif config["model"] == "Model 2":
            train = get_data_model("data/data_2/Train/", split = 1)
            # do not put the train=False parameter because test dataset is in a separate folder
            test = get_data_model("data/data_2/Test/", split = 1)

        else: # Model 3
            train = get_data_model("data/data_2/Train/", split = 1)
            test = get_data_model("data/data_2/Train/", split = 1)

    else: # selecction custom dataset and train split
        train = get_data_model(config["data_set"], split = config["split"])
        test = get_data_model(config["data_set"], split = 0.01, train = False)

    return train, test


##### BUILDING MODEL ##################################################################################

def built_model(config, device="cuda"):
    if config["model"] == "Model 1":
        model = Model1().to(device)

    elif config["model"] == "Model 2":
        model = Model2().to(device)

    elif config["model"] == "Model 3":
        model = Model3().to(device) 

    elif config["model"] == "ConVAE":
        model = ConvAE().to(device)

    else: # ColorizationNet by default mode
        model = ColorizationNet().to(device)

    return model

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2)) 

def set_criterion(config):
    if config["criterion"] == "RMSE":
        criterion = RMSELoss

    elif config["criterion"] == "MAE":
        criterion = torch.nn.L1Loss()

    else: # MSE loss function by default mode
        criterion = torch.nn.MSELoss()
    
    return criterion

def set_optimizer(config, model):
    if config["optimizer"] == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    
    elif config["optimizer"] == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"])
    
    elif config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    
    else: # Adam optimizer by default mode
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    return optimizer
    
def make(config, device = "cuda"):

    model = built_model(config)
    train, test = get_data(config)
    criterion = set_criterion(config)
    optimizer = set_optimizer(config, model)

    return model, train, test, criterion, optimizer
 

##### DATA SAVING #####################################################################################

def save_image(output_AB, output_L, size, path, name = "/img_"):
    output = tqdm(zip(output_AB, output_L, range(len(output_AB))), desc="Saving images")
    for AB, L, i in output:
        save_1_image(AB, L, size, path, name+str(i+1))

def save_1_image(AB, X, size, path, name):
    cur = np.zeros((size, size, 3)) # create a size x size x 3 (the 3th dimension corresponding to LAB)
    cur[:, :, 0] = np.array(X[0][0, :, :].cpu()) # add the L component 
    cur[:, :, 1:] = np.array(128 * AB[0].cpu().permute(1, 2, 0)) # add the AB dimensions (model output)
    imsave(path + name + ".png", (lab2rgb(cur) * 255).astype(np.uint8)) # saving image in its folder

def save_model(model):
    path = "weights/Weights "+model.get_name()+".pt" # path + file name of the model weights to save
    torch.save(model.state_dict(), path)

def import_model(model):
    path = "weights/Weights "+model.get_name()+".pt" # path + file name of the model weights
    model.load_state_dict(torch.load(path)) # importing trained model
    return model

def delete_files(dir = "image_log"):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f)) # deleting file of the "dir" folder
