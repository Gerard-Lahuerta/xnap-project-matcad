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
    '''
    INPUT: 
        --> X: list of images
        --> cropx: int (size of x axis of the image).
        --> cropy: int (size of y axis of the image).

    OUTPUT:
        --> ret: list of images (crop center).

    ABOUT IT:
        --> Crop the center of a list (X) of images into an image of dimensions cropx x cropy.

    RELEVANT INFORMATION:
        --> If some image in the list has less dimension (in one axis) than the needed to the crop
            it will be ignored (not cropped) and not be included in the returned list of images.
    '''

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
    '''
    INPUT:
        --> loader: list of 2 list (input list and label list respectively).
    
    OUTPUT:
        --> ret: list of 2 list (input list and label list respectively).

    ABOUT IT:
        --> Shuffle randomly the input list and the label list but in the same way (maintaining the
            input and the label in the same index after shuffling).
    '''

    p = np.random.permutation(len(loader[0])) # random index list to permutate randomly the loader
    train = [loader[0][i] for i in p]
    test = [loader[1][i] for i in p]
    return [train, test]


def get_data_model(path, split = 0.95, train = True):
    '''
    INPUT:
        --> path: string with the directory to obtain the images to train/test the model.
        --> split = 0.95: float between 0 and 1 to select the percentage of images of the folder to
                          return after treating.
        --> train = True: bool, determines if the returned list of images are for testing or training.
    
    OUTPUT:
        --> ret: list of 2 list, input (grey-scale image) and label (AB-scale image) respectively.

    ABOUT IT:
        --> Save all images in the directory selected by the parameter "path", returns the "split" 
            percentage of them in gray-scale version (first position of the array returned) and the
            AB version (second position of the array returned).

    RELEVANT INFORMATION:
        --> If the shape of the images in the folder indicated by the "path" parameter variable
            are not symmetric, the function will call to the "crop_center" function explained above to
            crop the image into a 256 x 256 image.
        --> If train = True selects the first split*"number of images in the directory" images.
        --> If train = False selects the last split*"number of images in the directory" images.
    '''

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
    '''
    INPUT:
        --> config: dict, has to contain the keys "dat_set" and (depending) "train" or "split".
            -> config["data_set"]: string with the word "default" or the path to the dataset (images).
            -> config["model"]: string, (useful if "default") generates the dataset of the model.
            -> config["split]: float, (useful if not "default") determines the percentage the data to
                               return in train list contained in the folder selected by the path.

    OUTPUT:
        --> train: list of 2 list with grey-scale and the AB images (respectively).
        --> test: list of 2 list with grey-scale and the AB images (respectively).

    ABOUT IT:
        --> If mode "default": returns the default dataset of each model to train and test.
        --> If mode "path": returns the dataset selected in the path to train and test.

    RELEVANT INFORMATION:
        --> If a path directory is given in the parameter "path" it will return a test train with a
            0.01 percent of all the dataset included in it.
        --> If while selecting "default" mode the config dictionary with the key "model" not contains
            a correct model, it will return (by default) the dataset of model 3. 
    '''

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
            test = get_data_model("data/data_2/Test/", split = 1)

    else: # selecction custom dataset and train split
        train = get_data_model(config["data_set"], split = config["split"])
        test = get_data_model(config["data_set"], split = 0.01, train = False)

    return train, test


##### BUILDING MODEL ##################################################################################

def built_model(config, device="cuda"):
    '''
    INPUT:
        --> config: dict, has to contain the key "model".
            -> config["model"]: string, name of the model to initialize.

    OUTPUT:
        --> model: CNN encoder-decoder model (pytorch).

    ABOUT IT:
        --> Initialize the model selected by the dictionary "config" using the key "model".

    RELEVANT INFORMATION:
        --> By default and if the model selected is not in the ones we distribute in this project, the
            function will return the model ColorizationNet.
    '''

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
    '''
    INPUT:
        --> yhat: torch.Tensor(), output of the model.
        --> y: torch.Tensor(), label of the input.
    
    OUTPUT:
        --> ret: float, loss of the model.
    
    ABOUT IT:
        --> RMSE function loss programed to be used as a criterion in the training of the model.
    '''

    return torch.sqrt(torch.mean((yhat-y)**2)) 


def set_criterion(config):
    '''
    INPUT:
        --> config: dict, has to contain the key "criterion".
            -> config["criterion"]: string, type of function to model the loss of the model.

    OUTPUT:
        --> criterion: function, used to optimize the model while it is training.

    ABOUT IT:
        --> Selects the loss function to train the model.

    RELEVANT INFORMATION:
        --> By default and if the criterion selected is not in the ones we distribute in this project, the
            function will return the criterion MSE.
    '''

    if config["criterion"] == "RMSE":
        criterion = RMSELoss

    elif config["criterion"] == "MAE":
        criterion = torch.nn.L1Loss()

    else: # MSE loss function by default mode
        criterion = torch.nn.MSELoss()
    
    return criterion


def set_optimizer(config, model):
    '''
    INPUT:
        --> config: dict, has to contain the key "optimizer" and "learning_rate.
            -> config["optimizer"]: string, type of function to model the loss of the model.
            -> config["learning_rate"]: float, learning rate of the optimizer.
        --> model: CNN encoder-decoder model (pytorch).

    OUTPUT:
        --> optimizer: function, used to optimize the model while it is training.

    ABOUT IT:
        --> Selects the optimizer function to train the model.

    RELEVANT INFORMATION:
        --> By default and if the optimizer selected is not in the ones we distribute in this project, the
            function will return the optimizer Adam.
    '''

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
    '''
    INPUT:
        --> config: dict, has to contain the keys "model", "data_set", "split", "criterion", 
                    "optimizer" and "learning_rate".
            -> config["model"]: config["model"]: string, name of the model to initialize.
            -> config["data_set"]: string with the word "default" or the path to the dataset (images).
            -> config["split"]: float, determines the percentage of the data (not default) returned.
            -> config["criterion"]: string, type of function to model the loss of the model.
            -> config["optimizer"]: string, type of function to model the loss of the model.
            -> config["learning_rate"]: float, learning rate of the optimizer.

    OUTPUT:
        --> model: model: CNN encoder-decoder model (pytorch).
        --> train: list of 2 list with grey-scale and the AB images (respectively).
        --> test: list of 2 list with grey-scale and the AB images (respectively).
        --> criterion: function, used to optimize the model while it is training.
        --> optimizer: function, used to optimize the model while it is training.

    ABOUT IT:
        --> Builds the model structure, dataset and the functions needed to train/test the model
            (criterion and optimizer).
    '''

    model = built_model(config)
    train, test = get_data(config)
    criterion = set_criterion(config)
    optimizer = set_optimizer(config, model)

    return model, train, test, criterion, optimizer
 

##### DATA SAVING #####################################################################################

def save_image(output_AB, output_L, size, path, name = "/img_"):
    '''
    INPUT:
        --> output_AB: tensor of 3 x size x size, output of the model (image in AB-scale).
        --> output_L: tensor of 3 x size x size, input of the model (image in grey-scale).
        --> size: int, dimensions of the image (image has to be scared).
        --> path: string, folder path to save the results of the image.
        --> name = "/img_": string, initial name of each image to save.

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Save images fragmented in a tensor of grey-scale and a tensor of AB-scale.
    '''

    output = tqdm(zip(output_AB, output_L, range(len(output_AB))), desc="Saving images")
    for AB, L, i in output:
        save_1_image(AB, L, size, path, name+str(i+1))


def save_1_image(AB, X, size, path, name):
    '''
    INPUT:
        --> AB: tensor of 3 x size x size, output of the model (image in AB-scale).
        --> X: tensor of 3 x size x size, input of the model (image in grey-scale).
        --> size: int, dimensions of the image (image has to be scared).
        --> path: string, folder path to save the results of the image.
        --> name: string, name of the image to be saved.

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Gathers the parts of the image fragmented in a tensor of grey-scale and a tensor of AB-scale and saves
            the resulting png image in the directory path.
    '''
    
    cur = np.zeros((size, size, 3)) # create a size x size x 3 (the 3th dimension corresponding to LAB)
    cur[:, :, 0] = np.array(X[0][0, :, :].cpu()) # add the L component 
    cur[:, :, 1:] = np.array(128 * AB[0].cpu().permute(1, 2, 0)) # add the AB dimensions (model output)
    imsave(path + name + ".png", (lab2rgb(cur) * 255).astype(np.uint8)) # saving image in its folder


def save_model(model):
    '''
    INPUT:
        --> model: CNN encoder-decoder model (pytorch).

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Saves the weights of the model to the file indicated in path.
    '''
    
    path = "weights/Weights "+model.get_name()+".pt" # path + file name of the model weights to save
    torch.save(model.state_dict(), path)


def import_model(model):
    '''
    INPUT:
        --> model: CNN encoder-decoder model (pytorch).

    OUTPUT:
        --> model: Trained CNN encoder-decoder model (pytorch).

    ABOUT IT:
        --> Imports the weights of the model from the file indicated in path, thus importing the already trained model.
    '''
    
    path = "weights/Weights "+model.get_name()+".pt" # path + file name of the model weights
    model.load_state_dict(torch.load(path)) # importing trained model
    return model


def delete_files(dir = "image_log"):
    '''
    INPUT:
        --> dir: string, path to the directory with the images generated with log during the training.

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Deletes the register log of the previous train in the directory dir, in order to save only the one that
            will be generated in the running execution.
    '''
    
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f)) # deleting file of the "dir" folder
