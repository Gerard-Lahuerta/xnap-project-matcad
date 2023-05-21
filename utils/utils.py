import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *
import matplotlib.pyplot as plt

import json
from tqdm.auto import tqdm

from PIL import Image
import os
import numpy as np
from skimage import io, color

from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray

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
    
###################################################################################

def lba_exclusion(image, channel):
    out = image
    if channel == 'L':
        return out[:, :, 0]
    else:
        return out[:, :, 1:]


def get_data_model_1(path):
    rgb = io.imread(path)
    
    lab = color.rgb2lab(1.0/255*rgb[:,:,0:3])
    lab = np.array(lab, dtype = "float")

    X = lba_exclusion(lab, "L")
    Y = lba_exclusion(lab, "AB")/128

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    X = transform(X).float().reshape(1,1,400,400)
    Y = transform(Y).float().reshape(1,2,400,400)

    return [X, Y]

def get_data_model_2(path, split = 0.95, train = True):

    X = []
    for filename in os.listdir(path):
        X.append(io.imread(path + filename))

    split = int(split * len(X))
    if not train:
        split = 1-split
        X = X[split:]
    else:
        X = X[:split]
    transform = transforms.ToTensor()

    Xtest = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 0]
    Xtest = Xtest.reshape(Xtest.shape + (1,))
    aux = []
    for i in Xtest:
        aux.append(transform(i).float().reshape(1,1,256,256))
    Xtest = aux

    Ytest = color.rgb2lab(1.0 / 255 * np.array(X, dtype="float"))[:, :, :, 1:]
    Ytest = Ytest / 128
    aux = []
    for i in Ytest:
        aux.append(transform(i).float().reshape(1,2,256,256))
    Ytest = aux

    return [Xtest, Ytest]


def get_data_model_3(path):
    Xaux = []
    
    X = []
    # path = ruta a la carpeta on es troben totes les imatges pel TRAIN
    for filename in os.listdir(path):
        Xaux.append(Image.open(path + filename))
    for image in Xaux:
        image = image.convert('L')
        transform = transforms.ToTensor()
        image = transform(image)
        X.append(image)

    #split = int(0.95 * len(X))
    #Xtrain = X[:split]
    #print(X)
    return X

'''
def get_test_data_model_2(path):
    X = []
    for filename in os.listdir(path):
        img = Image.open(path + filename)
        X.append(np.array(img).astype(float))
    X = np.array(X)

    X = rgb2lab(1.0/255 * X)[:,:,:,0]
    X = np.reshape(X, X.shape + (1,))
    #print(X)
    return X
'''

def get_especific_data(path):
    pass

def get_data(config):
    if config["data_set"] == "default":
        if config["model"] == "Model 1":
            train = get_data_model_1("data/data_1/woman.jpg")
            test = get_data_model_1("data/data_1/woman.jpg")

        elif config["model"] == "Model 2":
            train = get_data_model_2("data/data_2_3/Train/")
            test = get_data_model_2("data/data_2_3/Test/", train=False)

        else: # Model 3
            train = get_data_model_3("data/data_2_3/Train/")

    else:
        train, test = get_especific_data(config["data_set"])

    return train, test

def built_model(config):
    if config["model"] == "Model 1":
        model = Model1().to(device)
    elif config["model"] == "Model 2":
        model = Model2().to(device)
    else: # Model 3
        model = Model3().to(device) 

    return model

def set_criterion(config):
    if config["criterion"] == "MSE":
        criterion = torch.nn.MSELoss()
    
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
        optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.01)
    
    return optimizer
    

def make(config, device = "cuda"):

    model = built_model(config)

    train, test = get_data(config)

    criterion = set_criterion(config)

    optimizer = set_optimizer(config, model)

    return model, train, test, criterion, optimizer
 

def save_image(output_AB, output_L, size, path):
    output = tqdm(zip(output_AB, output_L, range(len(output_AB))), desc="Saving images")
    for AB, L, i in output: #zip(output_AB, output_L, range(len(output_AB))):
        cur = np.zeros((size, size, 3))
        cur[:,:,0] = np.array(L[0][0,:,:].cpu())
        cur[:,:,1:] = np.array(128*AB[0].cpu().permute(2,1,0))
        imsave(path+"/img_"+str(i+1)+".png", (lab2rgb(cur)*255).astype(np.uint8))


def save_model(model):
    doc = "weights/Weights "+model.get_name()+".json"
    weights = model.state_dict()

    # Convert the state_dict to a JSON-serializable format
    weight = tqdm(weights.items, desc="Saving model weights")
    weight_dic = {}
    for key, value in weight:
        weight_dic[key] = value.tolist()  # Convert tensors to lists

    # Save the JSON to a file
    with open(doc, 'w') as file:
        json.dump(weight_dic, file)


def import_model(model):
    doc = "weights/Weights "+model.get_name()+".json"

    with open(doc, 'r') as file:
        weights = json.load(file)

    # Convert the JSON-serialized state_dict back to PyTorch tensors
    weight = tqdm(weights.items, desc="Importing model weights")
    weight_dic = {}
    for key, value in weight:
        weight_dic[key] = torch.tensor(value)

    model.load_state_dict(weight_dic)
    return model
