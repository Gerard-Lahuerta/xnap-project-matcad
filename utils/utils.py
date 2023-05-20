import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import os
import numpy as np
from skimage import io, color
from skimage.color import rgb2lab

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

'''
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

def get_data_model_2(path):
    Xaux = []
    X = []
    # path = ruta a la carpeta on es troben totes les imatges
    for filename in os.listdir(path):
        Xaux.append(Image.open(path + filename))
    for image in Xaux:
        image = image.convert('L')
        transform = transforms.ToTensor()
        image = transform(image)
        X.append(image)

    split = int(0.95 * len(X))
    Xtrain = X[:split]
    #print(X)
    return Xtrain

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

def make(model_type, config, device = "cuda"):

    if model_type == "Model 1":
        train = get_data_model_1("data/data_1/woman.jpg")
        test = get_data_model_1("data/data_1/woman.jpg")
        model = Model1().to(device)
    elif model_type == "Model 2":
        train = get_data_model_2("data/data_2_3/Train/")
        test = get_test_data_model_2("data/data_2_3/Test/")
        model = Model2().to(device)
    elif model_type == "Model 3":
        train = get_data_model_3("data/data_2_3/Train/")
        model = Model3().to(device) # por hacer
        ### more things
    else:
        assert(False)

    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) # <-- ñe
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], momentum=0.9) # <-- una mierda tremenda
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=config["learning_rate"]) # <-- kk
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adagrad(model.parameters(), lr = 0.01)

    return model, train, test, criterion, optimizer
 

def save_image(output, size):
    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = np.array(output[0][0,:,:].cpu())
    cur[:,:,1:] = np.array(128*output[1].cpu().permute(2,1,0))
    imsave("results/"+model.get_name()+"/img_result.png", (lab2rgb(cur)*255).astype(np.uint8))