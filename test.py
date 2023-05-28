"""
    Auxiliar file to test the models
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo -- 1533327

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""


###### IMPORTS #########################################################################################################

import wandb
import torch

from utils.utils import save_image
from tqdm.auto import tqdm


###### TESTING OF THE MODEL ############################################################################################

def test(model, test, criterion, device="cuda"):
    '''
    INPUT:
        --> model: CNN encoder-decoder model (pytorch).
        --> test: list, images in grey-scale (input) and images in AB-scale (label) respectively.
        --> criterion: function, used to obtain the loss (error) of the model.
        --> device = "cuda": string, device where the test process is done.

    OUTPUT:
        --> output: list, colorized images produced by the model.
        --> input: list, gray-scale images used to produce the output.
        --> loss: float, average loss obtained by the model with all the images in the test parameter.

    ABOUT IT:
        --> Testing proces of the model, compute the loss (error) of the model for each image in the
            test list passed by parameter and shows in real time the testing process ( using tqdm, 
            useful with large test datasets). 

    RELEVANT INFORMATION:
        --> By default, device is set to "cuda" to distribute the testing proces into the GPU.
    '''

    model.eval()

    # Inicialization of parameters
    loss = 0
    output = []
    input = []
    n = len(test[0])

    # Tracking the evolution of the test
    test = tqdm(zip(test[0], test[1]), desc="Testing "+model.get_name())

    # Testing images
    for L, AB in test:
        X = L.to(device)
        Y = AB.to(device)

        with torch.no_grad():
            out = model(X)

        # compute training reconstruction loss
        loss += criterion(out, Y)
        input.append(X) # list with the greyscale images
        output.append(out) # resulting colorized images

    return output, input, loss/n


def test_model(model, test_loader, criterion, save: bool = True):
    '''
    INPUT:
        --> model: CNN encoder-decoder model (pytorch).
        --> test_loader: list, images in L-scale (input) and images in AB-scale (label) respectively.
        --> criterion: function, used to obtain the loss (error) of the model.
        --> save = True: bool, decide if the images are going to be saved in the model's results folder.

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Tests the model with all the images contained in the "test_loader" parameter, register the
            loss using W&B (Weights and Bias) and save (if desired) the whole output of the testing 
            proces into the specific model's results folder.

    RELEVANT INFORMATION:
        --> By default, save is set to "True" to register the colorized images generates by the model.
    '''

    # Set model in evaluation mode
    model.eval()
    size = test_loader[0][0].shape[2]

    # Run testing and track with wandb
    with torch.no_grad():
        output_AB, output_L, loss = test(model, test_loader, criterion)
        wandb.log({"Test Loss": loss})

    # Saving the resulting colorized images
    if save:
        path = "results/" + model.get_name()
        save_image(output_AB, output_L, size, path)
