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
        --> test: list of 2 list with grey-scale and the AB images (respectively).
        --> criterion: function, used to optimize the model while it is training.

    OUTPUT:
        --> output:
        --> input:
        --> loss/n:

    ABOUT IT:
        -->
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
        --> test_loader: list of 2 list (input test list and label test list respectively)
        --> criterion: function, used to optimize the model while it is training.
        --> save = True: bool, marker to decide if we save (or not) the resulting images.

    ABOUT IT:
        -->
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
