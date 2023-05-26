"""
    Program for the colorization of white and black images
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo Miniéri

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""


###### IMPORTS #########################################################################################################

import random
import wandb

import numpy as np
import torch
import torch.nn as nn

from train import *
from test import *
from utils.utils import *

import warnings


###### CONFIGURATIONS ##################################################################################################

warnings.filterwarnings('ignore')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### MAIN PROGRAM ####################################################################################################

if __name__ == "__main__":
    wandb.login()

    # Configuration of the image colorization
    config = dict(
        model = "Model 1",
        epochs = 1000,

        learning_rate = 0.0001,
        optimizer = "Adam",
        criterion = "MSE",

        data_set = "default", # "data/data_2/Train/", data/PERROS/", "data/Captioning/"
        split = 0.25,

        save_weights = True,
        import_weights = False,
        save_images = True,

        train = True,
        test = True
        )

    # Init wandb for tracking the evolution
    with wandb.init(project="proyect-xnap", config=config):

        # Building of the model with the chosen configuration
        model, train_loader, test_loader, criterion, optimizer = make(config=config)

        if config["import_weights"]:
            model = import_model(model)

        if config["train"]:
            train_model(model, train_loader, criterion, optimizer, config)

        if config["test"]:
            test_model(model, test_loader, criterion, save = config["save_images"])

        if config["save_weights"]:
            save_model(model)
