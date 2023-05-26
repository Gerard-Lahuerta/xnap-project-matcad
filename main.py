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

import random
import wandb

import numpy as np
import torch
import torch.nn as nn

from train import *
from test import *
from utils.utils import *

import warnings
warnings.filterwarnings('ignore')

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




'''
def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=cfg):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model
'''


if __name__ == "__main__":
    wandb.login()

    config = dict(
        model = "Model 1",
        epochs = 1000,

        learning_rate = 0.0001,
        sch = "StepLR",
        params = {"step_size": 30, "gamma": 0.1},

        optimizer = "Adam",
        criterion = "MSE",

        data_set = "default", #"data/Captioning/",#"data/data_2/Train/", #"data/Captioning/", ##data/PERROS/",#"default", "data/Captioning/" 
        split = 0.25,

        save_weights = True,
        import_weights = False,
        save_images = True,

        train = True,
        test = True
        )

    with wandb.init(project="proyect-xnap", config=config):

        model, train_loader, test_loader, criterion, optimizer = make(config=config)

        if config["import_weights"]:
            model = import_model(model)

        if config["train"]:
            train_model(model, train_loader, criterion, optimizer, config)

        if config["test"]:
            test_model(model, test_loader, criterion, save = config["save_images"])

        if config["save_weights"]:
            save_model(model)

