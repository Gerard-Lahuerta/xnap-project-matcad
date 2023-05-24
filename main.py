import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *
from tqdm.auto import tqdm

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

# remove slow mirror from list of MNIST mirrors
torchvision.datasets.MNIST.mirrors = [mirror for mirror in torchvision.datasets.MNIST.mirrors
                                      if not mirror.startswith("http://yann.lecun.com")]



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
    #wandb.login()

    config = dict(
        model = "Model 2",
        epochs = 100,

        learning_rate = 0.01,
        sch = "StepLR",
        params = {"step_size": 30, "gamma": 0.1},

        optimizer = "SGD",
        criterion = "MSE",

        data_set = "default",
        split = 0.95,

        save_weights = True,
        import_weights = False,
        save_images = True,

        train = True,
        test = True
        )

    model, train_loader, test_loader, criterion, optimizer = make(config=config)

    if config["import_weights"]:
        model = import_model(model)
    
    if config["train"]:
        train_model(model, train_loader, criterion, optimizer, config)

    if config["test"]:
        test_model(model, test_loader, criterion, save = config["save_images"])

    if config["save_weights"]:
        save_model(model)

