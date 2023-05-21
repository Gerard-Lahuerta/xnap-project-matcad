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
        epochs=1000,
        classes=10,
        kernels=[16, 32],
        batch_size=128,
        learning_rate=0.01,
        dataset="MNIST",
        architecture="CNN")
    
    save_weights_model = False
    import_weights_model = True
    
    #model = model_pipeline(config)

    model, train_loader, test_loader, criterion, optimizer = make(model_type="Model 1", config=config)

    # and use them to train the model
    if not import_weights_model:
        train_model(model, train_loader, criterion, optimizer, config)
    else:
        model = import_model(model)

    # and test its final performance
    test_model(model, test_loader, criterion)

    if save_weights_model:
        save_model(model)
    #test_model2(model, test_loader, criterion)

