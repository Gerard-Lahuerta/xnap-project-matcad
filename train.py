"""
    Auxiliar file to train the models
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo -- 1533327

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""


###### IMPORTS #########################################################################################################

from tqdm.auto import tqdm
import wandb
from utils.utils import shuffle, save_1_image, delete_files
import torch


###### TRAINING OF THE MODEL ###########################################################################################

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "Train Loss": loss}, step=example_ct)

    
def train_batch(image, label, model, optimizer, criterion, device="cuda"):
    images, labels = image.to(device), label.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_batch_model(loader, model, optimizer, criterion, ct, e_info, shuffle_loader=True, n_batch=4):
    if shuffle_loader:
        # Shuffling of the images/labels
        loader = shuffle(loader)

    for images, labels in zip(loader[0], loader[1]):
        # Training of each batch
        loss = train_batch(images, labels, model, optimizer, criterion)
        # Image counter
        ct[1] += len(images)
        # Batch counter
        ct[0] += 1

        # Report metrics every 25th batch
        if ((ct[0] + 1) % 25) == 0:
            e_info[1].set_postfix({'Loss': f"{loss:.6f}"})
            train_log(loss, e_info[0], e_info[0])

    return [ct[0], ct[1]]


def train_model(model, loader, criterion, optimizer, config, n_show_image=10):
    # Watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Delete the register log of the previous train
    delete_files()

    # Run training and track with wandb
    model.train()

    ct = [0, 0]  # batch_ct, image_ct
    L = loader[0][0].to("cuda")

    # Tracking the evolution of the train
    epochs = tqdm(range(config["epochs"]), desc="Train {0}: ".format(model.get_name()))
    e_info = [None, epochs]

    # Training of the batches for all epochs
    for epoch in epochs:
        e_info[0] = epoch

        ct = train_batch_model(loader, model, optimizer, criterion, ct, e_info)

        # Track of the colorizaton
        if epoch % n_show_image == 0:
            with torch.no_grad():
                AB = model(L)
                size = L.shape[2]
                save_1_image(AB, L, size, "image_log", "/img_" + str(epoch))
