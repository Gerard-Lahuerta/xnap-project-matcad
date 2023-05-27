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
    '''
    INPUT:
        --> loss: torch.Tensor, has to contain the loss of the model. 
        --> example_ct: float, number of images that the model has used to train.
        --> epoch: int, actual epoch of the trainning proces of the model.

    OUTPUT:
        --> None.

    ABOUT IT:
        --> Generates a register in W&B (weights and Bias).
    '''
    # Where the magic happens
    wandb.log({"epoch": epoch, "Train Loss": loss}, step=example_ct)

    
def train_batch(image, label, model, optimizer, criterion, device="cuda"):
    '''
    INPUT:
        --> image: torch.Tensor, matrix of dimensions 1 x 1 x size x size to train the model
        --> label: torch.Tensor, matrix of dimensions 1 x 2 x size x size to compare the model output.
        --> model: CNN encoder-decoder model (pytorch).
        --> optimizer: function, used to optimize the model weights.
        --> criterion: function, used to obtain the loss of the model and optimize its weights.
        --> device = "cuda": string, device where the training process is done.

    OUTPUT:
        --> loss: torch.Tensor, contains the loss and the backpropagation information of the model.

    ABOUT IT:
        --> Pass the image and label tensor into  the "device", obtains the output of the model (using
            the "image" as the input), compares the output and the label (obtainig the loss) and 
            optimize the model using backpropagation.

    RELEVANT INFORMATION:
        --> By default, device is set to "cuda" to distribute the training proces into the GPU.
    '''

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


def train_batch_model(loader, model, optimizer, criterion, ct, e_info, shuffle_loader=True):
    '''
    INPUT:
        --> loader: list, images in grey-scale (input) and images in AB-scale (label) respectively.
        --> model: CNN encoder-decoder model (pytorch).
        --> optimizer: function, used to optimize the model weights.
        --> criterion: function, used to obtain the loss of the model and optimize its weights.
        --> ct: list, contains the total batch and the total of images used to train (respectively).
        --> e_info: list, contains information of epochs: trainig epoch and tqdm-for (respectively).
        --> shuffle_loader = True: bool, used to decide if we shuffle the loader or not.

    OUTPUT:
        --> ret: list, retuns the actualizated ct list passed by parameter.

    ABOUT IT:
        --> Uses every image passed in the loader to train the model, it prints and logs (using W&B)
            the informatino of the training every 25 batches (showing the loss of the model in one,
            unknown, image.

    RELEVANT INFORMATION:
        -->By default "shuffle_loader" is set to True to avoid overfitting / memorization of the model.
    '''

    if shuffle_loader:
        # Shuffling of the images/labels
        loader = shuffle(loader)

    for images, labels in zip(loader[0], loader[1]):
        # Training of each batch
        loss = train_batch(images, labels, model, optimizer, criterion)
        # Image counter
        ct[1] += 1
        # Batch counter
        ct[0] += 1

        # Report metrics every 25th batch
        if ((ct[0] + 1) % 25) == 0:
            e_info[1].set_postfix({'Loss': f"{loss:.6f}"})
            train_log(loss, e_info[0], e_info[0])

    return [ct[0], ct[1]]


def train_model(model, loader, criterion, optimizer, config, n_show_image=10):
    '''
    INPUT:
        --> loader: list, images in grey-scale (input) and images in AB-scale (label) respectively.
        --> model: CNN encoder-decoder model (pytorch).
        --> optimizer: function, used to optimize the model weights.
        --> criterion: function, used to obtain the loss of the model and optimize its weights.
        --> config: dict, has to contain the key "epochs".
            -> config["epochs"]: int, epochs of the training process of the model.
        --> n_show_image = 10: int, number of epochs to log one test image; show the training proces.

    ABOUT IT:
        -->
    '''
    
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
