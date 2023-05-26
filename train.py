from tqdm.auto import tqdm
import wandb
from utils.utils import shuffle, save_1_image
import torch

'''
def train(model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)
'''

def train_batch(image, label, model, optimizer, criterion, device="cuda"):
    images, labels =image.to(device), label.to(device)

    # Forward pass ➡
    outputs = model(images)

    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "Train Loss": loss}, step=example_ct)


############################################################################################

def create_minibatch(loader, n_batch):
    n = len(loader[0])//n_batch
    aux = [[],[]]
    aux_aux = [[],[]]
    k = 0

    for i,j in zip(loader[0], loader[1]):
        aux_aux[0].append(i)
        aux_aux[1].append(j)
        k += 1
        if k == n:
            k = 0
            aux[0].append(aux_aux[0])
            aux[1].append(aux_aux[1])
    
    return aux


def train_batch_model(loader, model, optimizer, criterion, ct, e_info, shuffle_loader = True, n_batch = 4):
    if shuffle_loader:
        loader = shuffle(loader)


    for images, labels in zip(loader[0], loader[1]):
        loss = train_batch(images, labels, model, optimizer, criterion)
        ct[1] += len(images)
        ct[0] += 1

        # Report metrics every 25th batch
        if ((ct[0] + 1) % 25) == 0:
            e_info[1].set_postfix({'Loss': f"{loss:.6f}"})
            train_log(loss, e_info[0], e_info[0])
    return [ct[0], ct[1]]


def train_model(model, loader, criterion, optimizer, config, n_show_image = 50):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more! 
    wandb.watch(model, criterion, log="all", log_freq=10)

    model.train()

    # Run training and track with wandb       
    ct = [0,0] # batch_ct, example_ct
    X = loader[0][0].to("cuda")

    epochs = tqdm(range(config["epochs"]), desc="Train {0}: ".format(model.get_name()))
    e_info = [None, epochs]

    for epoch in epochs:
        e_info[0] = epoch
        ct = train_batch_model(loader, model, optimizer, criterion, ct, e_info)

        if epoch%n_show_image == 0:
            with torch.no_grad():
                AB = model(X)
                size = X.shape[2]
                save_1_image(AB, X, size, "image_log", "/img_"+str(epoch))
