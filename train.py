from tqdm.auto import tqdm
import wandb

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
    images, labels = image.to(device), label.to(device)

    # Forward pass ➡
    #print(image.shape)
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
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    #print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


############################################################################################


def train_batch_model_1(image, label, model, optimizer, criterion, epochs, epoch, batch_ct, example_ct):
    loss = train_batch(image, label, model, optimizer, criterion)
    batch_ct += 1
    example_ct += len(image)

    # Report metrics every 25th batch
    if ((batch_ct + 1) % 25) == 0:
        epochs.set_postfix({'Loss': f"{loss:.6f}"})
        train_log(loss, example_ct, epoch)


def train_batch_model_2():
    pass


def train_batch_model_3():
    pass


def train_model(model, image, label, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    #wandb.watch(model, criterion, log="all", log_freq=10)

    model.train()

    # Run training and track with wandb

    if model.get_name() == "Model 1":
        train_batch_model = train_batch_model_1
    elif model.get_name() == "Model 2":
        train_batch_model = train_batch_model_2
    else:
        train_batch_model = train_batch_model_3

    batch_ct = 0
    example_ct = 0
    epochs = tqdm(range(config["epochs"]), desc="Train: ")
    for epoch in epochs:
        train_batch_model(image, label, model, optimizer, criterion, epochs, epoch, batch_ct, example_ct)
