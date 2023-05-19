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
    
    print("uf")

    # Forward pass ➡
    outputs = model(images)

    print(outputs.size())
    print(labels.size())

    loss = criterion(outputs, labels)
    
    print("que")

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    print("rico")

    # Step with optimizer
    optimizer.step()

    print("mami")

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


############################################################################################

def train_model(model, image, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    #wandb.watch(model, criterion, log="all", log_freq=10)

    #model.train()

    # Run training and track with wandb
    total_batches = config["epochs"]
    batch_ct = 0
    for epoch in tqdm(range(config["epochs"])):
        loss = train_batch(image, image, model, optimizer, criterion)
        batch_ct += 1
        print("hola")

        # Report metrics every 25th batch
        if ((batch_ct + 1) % 25) == 0:
            print(f"Loss after {str(batch_ct).zfill(5)} examples: {loss:.3f}")