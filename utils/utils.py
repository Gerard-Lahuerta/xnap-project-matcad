import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *
import matplotlib as plt
from torchvision.utils import make_grid

def get_data(slice=1, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

'''
def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
'''
    
###################################################################################

def get_data_model_1(path):
    image = Image.open(path)
    imgGray = image.convert('L')

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(imgGray)
    X = tensor.reshape(1, 400, 400, 1)

    return X

def make(model_type, config, device = "cuda"):

    if model_type == "Model 1":
        train, test = get_data_model_1(), get_data_model_1(train = False)
        model = Model1()#.to(device)
        criterion = nn.MSELoss()
    else:
        assert(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    return model, train, test, criterion, optimizer


def plot_image(img):
    img = img.clamp(0, 1) # Ensure that the range of greyscales is between 0 and 1
    npimg = img.numpy()   # Convert to NumPy
    npimg = np.transpose(npimg, (2, 1, 0))   # Change the order to (W, H, C)
    plt.imshow(npimg)
    plt.show()


def show_image(img):
    plot_image(make_grid(img.detach().cpu().view(-1, 1, 28, 28).transpose(2, 3), nrow=2, normalize = True))
    plot_image(make_grid(img.detach().cpu().view(-1, 1, 28, 28).transpose(2, 3), nrow=2, normalize = True))   
