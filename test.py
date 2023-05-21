import wandb
import torch
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np

from utils.utils import save_image
from tqdm.auto import tqdm

'''
def test(model, test_loader, device="cuda", save:bool= True):
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    if save:
        print(len(images))
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model,  # model being run
                          images,  # model input (or a tuple for multiple inputs)
                          "model.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
        wandb.save("model.onnx")
'''

########################################################################################################################


def test_model(model, test_loader, criterion, save:bool = True):
        
    model.eval()
    size = test_loader[0][0].shape[2]

    with torch.no_grad():
        output_AB, output_L, loss = test(model, test_loader, criterion)
        print("Test loss = {:.6f}".format(loss))
        #wandb.log({"Loss": loss})

    if save:
        path = "results/"+model.get_name()
        save_image(output_AB, output_L, size, path)

     

def test(model, test, criterion, device="cuda", save: bool = True):
    model.eval()

    loss = 0
    output = []
    input = []
    n = len(test[0])


    test = tqdm(zip(test[0], test[1]), desc="Testing "+model.get_name())
    for L, AB in test:
        X = L.to(device)
        Y = AB.to(device)

        with torch.no_grad():
            out = model(X)
        # compute training reconstruction loss
        loss += criterion(out, Y)
        input.append(X)
        output.append(out)

    return output, input, loss/n
