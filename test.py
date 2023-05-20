import wandb
import torch
from utils.utils import show_image
from skimage.io import imsave
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np

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


def test(model):

    if model.get_name() == "Model 1":
        test_model = test_model_1
    elif model.get_name() == "Model 2":
        test_model = test_model_2
    elif model.get_name() == "Model 3":
        test_model = test_model_3
    else:
        assert(False)

    model.eval()

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


def test_model_1(model, test, label, criterion, device="cuda", save:bool= True):
    model.eval()

    test = test.to(device)
    label = label.to(device)

    with torch.no_grad():
        output = model(test)
    
    # compute training reconstruction loss
    loss = criterion(output, label)

    # display the epoch training loss
    print("Test loss = {:.6f}".format(loss))

    if(save):
        cur = np.zeros((400, 400, 3))
        cur[:,:,0] = np.array(test[0][0,:,:].cpu())
        cur[:,:,1:] = np.array(128*output[0].cpu().permute(2,1,0))
        imsave("results/"+model.get_name()+"/img_result.png", (lab2rgb(cur)*255).astype(np.uint8))
        imsave("results/"+model.get_name()+"/img_gray_version.png", ((rgb2gray(lab2rgb(cur)))*255).astype(np.uint8))
    #wandb.log({"Loss": loss})


def test_model_2(model, test, criterion, device="cuda", save: bool = True):
    model.eval()

    testa = test[0].to(device)
    label = test[1].to(device)

    with torch.no_grad():
        output = model(testa)

    # compute training reconstruction loss
    loss = criterion(output, label)

    # display the epoch training loss
    print("Test loss = {:.6f}".format(loss))
    # show_image(label)
    # show_image(output)

    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:, :, 0] = test[i][:,:,0]
        cur[:, :, 1:] = output[i]
        imsave("result/img_" + str(i) + ".png", lab2rgb(cur))
        # wandb.log({"Loss": loss})


def test_model_3():
    pass