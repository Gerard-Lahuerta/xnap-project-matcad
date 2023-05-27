[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11121442&assignment_repo_type=AssignmentRepo)


# Colouring greyscale images
Final project of "Xarxes Neuronals i Aprenentatge Profund".

Testing and training different CNN encoder-decoder models with the objective of automatically add colour on greyscale images.

We expose the most rellevant conclusions and significant information about each model in the following sections.



## Objectives
- Test the habilities we have build during this course by being able to do our own train and test using new models.
- Be able to add colour to greyscale images.
- Understand the operation of an autoencoder.
- Acquire more agility in the use of github and new platforms such as azure VMs.
- Being able to analyze the behavior of each model.


## Requirements
The requirements needed to include in each file are shown in the table below, marked with an X:

| Import | main.py |  utils.py | models.py | train.py | test.py |
| :-------- | :------- | :------- | :------- | :------- | :------- |
| `random` | X | X | | | |
| `wandb` | X | | | X | X |
| `numpy` | X | X | | | |
| `torch` | X | X | X | X | X |
| `warnings` | X | | | | | 
| `tqdm` |  | X | | X | X |
| `torchvision` |  | X | X | | |
| `os` |  | X | | | |
| `skimage` |  | X | | | |

## Model 1
Also knows as the alpha version, this model is a starting poing, helping to understand how an image is transformed into RGB pixel values and later translated into LAB pixel values, changing the color space. 

The data set to test and train the model initially consisted of a single image to train, which made the model learn only how to paint faces. 

Subsequently, the data set of model 2 (consisting of 8 images) was used to train and test the model, where it was observed that the model was capable of coloring not only faces, but complete people with different backgrounds.

This model is the fastest, providing results for 1000 epochs in a few minutes and coloring the photos quite accurately.

CONFIRMAS SI SE HA PROBADO CON EL CAPTIONING!!!!!!!!!!!!

AÑADIR CONCLUSIONES Y COSIÑAS

COMENTAR DEL ENCODER DECODER

## Model 2
Also knows as the beta version, the model is based in the alpha version. It has a similar convolution network but has a differed purpose. 
It is designed to use more than one image to train the network (avoid memorization and starting to have a model able to learn).
Nevertheless, and despite of his bigger network respect its precessor, does not obtain quite good results colorizing a whole group of images.

After testing the model with various datasets, from the one offered in the starting point with 8 images to our own dataset consisting of more than 2000 images of dogs, it has been observed that the beta version is only capable of giving the pictures grayish and brown tones.

This behavior is not compleatly understood but the register while training given makes us conclude the following statments:

- 
-
-

However, this conclusions may be wrong and this behavior (that has in a similar way the alpha version in the eraliest epochs of the trainig process) could be explained by not having enough epochs to train.

Although the statements obteined in the register while trainig or the hypothesis of not having enough epochs, the new model (model 3) gives a new possibility branch to give better performance and in the early testing show quite better results.

Moreover, trying to obtain more conclusions on what makes de beta version give that bad results, the alpha version was tested with the beta dataset (to compare results because this version was not thought to train with more than one image) and gave significant better results (but with a huge amount of epochs respect the training done with the dataset it was testet initialy).

Gotta point out that this last behaviour (that the alpha verions obtained better results than it successor, the beta version), is a relevant evidence of the hypothesis explained that the beta version does not train with enough epochs. But with the right amount of time it would need to determine this possibility or to obtain better results; the option of dismiss this model and take to the model 3 became our decision.

## Model 3


## Other autoencoders
In the file models.py, we have also included two extra models, named `ConvAE`and `ColorizationNet`.

While `ConvAE` is a very simple general autoencoder we practiced in class, `ColorizationNet` is a CNN found online designed specifically for the task of colorizing images.

It is important to note that while the first one gave completely erroneous results, it has been observed that ColorizationNet actually gives color to the images with few epochs, although not completely correctly.

Both have been used in the project to debug and verify the correct operation of the code when the models 1, 2 and 3 were still being adapted to pytorch. 

Because the main function of these models was to facilitate code and bug cleanup, they have not been thoroughly discussed nor will their operation be explained, as they have not been added to the project to colorize the images.



## Rellevant information

--> hemos probado diversos optimizadores

--> intentamos hacer un data augmentation
## General conclusions
## Authors

- Gerard Lahuerta Martín --- 1601350
- Ona Sánchez Núñez --- 1601181
- Bruno Tejedo Miniéri --- 1533327


## Documentation

 - [Starting poing](https://github.com/emilwallner/Coloring-greyscale-images.git)
 - [Data Starting point](https://github.com/emilwallner/Coloring-greyscale-images.git)
 - [Dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
 - [Captioning dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)


Xarxes Neuronals i Aprenentatge Profund
Grau de Computational Mathematics & Data analyitics, UAB, 2023
