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

## Model 2
## Model 3
## Other autoencoders
In the file models.py, we have also included two extra models, named `ConvAE`and `ColorizationNet`.

While `ConvAE` is a very simple autoencoder we practiced in class, `ColorizationNet` is a CNN we found on the internet.

Both have been used in the project to debug and verify the correct operation of the code, when the models 1, 2 and 3 were still being adapted to pytorch. 

It is important to note that while the first one gave completely erroneous results, it has been observed that colorizationNet colorizes the images, although not completely correctly.

AÑADIR ALGUN DATO SOBRE LOSS O COSITA
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
