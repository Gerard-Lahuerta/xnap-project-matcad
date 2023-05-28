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
Also known as the alpha version, this model is a starting point, helping to understand how an image is transformed into RGB pixel values and later translated into LAB pixel values, changing the color space. 

The data set to test and train the model initially consisted of a single image to train, which made the model learn only how to paint faces. 

Subsequently, the data set of model 2 (consisting of 8 images) was used to train and test the model, where it was observed that the model was capable of coloring not only faces, but complete people with different backgrounds.

This model is the fastest, providing results for 1000 epochs in a few minutes and coloring the photos quite accurately, but, if this model is trained with diferents datasets, his behaviour will change radicaly depending on its hyperparameters.

Depending on the amount of epochs and images used in the trainig process the model will generate a colorized image (as a test output) that goes to:
- grey-scale (no changes)
- veige-filter (does not learn and give a neutral color, sum of all the primary colors that has the less "distance" between all them)
- colourized but not all of the image (not learn enough)
- colourized semi-perfectly or perfectly (learned)
- colourized but the colour not maches, in a significant way, the image (overfitted learning)

To adjust this problem we must augment in a semi-proportional form the epochs and the images used in the train process.
This effect (augmenting the epochs proportionally to the number of images used to train) has the next effect; the time needed to train the model increases critically.

However, if the target to colorize is similar to the images learned in training, the model can  provide quite good results with less images and a significant amount of epochs.

To sum up, this model, despite being the early prototipe and having some limitations is versatil enough to adapt diferent datasets and necesities.


## Model 2
Also known as the beta version, the model is based on the alpha version. It has a similar convolution network but has a different purpose. 
It is designed to use more than one image to train the network (avoid memorization and start to have a model able to learn).
Nevertheless, despite his bigger network respect for its predecessor, does not obtain quite good results colorizing a whole group of images.

After testing the model with various datasets, from the one offered in the starting point with 8 images to our dataset consisting of more than 2000 images of dogs, it has been observed that the beta version is only capable of giving the pictures grayish and brown tones.

This behavior is not completely understood but the register while training given makes us conclude the following statements:

- Architecture problem, CNN encoder-decoder is not appropriate for the task.
- Wrong training for this model.
- The appropriate parameters for the model are not used (learning rate or optimizer, for example). Despite having tried various optimizers, losses, and learning rates, the model does not seem to learn correctly.

One of the first suspicions about the malfunction of the model was the lack of epochs, however, we studied the model using 2000 epochs and saw that it was only capable of coloring in brown tones and some greenish areas, but in an almost imperceptible way.

The new model (model 3) gives a new possibility branch to give better performance and in the early testing shows quite better results.

Moreover, in trying to obtain more conclusions on what makes de beta version gives that bad results, the alpha version was tested with the beta dataset (to compare results because this version was not thought to train with more than one image) and gave significantly better results (but with a huge amount of epochs respect the training done with the dataset it was tested initially).


## Model 3
Model 3 is called the full-version model and, theoretically,  it combines a deep Convolutional Neural Network encoder-decoder trained from scratch with high-level features extracted from the Inception-ResNet-v2 pre-trained model (pre-trained classifier on ImageNet dataset). 

In practice, the Inception-ResNet-v2 is not implemented (or we could not find it) in Pytorch and does not have an equivalent extension.

This issue will have a significant consequence on the learning process, results generated, and behavior of the model.

In general terms, the predictions made by this model are not quite satisfying. Even though it works better than the model 2, it works worse than the model 1. 

Different combinations of parameters have been tried to see how it works and for which the model obtains better results.
Although the appliances of other optimizers were not mentioned before (like Adagrad, RMSprop, etc.), criterions (such as RMSE or MAE), and other parameter configurations, the model could not obtain much better predictions.

This is explained by the low amount of epochs used to train the model (indifferently the images used to train).
This should not be a problem if the model weights are initialized with the Inception-ResNet-v2, but this is not the case :'(.

Moreover, the model has been trained and tested with different datasets, and consequently, getting clear results.
- "Captioning" dataset makes the model have (initially) random behavior. Throughout the training epochs, the model stabilizes and obtains relatively good results (but not enough for the complexity of the model).
- "PERROS" dataset shows better results, obtaining some good predictions with 100 epochs. Nevertheless, it does not obtain the same quality in all the types of images (generating better results for "chow chow" dogs than others).


## Other autoencoders
In the file models.py, we have also included two extra models, named `ConvAE`and `ColorizationNet`.

While `ConvAE` is a very simple general autoencoder we practiced in class, `ColorizationNet` is a CNN found online designed specifically for the task of colorizing images.

It is important to note that while the first one gave completely erroneous results, it has been observed that ColorizationNet gives color to the images with few epochs, although not completely correct.

Both have been used in the project to debug and verify the correct operation of the code when models 1, 2, and 3 were still being adapted to Pytorch. 

Because the main function of these models was to facilitate code and bug cleanup, they have not been thoroughly discussed nor will their operation be explained, as they have not been added to the project to colorize the images.



## Rellevant information
- The ConvAE and ColorizationNet models have been used for debugging and comparison purposes only.

- The folder image_log contains the register of the last training.

- Wandb has been used to supervise the training of the models.

- Both the models and the code using Pytorch.

- Each function and model has its informative doc.

- The project is designed so that any model can be executed under any condition just by changing the main configuration.

## General conclusions

- Once the models worked to a better or worse extent, various functions were created to perform data augmentation. After some tests and seeing that it did not work, we realized that creating new data by using crops or rotations on the initials does not make sense when talking about learning to color images.

- It has been observed that the models that manage to color need many epochs (greater than 1000) when training them with a relatively high number of images.

- Several tests were carried out with new optimizers, criteria, adaptive learning rates, and with mini-batches, but no significant improvement was achieved.

- The best model is Alpha whereas Beta is the worst.

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
