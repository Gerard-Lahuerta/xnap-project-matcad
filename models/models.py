import torch.nn as nn
import numpy as np

# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
#############################################################################

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1, stride=2),
            nn.ReLU()  
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor= 2),
            #nn.MaxUnpool2d(2, stride=1),
            nn.ConvTranspose2d(32, 32, (3,3), padding=1, stride = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2),
            #nn.MaxUnpool2d(2, stride=1),
            nn.ConvTranspose2d(32,16, (3,3), padding=1, stride = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2),
            #nn.MaxUnpool2d(2, stride=1),
            nn.ConvTranspose2d(16, 2, (3,3), padding=1, stride = 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3,3), padding=0, stride=1),
            nn.ReLU()
        )


        self.decoder = nn.Sequential( 
            nn.MaxUnpool2d(2, 2),
            nn.Conv2d(64, 64, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.MaxUnpool2d(2,2),
            nn.Conv2d(32, 2, (3,3), padding=0, stride=1),
            nn.ReLU(),
            nn.Conv2d(2, 2, (3,3), padding=0, stride=1),
            nn.Tanh(),
            nn.MaxUnpool2d(2,2)
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
