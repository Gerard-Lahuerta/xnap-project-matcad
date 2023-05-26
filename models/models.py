"""
    Auxiliar file with the encoder-decoder models
    @Version: 1.0.0

    @Authors:
            --> Ona Sánchez -- 1601181
            --> Gerard Lahuerta -- 1601350
            --> Bruno Tejedo Miniéri

    @Copyright (c) 2023 All Right Reserved

    Information about the program in: https://github.com/DCC-UAB/xnap-project-matcad_grup_6.git
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

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

        self.name = "Model 1"

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
            nn.ConvTranspose2d(32, 32, (3,3), padding=1, stride = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2),
            nn.ConvTranspose2d(32,16, (3,3), padding=1, stride = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2),
            nn.ConvTranspose2d(16, 2, (3,3), padding=1, stride = 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_name(self):
        return self.name

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        
        self.name = "Model 2"

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3,3), padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3,3), padding=1, stride=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential( 
            nn.Upsample(scale_factor= 2),
            nn.ConvTranspose2d(128, 64, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor= 2),
            nn.ConvTranspose2d(64, 32, (3,3), padding=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, (3,3), padding=1, stride=1),
            nn.Tanh(),
            nn.Upsample(scale_factor= 2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
   
    def get_name(self):
        return self.name

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        
        self.name = "Model 3"

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), padding = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, (3, 3), padding = 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 32, (3, 3), padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (3, 3), padding = 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, (3, 3), padding = 1, stride = 1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )

        self.conv_fusion = nn.Conv2d(1256, 256, (1,1), padding = 0, stride = 1)

    def fusion(self, x):
        input_aux = torch.randn(1,1000, 32, 32).to("cuda")
        fusion_aux = torch.cat((x,input_aux), 1)
        fusion_aux = self.conv_fusion(fusion_aux)
        nn.ReLU()
        return fusion_aux


    def forward(self, x):
        x = self.encoder(x)
        x = self.fusion(x)
        x = self.decoder(x)
        return x

    def get_name(self):
        return self.name
    
#######################################################################################

class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, 2, stride=2, padding=1),
            nn.Sigmoid()    
        )

        self.name = "ConVAE"

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_name(self):
        return self.name


import torchvision.models as models

class ColorizationNet(nn.Module):
    def __init__(self, input_size=128):
        super(ColorizationNet, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        ## First half: ResNet
        resnet = models.resnet18(num_classes=365) 
        # Change first conv layer to accept single-channel (grayscale) input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        # Extract midlevel features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        ## Second half: Upsampling
        self.upsample = nn.Sequential(     
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

        self.name = "ColorizationNet"

    def forward(self, input):

        # Pass input through ResNet-gray to extract features
        midlevel_features = self.midlevel_resnet(input)

        # Upsample to get colors
        output = self.upsample(midlevel_features)
        return output

    def get_name(self):
        return self.name
