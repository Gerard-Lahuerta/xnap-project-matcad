import torch.nn as nn
import numpy as np
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
            nn.Upsample(scale_factor= 2), #bilinear, nearest, bicubic
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
        #print(x.shape)
        #print(fusion_aux.shape)
        fusion_aux = self.conv_fusion(fusion_aux)
        nn.ReLU()
        return fusion_aux
        ''''
        embed_input = self.fusion_repeat(embed_input)
        embed_input = embed_input.view(embed_input.size(0), 32, 32)
        embed_input = embed_input.unsqueeze(1)
        fusion_input = torch.cat((x, embed_input), dim=1)
        print(fusion_input.shape)
        return self.fusion_conv(fusion_input)
        '''

    def forward(self, x):
        #embed_input = torch.Tensor(1, 1000).to("cuda") #el 1 és el batch size
        x = self.encoder(x)
        #print(x.shape)
        x = self.fusion(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x

    def get_name(self):
        return self.name
