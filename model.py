import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes,dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
          
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # H/2, W/2
            nn.MaxPool2d(2, 2),           

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # H/4, W/4
            nn.MaxPool2d(2, 2),           

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # H/8, W/4
            nn.MaxPool2d((2, 1)),         
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # H/16, W/4
            nn.MaxPool2d((2, 1)),         
            nn.Dropout(0.3),
            # H/32=1, W/4
            nn.Conv2d(512, 512, kernel_size=(2, 1)),  
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )
        self.dropout_fc=nn.Dropout(0.6)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)                   
        B, C, H, W = x.shape
        x = x.reshape(B, C * H, W)         
        x = x.permute(0, 2, 1)            
        x, _ = self.rnn(x)   
        x=self.dropout_fc(x)             
        x = self.fc(x)                     
        return F.log_softmax(x.permute(1, 0, 2), dim=2)  