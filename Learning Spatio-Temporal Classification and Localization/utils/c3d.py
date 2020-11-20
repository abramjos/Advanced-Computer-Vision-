import torch
import torch.nn as nn
import torch.nn.init as init
import os
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)



def weights_init(m):
    if isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def xavier(param):
    init.xavier_uniform(param)

# C3D Model
class C3D(nn.Module):
    def __init__(self, no_classes=101):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        #init.xavier_normal(self.group1.state_dict()['weight'])
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group2.state_dict()['weight'])
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group3.state_dict()['weight'])
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group4.state_dict()['weight'])
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group5.state_dict()['weight'])

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),               #
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc3 = nn.Sequential(
            nn.Linear(2048, no_classes),
            nn.Softmax(dim=1))           #101

        self.fc4 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5))           #101

        self.fc5 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5))           #101

        self.fc6 = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid())           #101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4,
            self.group5
        )

        self._classifier_1 = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

        self._classifier_2 = nn.Sequential(
            self.fc4,
            self.fc5,
            self.fc6
        )
 
    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out_1 = self._classifier_1(out)
        out_2 = self._classifier_2(out)
        return (out_1,out_2)

if __name__ == '__main__':
    
    c3d = C3D().cuda(0)
    c3d.apply(weights_init)

    inx =torch.rand(( 16, 3, 16, 112, 112 )).to('cuda')
    c3d.eval()
    axs = c3d(inx)