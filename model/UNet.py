from torch import nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
        
    def forward(self, img):
        return self.layer(img)
    
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(2)

    def forward(self, img):
        return self.layer(img)

class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, 2, 2),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self, img):
        img1, img2 = img
        o = self.layer(img1)
        return torch.cat((o, img2), dim=1)
    
class End(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(End, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.layer(img)

class UNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet, self).__init__()
        self.UConv = DoubleConv(in_channel, 64)
        self.UDownConv1 = nn.Sequential(
            DownSample(),
            DoubleConv(64, 128)
        )
        self.UDownConv2 = nn.Sequential(
            DownSample(),
            DoubleConv(128, 256)
        )
        self.UDownConv3 = nn.Sequential(
            DownSample(),
            DoubleConv(256, 512)
        )
        self.UDownConv4 = nn.Sequential(
            DownSample(),
            DoubleConv(512, 1024)
        )
        self.UUpConv1 = nn.Sequential(
            UpSample(1024, 512),
            DoubleConv(1024, 512)
        )
        self.UUpConv2 = nn.Sequential(
            UpSample(512, 256),
            DoubleConv(512, 256)
        )
        self.UUpConv3 = nn.Sequential(
            UpSample(256, 128),
            DoubleConv(256, 128)
        )
        self.UUpConv4 = nn.Sequential(
            UpSample(128, 64),
            DoubleConv(128, 64)
        )
        self.End = End(64, n_classes)

    def forward(self, img):
        I1 = self.UConv(img)
        I2 = self.UDownConv1(I1)
        I3 = self.UDownConv2(I2)
        I4 = self.UDownConv3(I3)
        I5 = self.UDownConv4(I4)
        I6 = self.UUpConv1((I5, I4))
        I7 = self.UUpConv2((I6, I3))
        I8 = self.UUpConv3((I7, I2))
        I9 = self.UUpConv4((I8, I1))
        return self.End(I9)

# if __name__ == '__main__':
#     img = torch.randn(64, 3, 128, 128)
#     net = UNet(3, 21)
#     out = net(img)
#     print(out.shape)