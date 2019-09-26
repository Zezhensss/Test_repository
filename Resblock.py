import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding = 1, bn=True):
        super(ResnetBlock, self).__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.up1 = nn.AvgPool2d((2,2),2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if in_dim == out_dim:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        if self.bn == True:
            out = F.relu(self.bn1(self.conv1(x)),inplace=True)
            out = self.bn2(self.conv2(out))
            res = x if self.shortcut is None else self.bn3(self.shortcut(x))
        if self.bn == False:
            out = F.relu(self.conv1(x),inplace=True)
            out = F.relu(self.conv2(out),inplace=True)
            res = x if self.shortcut is None else self.shortcut(x)



        return self.up1(F.relu(res+out,inplace=True))


class ResnetBlockTranspose(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding = 1):
        super(ResnetBlockTranspose, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if in_dim == out_dim:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        x = F.interpolate(x,scale_factor=2)
        out = F.relu(self.bn1(self.conv1(x)),inplace=True)
        out = self.bn2(self.conv2(out))
        res = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(res+out,inplace=True)

        return out
