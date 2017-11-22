import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch import optim
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from util import VideoTripletDataset

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True)

class TCNModel(nn.Module):
    def __init__(self, inception):
        super().__init__()
        self.transform_input = inception.transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Conv2d_6a_3x3 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(31 * 31 * 20, 1000)
        self.FullyConnected7b = Dense(1000, 128)

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 33 x 33 x 100
        x = self.Conv2d_6a_3x3(x)
        # 31 x 31 x 20
        x = self.Conv2d_6b_3x3(x)
        # 31 x 31 x 20
        x = self.SpatialSoftmax(x)
        # 1000
        x = self.FullyConnected7a(x.view(x.size()[0], -1))
        # 128
        x = self.FullyConnected7b(x)

        return x

def distance(x1, x2):
    assert(x1.size() == x2.size())
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def define_model(use_cuda):
    tcn = TCNModel(models.inception_v3(pretrained=True))
    if use_cuda:
        tcn.cuda()
    return tcn

def main():
    use_cuda = torch.cuda.is_available()

    tcn = define_model(use_cuda)

    dataset = VideoTripletDataset('./data/train/')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False
    )

    margin = 0.5

    optimizer = optim.SGD(tcn.parameters(), lr=1e-3, momentum=0.9)

    for minibatch in data_loader:
        frames = Variable(minibatch)

        if use_cuda:
            frames = frames.cuda()

        anchor_frames = frames[:, 0, :, :, :]
        positive_frames = frames[:, 1, :, :, :]
        negative_frames = frames[:, 2, :, :, :]

        positive_output = tcn(positive_frames)
        anchor_output = tcn(anchor_frames)
        negative_output = tcn(negative_frames)

        loss = distance(anchor_output, positive_output) - (
            distance(anchor_output, negative_output)
        ) + margin

        loss = torch.sum(loss)

        loss.backward()
        optimizer.step()
        print('loss: ', loss.data[0])


if __name__ == '__main__':
    main()