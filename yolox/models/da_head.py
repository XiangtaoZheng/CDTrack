# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from yolox.models.gradient_scalar_layer import grad_reverse
from yolox.models.network_blocks import get_activation
# from gradient_scalar_layer import grad_reverse


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

        # for l in [self.conv1_da, self.conv2_da]:
        #     torch.nn.init.normal_(l.weight, std=0.001)
        #     torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1_da(x))
        x = self.conv2_da(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.act = get_activation('silu')
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.classifier(x)
        return x


class DomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self, in_channels, width):
        super(DomainAdaptationModule, self).__init__()

        in_channels = [int(channel * width) for channel in in_channels]
        self.w = nn.Parameter(torch.Tensor([1, 1, 1, 1]), requires_grad=True)
        for i in range(len(in_channels)):
            conv = Discriminator(in_channels[i])
            setattr(self, 'Discriminator_{}'.format(i), conv)

    # def forward(self, source_features, target_features):
    #     source_features = [grad_reverse(feature) for feature in source_features]
    #     target_features = [grad_reverse(feature) for feature in target_features]
    #
    #     loss_img_s, loss_img_t = 0, 0
    #     for i in range(len(source_features)):
    #         conv = getattr(self, 'Discriminator_{}'.format(i))
    #         source = conv(source_features[i])
    #         target = conv(target_features[i])
    #
    #         source_label = torch.zeros_like(source)
    #         target_label = torch.ones_like(target)
    #
    #         w = torch.exp(self.w[i]) / torch.sum(torch.exp(self.w))
    #         loss_img_s += w * F.binary_cross_entropy_with_logits(source, source_label)
    #         loss_img_t += w * F.binary_cross_entropy_with_logits(target, target_label)
    #
    #     return self.loss_weight * loss_img_s, self.loss_weight * loss_img_t

    def forward(self, features, source=True):
        features = [grad_reverse(feature) for feature in features]

        loss_img = 0
        for i in range(len(features)):
            conv = getattr(self, 'Discriminator_{}'.format(i))
            feature = conv(features[i])

            if source:
                label = torch.zeros_like(feature)
            else:
                label = torch.ones_like(feature)

            w = torch.exp(self.w[i]) / torch.sum(torch.exp(self.w))
            loss_img += w * F.binary_cross_entropy_with_logits(feature, label)

        return loss_img


if __name__ == '__main__':
    # x = torch.zeros((1, 32, 128, 128))
    # model = Discriminator(32)
    # y = model(x)
    # print(y.shape)

    in_channels = [128, 256, 512, 1024]
    width = 0.25
    model = DomainAdaptationModule(in_channels, width)
    x = [torch.ones((1, 32, 512, 512)), torch.ones((1, 64, 256, 256)),
         torch.ones((1, 128, 128, 128)), torch.ones((1, 256, 64, 64))]
    y = [torch.ones((1, 32, 512, 512)), torch.ones((1, 64, 256, 256)),
         torch.ones((1, 128, 128, 128)), torch.ones((1, 256, 64, 64))]
    loss_s, loss_t = model(x, y)
    print(loss_s, loss_t)