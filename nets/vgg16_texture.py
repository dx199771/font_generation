import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import inspect

VGG_MEAN = [ 0.40760392,  0.45795686,  0.48501961]
cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

class Vgg16(nn.Module):
    def __init__(self, vgg16_npy_path=None):
        super(Vgg16, self).__init__()
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
        print("npy file loaded")


    def forward(self, rgb):
        """
        :param rgb:
        :return:
        """
        start_time = time.time()
        print("build model...")
        self.rgb_scaled = rgb
        # convert
        r, g, b = torch.split(self.rgb_scaled, 1, 3)

        bgr = torch.cat(([
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2],
        ]), 3)
        bgr = torch.cuda.FloatTensor(bgr).permute(0, 3, 1, 2)
        self.bgr = bgr
        self.conv1_1 = self.conv_layer(self.bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.avg_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.avg_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.avg_pool(self.conv5_3, 'pool5')

        print(("build model finished: %ds" % (time.time() - start_time)))

        return self.pool5

    def conv_layer(self, in_, name):
        filt = self.get_conv_filter(name)
        bias = self.get_bias(name)
        conv = F.conv2d(in_,weight=filt,bias=bias,stride=1,padding=1)
        relu = F.relu(conv)
        return relu

    def avg_pool(self,in_,name):
        avg = F.avg_pool2d(input=in_,kernel_size=(2, 2),stride=(2,2))
        return avg

    def get_conv_filter(self, name):

        self.weight = torch.tensor(self.data_dict[name][0],requires_grad=True).permute(3,2,0,1).cuda()
        return self.weight

    def get_bias(self, name):
        self.bias = torch.tensor(self.data_dict[name][1],requires_grad=True).cuda()

        return self.bias
