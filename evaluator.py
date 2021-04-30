import cv2
import os
import numpy as np
import skimage.metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import random
from tesserocr import PyTessBaseAPI

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from nets.info_GAN import Discriminator, Generator

# Evaluation methods for texture similarity

def get_PSNR(target, src):
    psnr = skimage.metrics.peak_signal_noise_ratio(target, src)
    return psnr

def get_SSIM(target, src, multi_channel = True):
    ssim_ = ssim(target, src, data_range=target.max() - target.min(), multichannel=multi_channel)
    return ssim_

def get_MSE(target, src):
    mse_none = mean_squared_error(target, src)
    return mse_none

def read_img(url,url_,gray=False):
    image = cv2.imread("data/texture_data/earth_final_v4.jpg")
    image_ = cv2.imread("data/texture_data/wood_3_Texture_processed.jpg")
    if (gray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)

    return image, image_

# Evaluation method for glyph generation recognition accuracy
def glyph_OCR(theme ="./models/generator_SansSerif.pt",one_hot_ = 0):
    # if GPU available
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Define some useful variables
    # Size of z latent vector
    latent_dim = 62

    # Number of channels for generator
    channels = 1

    # Size of generated image
    img_size = 64

    # Size of z latent vector
    latent_dim = 62

    # Size of label vector
    label_dim = 26

    # Size of code vector
    code_dim = 2

    # Output directory
    opt_dir = ""

    # Pre-trained model
    dataset = theme

    z = Variable(FloatTensor(np.random.normal(0, 1,(1, latent_dim))))
    one_hot = np.zeros((1,26))
    #random_label = random.randint(0, 25)

    one_hot[0,one_hot_]=1
    label_input = Variable(FloatTensor(one_hot))
    # Code input for infoGAN (parameter of output image, shape, thickness, rotation etc.)
    first_code_input = 0
    second_code_input = 0
    code_input = Variable(FloatTensor([[first_code_input, second_code_input]]))
    noise_input = torch.cat((label_input, z),-1)

    # Load model
    model = Generator(channels,img_size,latent_dim+label_dim,code_dim)
    model.load_state_dict(torch.load(dataset))
    model.to(device)
    with torch.no_grad():
        sample1 = model(noise_input, code_input).detach().cpu()
        #save the image
        save_image(sample1.data, './img{}.jpg'.format(one_hot_), normalize=True)
    return "./img.jpg"


def OCR_recognise(filename):
    with PyTessBaseAPI(path=r'C:\Program Files\Tesseract-OCR\tessdata') as api:
        api.SetImageFile(filename)
        text = api.GetUTF8Text() # get OCR object

    return text


for i in range(26):
    glyph_OCR(one_hot_= i)


