
import torch
import argparse
import numpy as np
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
from PIL import Image

# import some pytorch libraries
#from nets.info_GAN import Discriminator, Generator
from torch.autograd import Variable
from torchvision.utils import save_image
from nets.info_GAN import Discriminator, Generator

# if GPU available
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#@title Choose your favourite font theme.
# choose your font theme model
model_choice = "sans_serif"  # @param ["old_school", "comic", "fancy", "western(not available)", "curly", "medieval(not available)", "pixel", "sans_serif"]

switcher = {
    "old_school": "./models/old_school.pt",
    "comic": "./models/comic.pt",
    "fancy": "./models/fancy.pt",
    "western(not available)":"./models/western.pt",
    "curly":"./models/curly.pt",
    "medieval(not available)":"./models/medieval.pt",
    "pixel":"./models/pixel.pt",
    "sans_serif":"./models/generator_SansSerif.pt",
}
dataset = switcher[model_choice]

# Define some useful variables
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

#@title Try different latent code and input character.

# Input latent vector
z = Variable(FloatTensor(np.random.normal(0, 1,(1, latent_dim))))

# Input label vector
input_char = 'A'
one_hot = np.zeros((1,26))
one_hot[0,5]=1
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
  #save and show the image
  save_image(sample1.data, './img{}.png'.format(1) , normalize=True)
  img = plt.imread("img1.png")
  plt.imshow(img, cmap="gray")
  plt.show()


