# code for testing
# source code of Playground.ipynb
# We recommend using Playground.ipynb
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
model_choice = "pixel"  # @param ["sans_serif", "comic", "old_school",  "fancy", "western(not available)", "curly(not available)", "medieval(not available)", "pixel"]

switcher = {
    "old_school": "./models/generator_OldSchool.pt",
    "comic": "./models/generator_comic.pt",
    "fancy": "./models/generator_fancy.pt",
    "western(not available)":"./models/generator_western.pt",
    "curly":"./models/generator_curly.pt",
    "medieval(not available)":"./models/generator_medieval.pt",
    "pixel":"./models/generator_bitmap.pt",
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
input_char = 'P' # @param ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "E", "S", "T", "U", "V", "W", "X", "Y", "Z"]
if model_choice == "sans_serif":
  one_hot_lookup = {
      "A":1,"B":2,"C":3,"D":4,"E":6,"F":5,"G":15,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":7,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":0
  }
if model_choice == "comic":
  one_hot_lookup = {
      "A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,"L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":0
  }
if model_choice == "old_school":
  one_hot_lookup = {
      "A":1,"B":18,"C":4,"D":16,"E":3,"F":6,"G":7,"H":8,"I":9,"J":11,"K":21,"L":12,"M":23,"N":13,"O":17,"P":5,"Q":15,"R":2,"S":19,"T":0,"U":10,"V":22,"W":23,"X":24,"Y":25,"Z":20
  }
if model_choice == "pixel":
  one_hot_lookup = {
      "A":3,"B":2,"C":5,"D":4,"E":0,"F":6,"G":7,"H":8,"I":12,"J":10,"K":13,"L":11,"M":9,"N":14,"O":16,"P":18,"Q":17,"R":15,"S":19,"T":20,"U":22,"V":23,"W":21,"X":24,"Y":25,"Z":1
  }
if model_choice == "fancy":
  one_hot_lookup = {
      "A":1,"B":2,"C":4,"D":16,"E":5,"F":6,"G":3,"H":8,"I":9,"J":11,"K":11,"L":12,"M":13,"N":14,"O":4,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,"V":22,"W":23,"X":24,"Y":25,"Z":0
  }
# create one hot encoder
one_hot = np.zeros((1,26))
one_hot[0,one_hot_lookup[input_char]]=1
label_input = Variable(FloatTensor(one_hot))


# Code input for infoGAN (parameter of output image, shape, thickness, rotation etc.)
first_code_input = -0.1 #@param {type:"slider", min:-2, max:2, step:0.1}
second_code_input = -0.1 #@param {type:"slider", min:-2, max:2, step:0.1}
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
  gly_img = plt.imread("img1.png")
  plt.imshow(gly_img, cmap="gray")
  plt.show()


#@title Choose your favourite texture.
# choose your texture image.
model_choice = "flower"  # @param ["stone", "wood", "desert", "flower", ""]

switcher = {
    "stone": "./font_generation/data/texture_data/stone_final_av2.jpg",
    "wood": "./font_generation/data/texture_data/wood3_final_av2.jpg",
    "desert": "./font_generation/data/texture_data/deset_final_av3.jpg",
    "flower": "./font_generation/data/texture_data/texture_final_v2.jpg",
    "": "./font_generation/data/texture_data/",
}
dataset = switcher[model_choice]


tex_img = plt.imread(dataset)
plt.imshow(tex_img, cmap="gray")
plt.show()

import cv2
def mixture(src, texture_, size):
    """
    Apply synthesised texture on generated glyph images
    :param src: sorce glyph image
    :param texture_: source texture image
    :return:
    """

    s_h, s_w, s_c = src.shape
    texture = cv2.resize(texture_,(s_h,s_w),cv2.INTER_AREA)
    imgray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY).astype('uint8')

    #thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    for i in range(s_h):
        for j in range(s_w):
            if (imgray[j,i] == 0):
                texture[j,i,0] = 255
                texture[j,i,1] = 255
                texture[j,i,2] = 255

    texture = cv2.resize(texture,(size,size),cv2.INTER_AREA)

    cv2.imwrite("../output.png",texture)
    plt.imshow(texture)
    plt.show()


    #cv2.imshow("contours",texture)
    #cv2.waitKey(0)
    #cv2.destroyWindow()

mixture(gly_img,tex_img,256)
