import torch
import argparse
from models.info_GAN import Discriminator, Generator
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--pre_trained", type=str, default="./trained/generator.pt", help="pre-trained model")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code dimension")
parser.add_argument("--label_dim", type=int, default=26, help="training labels dimension")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=26, help="number of classes for dataset")

opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model = Generator(opt.channels,opt.img_size,opt.latent_dim+opt.label_dim,opt.code_dim)
model.load_state_dict(torch.load(opt.pre_trained))
model.eval()

def one_hot_generation(target,num_classes,batch_size=1):
    """ One-hot encoding generator for Conditional GAN"""
    one_hot = np.zeros((batch_size,num_classes))
    for i in range(len(target)):
        one_hot[i,target[i]] = 1
    return one_hot


z = Variable(FloatTensor(np.random.normal(0, 1,(1, opt.latent_dim))))

for i in range(10):
    one_hot = np.zeros((1, 26))
    one_hot[0,i]=1

    label_input = Variable(FloatTensor(one_hot))
    l1=-2
    l2=2
    for j in range(20):
        l1+=0.2
        l2-=0.2
        #c1 = Variable(FloatTensor(np.hstack((c_varied, c_varied2))))

        code_input = Variable(FloatTensor([[l1,l2]]))

        print(code_input)
        device = torch.device("cuda")
        model.to(device)

        noise_input = torch.cat((z, label_input),-1)
        print(noise_input.shape)
        with torch.no_grad():
            output_ = model(noise_input, code_input).detach().cpu()
            save_image(output_, 'img{}{}.png'.format(i,l1))

