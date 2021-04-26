import torch
import argparse
from nets.info_GAN import Discriminator, Generator
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--pre_trained", type=str, default="./models/generator_SansSerif.pt", help="pre-trained model")
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
j = -2
for i in range(21):
    # one-hot encoder
    one_hot = np.zeros((1, 26))
    one_hot[0,1]=1

    z = Variable(FloatTensor(np.random.normal(0, 1, (1 * 20, opt.latent_dim))))
    print(z.shape)
    label_input_ = Variable(FloatTensor(np.tile(one_hot,(20,1))))
    noise_input = torch.cat((label_input_,z), -1)
    print(noise_input.shape)

    # latent code generator
    c_varied = np.repeat(np.linspace(-2, 2, 20)[:, np.newaxis], 1, 0)
    c_varied2 = np.repeat(j,20).reshape(20,1)
    j+=0.2
    print(c_varied.shape)

    print(c_varied2)

    c1 = Variable(FloatTensor(np.hstack((c_varied2,c_varied))))
    print(c1.shape)
    device = torch.device("cuda")
    model.to(device)
    with torch.no_grad():

        sample1 = model(noise_input, c1).detach().cpu()

        print(sample1.shape)
        img = save_image(sample1.data, 'cache/img{}.png'.format(i) , nrow=20, normalize=True)

"""
with torch.no_grad():
    output_ = model(noise_input, sample1).detach().cpu()
    save_image(output_, 'cache/img{}.png'.format(1))

""""""
for i in range(10):
    one_hot = np.zeros((1, 26))
    one_hot[0,i]=1

    label_input = Variable(FloatTensor(one_hot))
    l1=-2
    l2=-0.4
    for j in range(20):
        l1+=0.1
        #2-=0.1
        #c1 = Variable(FloatTensor(np.hstack((c_varied, c_varied2))))

        code_input = Variable(FloatTensor([[l2,l1]]))

        device = torch.device("cuda")
        model.to(device)

        noise_input = torch.cat((label_input,z),-1)

        with torch.no_grad():
            output_ = model(noise_input, code_input).detach().cpu()
            save_image(output_, 'cache/img{},{}.png'.format(i,j))

"""