import numpy as np
import torch, os, argparse, itertools

import torch.utils.data as data
from torchvision import datasets
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from models.info_GAN import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=12, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
parser.add_argument("--buffer_size", type=int, default=1536, help="size of training data of each letter")
parser.add_argument("--lambda_con", type=float, default=0.1, help="lambda constant for info loss")
parser.add_argument("--lambda_gp", type=float, default=10, help="lambda constant for for gradient penalty")
parser.add_argument("--training_dir", type=str, default="./data/GAN_training_data", help="style font training directory")
parser.add_argument("--training_label_dir", type=str, default="./data/trainin_labels.txt", help="style font training label directory")
parser.add_argument("--output_static_dir", type=str, default="./results/GAN_opt/static/", help="GANs static output directory")
parser.add_argument("--output_v1_dir", type=str, default="./results/GAN_opt/varying_c1", help="GANs variation 1 output directory")
parser.add_argument("--output_v2_dir", type=str, default="./results/GAN_opt/varying_c2", help="GANs variation 2 output directory")
parser.add_argument("--trained_dir", type=str, default="./trained/generator.pt", help="GANs generator model directory")

opt = parser.parse_args()

def dataset(training_dir=opt.training_dir):
    """Get training data from data folder"""
    data_transform = transforms.Compose([
                                    transforms.Resize(opt.img_size),
                                    transforms.CenterCrop(opt.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Grayscale(num_output_channels=1),
                                ])
    img = datasets.ImageFolder(training_dir, transform=data_transform)
    imgLoader = torch.utils.data.DataLoader(img, batch_size=opt.batch_size, shuffle=True)

    return imgLoader

def weights_init_normal(m):
    """Weight initialization function for G and D"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates[0],
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""

    os.makedirs(opt.output_static_dir, exist_ok=True)
    os.makedirs(opt.output_v1_dir, exist_ok=True)
    os.makedirs(opt.output_v2_dir, exist_ok=True)

    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_code)
    save_image(static_sample.data, opt.output_static_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c_varied2 = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)

    c1 = Variable(FloatTensor(np.hstack((c_varied,c_varied2))))
    c2 = Variable(FloatTensor(np.hstack((c_varied2,c_varied))))

    sample1 = generator(static_z, c1)
    sample2 = generator(static_z, c2)


    save_image(sample1.data, opt.output_v1_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, opt.output_v2_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)


def train_step(dataloader):
    print(
        "--START TRAINING-- \n--TOTAL: %d EPOCHS, BATCH SIZE: %d, LEARNING RATE: %f.--"
        % (opt.n_epochs, opt.batch_size, opt.lr)
    )
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            code_input = Variable(FloatTensor(np.random.uniform(-2, 2, (batch_size, opt.code_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, code_input)

            # Loss measures generator's ability to fool the discriminator
            validity, _ = discriminator(gen_imgs)
            g_loss = -torch.mean(validity)
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, _ = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred, _ = discriminator(gen_imgs.detach())

            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + opt.lambda_gp * gradient_penalty


            # Total discriminator loss

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Information Loss
            # ------------------

            optimizer_info.zero_grad()

            # Sample noise, labels and code as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            code_input = Variable(FloatTensor(np.random.uniform(-2, 2, (batch_size, opt.code_dim))))
            gen_imgs = generator(z, code_input)
            _, pred_code = discriminator(gen_imgs)
            info_loss = opt.lambda_con * continuous_loss(
                pred_code, code_input
            )

            info_loss.backward()
            optimizer_info.step()

            # --------------
            # Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=12, batches_done=batches_done)

    return 0

"""
    GANs 
"""
# Generator and discriminator initialization
generator = Generator(opt.channels,opt.img_size,opt.latent_dim,opt.code_dim)
discriminator = Discriminator(opt.channels,opt.img_size,opt.latent_dim,opt.code_dim)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

# Cuda configuration
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

# Start training
train_step(dataset(opt.training_dir))

# Save generator model
torch.save(generator.state_dict(), opt.trained_dir)
