import numpy as np
import torch, os, argparse, itertools

import torch.utils.data as data
from torchvision import datasets
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from nets.info_GAN import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=155, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code dimension")
parser.add_argument("--label_dim", type=int, default=26, help="training labels dimension")
parser.add_argument("--n_classes", type=int, default=26, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--buffer_size", type=int, default=1536, help="size of training data of each letter")
parser.add_argument("--lambda_con", type=float, default=0.1, help="lambda constant for info loss")
parser.add_argument("--lambda_gp", type=float, default=10, help="lambda constant for for gradient penalty")
parser.add_argument("--training_dir", type=str, default="./data/GAN_training_data/dafont/Sans serif", help="style font training directory")
parser.add_argument("--training_label_dir", type=str, default="./data/trainin_labels.txt", help="style font training label directory")
parser.add_argument("--output_static_dir", type=str, default="./results/GAN_opt/static/", help="GANs static output directory")
parser.add_argument("--output_v1_dir", type=str, default="./results/GAN_opt/varying_c1", help="GANs variation 1 output directory")
parser.add_argument("--output_v2_dir", type=str, default="./results/GAN_opt/varying_c2", help="GANs variation 2 output directory")
parser.add_argument("--trained_dir", type=str, default="./models/generator.pt", help="GANs generator model directory")

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
    training_data_lists = [name for name in os.listdir(training_dir)]

    img = datasets.ImageFolder(training_dir, transform=data_transform)
    img_loader = torch.utils.data.DataLoader(img, batch_size=opt.batch_size, shuffle=False)
    return img_loader

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

def one_hot_generation(target,num_classes,batch_size=1):
    """ One-hot encoding generator for Conditional GAN"""
    one_hot = np.zeros((batch_size,num_classes))
    for i in range(batch_size):
        one_hot[i,target] = 1
    return one_hot

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # if folder doen't exist
    os.makedirs(opt.output_static_dir, exist_ok=True)
    os.makedirs(opt.output_v1_dir, exist_ok=True)
    os.makedirs(opt.output_v2_dir, exist_ok=True)

    os.makedirs(opt.output_static_dir+'/'+str(batches_done), exist_ok=True)

    z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))))

    for i in range(26):

        one_hot = np.zeros((1, 26))
        one_hot[0, i] = 1
        label_input = Variable(FloatTensor(one_hot))


        code_input = Variable(FloatTensor([[0.5, -0.5]]))
        device = torch.device("cuda")
        generator.to(device)

        noise_input = torch.cat((label_input, z), -1)
        with torch.no_grad():
            output_ = generator(noise_input, code_input).detach().cpu()
            save_image(output_, opt.output_static_dir+str(batches_done)+"/img{}.png".format(i))

    """
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * 10, opt.latent_dim))))
    label_input_ = [[]]
    for i in range(opt.label_dim):
        label_input = one_hot_generation(i,opt.n_classes,1)
        label_input = np.tile(np.transpose(label_input),10)
        label_input_ = np.append(label_input,label_input_)
    label_input_ = label_input_.reshape(opt.label_dim, opt.label_dim*10)

    label_input_ = Variable(FloatTensor(np.transpose(label_input_)))
    noise_input = torch.cat((z, label_input_), -1)
    static_sample = generator(noise_input, static_code)
    save_image(static_sample.data, opt.output_static_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Get varied c1 and c2
    c_varied = np.repeat(np.linspace(-1, 1, 10)[:, np.newaxis], n_row, 0)
    c_varied2 = np.repeat(np.linspace(-1, 1, 10)[:, np.newaxis], n_row, 0)

    c1 = Variable(FloatTensor(np.hstack((c_varied,c_varied2))))
    c2 = Variable(FloatTensor(np.hstack((c_varied2,c_varied))))
    sample1 = generator(noise_input, c1)
    sample2 = generator(noise_input, c2)


    save_image(sample1.data, opt.output_v1_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, opt.output_v2_dir+"/%d.png" % batches_done, nrow=n_row, normalize=True)

"""

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
            label_input = Variable(FloatTensor(one_hot_generation(labels[0],opt.n_classes,batch_size)))
            code_input = Variable(FloatTensor(np.random.uniform(-2, 2, (batch_size, opt.code_dim))))
            noise_input = torch.cat((label_input, z), -1)
            # Generate a batch of images
            gen_imgs = generator(noise_input, code_input)

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

            # WGAN-GP gradient penalty function
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            # Wasserstein GAN loss function
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
            noise_input = torch.cat((label_input, z), -1)

            gen_imgs = generator(noise_input, code_input)
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
                "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f] [Info loss: %f]"
                % ( epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=opt.n_classes, batches_done=batches_done)
                torch.save(generator.state_dict(), opt.trained_dir)

    return 0

"""GANs training process"""
# Generator and discriminator initialization
generator = Generator(opt.channels,opt.img_size,opt.latent_dim+opt.label_dim,opt.code_dim)
discriminator = Discriminator(opt.channels,opt.img_size,opt.latent_dim+opt.label_dim,opt.code_dim)

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
static_z = Variable(FloatTensor(np.zeros((opt.n_classes * 10, opt.latent_dim))))
label_input = Variable(FloatTensor(np.random.randint(0, 26, (opt.n_classes * 10, opt.label_dim))))
static_noise_input = torch.cat((static_z, label_input), -1)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes * 10, opt.code_dim))))

# Start training
train_step(dataset(opt.training_dir))

# Save generator model
torch.save(generator.state_dict(), opt.trained_dir)
