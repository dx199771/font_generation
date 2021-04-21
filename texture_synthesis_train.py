import numpy as np
import torch
import argparse
import utils.texture_synthesis_tools as tools
import models.vgg16_texture as net

parser = argparse.ArgumentParser()
parser.add_argument("--input_width", type=int, default=256, help="width of input image")
parser.add_argument("--input_height", type=int, default=256, help="height of input image")
parser.add_argument("--input_file", type=str, default="./data/texture_2.jpg", help="input image url")
parser.add_argument("--output_file", type=str, default="./Texture_2_processed.jpg", help="output image name")
parser.add_argument("--output_path", type=str, default="./data", help="output image url")
parser.add_argument("--epochs", type=int, default="10000", help="numbers epochs of training")
parser.add_argument("--lr", type=int, default="0.001", help="adam: learning rate")
parser.add_argument("--output_dir", type=str, default="./results/Texture_opt/", help="Output directory of texture synthesis")
parser.add_argument("--before_opt_filename", type=str, default="Texture_0_before.jpg", help="Output file name before texture synthesis")
parser.add_argument("--after_opt_filename", type=str, default="Texture_0_after", help="Output file name after texture synthesis")
opt = parser.parse_args()


def loss_func(m, texture_opt, noise_opt):
    loss_ = 0
    for i in range(len(m)):
        texture_filters = torch.squeeze(texture_opt[m[i][0]], 0)
        texture_filters_ = torch.reshape(texture_filters, shape=(
        texture_filters.shape[1] * texture_filters.shape[2], texture_filters.shape[0]))

        #Gram Matrix computation
        gram_matrix_texture = torch.matmul(texture_filters_.T, texture_filters_)
        noise_filters = torch.squeeze(noise_opt[m[i][0]], 0)
        noise_filters_= torch.reshape(noise_filters,
                                   shape=(noise_filters.shape[1] * noise_filters.shape[2], noise_filters.shape[0]))
        gram_matrix_noise = torch.matmul(torch.transpose(noise_filters_,1,0), noise_filters_)
        denominator = (4 * torch.tensor(texture_filters_.shape[1]) * torch.tensor(
            texture_filters_.shape[0]))
        loss = m[i][1] * (torch.sum(torch.square(torch.subtract(gram_matrix_texture, gram_matrix_noise))) / denominator.float())
        loss_ = loss_ + loss
    return loss_


def texture_train(input_filename, processed_path, processed_filename, m, eps, op_dir, initial_filename, final_filename):

    # Load vgg16 net
    vgg16 = net.Vgg16()
    if cuda:
        vgg16 = vgg16.cuda()

    # Read and pre-process training image
    texture_data = tools.resize_img(input_filename, opt.input_width, opt.input_height, processed_path,
                                    processed_filename)
    texture_outputs = tools.compute_layer_output(texture_data, vgg16)

    # Generate random initial noise data
    random = 0.5 * np.random.randn(texture_data.shape[0],texture_data.shape[1],texture_data.shape[2],texture_data.shape[3])
    random = Tensor(random)

    # Feed to vgg16
    vgg16.forward(random)

    # Optimizer
    optimizer = torch.optim.Adam([random.requires_grad_()], lr=opt.lr)

    for i in range(30000):
        vgg16.forward(random)
        noise_layers_list = dict(
            {0: vgg16.conv1_1, 1: vgg16.conv1_2, 2: vgg16.pool1, 3: vgg16.conv2_1, 4: vgg16.conv2_2, 5: vgg16.pool2,
             6: vgg16.conv3_1,
             7: vgg16.conv3_2,
             8: vgg16.conv3_3, 9: vgg16.pool3, 10: vgg16.conv4_1, 11: vgg16.conv4_2, 12: vgg16.conv4_3, 13: vgg16.pool4,
             14: vgg16.conv5_1, 15: vgg16.conv5_2,
             16: vgg16.conv5_3, 17: vgg16.pool5})
        loss = loss_func(m, texture_outputs, noise_layers_list)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(i,loss)
        if(i%150 == 0):
            final_noise = random
            final_filename = final_filename+str(i)+".jpg"
            final_noise_ = tools.post_process_and_display(final_noise, op_dir, final_filename,input_file = 0)

    initial_noise = tools.post_process_and_display(random, op_dir, initial_filename,save_file=True)
    return final_noise_

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

m = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),(6, 1),(7, 1),(8, 1),(9 ,1)]

texture_train(opt.input_file,opt.output_path,opt.output_file,m, opt.epochs,opt.output_dir,
              opt.before_opt_filename,opt.after_opt_filename)

