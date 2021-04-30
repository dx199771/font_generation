from PIL import Image
import numpy as np
import os
import torch
import scipy
import scipy.interpolate
import cv2
import torchvision.transforms as transforms


def resize_img(image_path, w, h, output_path_, output_filename,device):
    """
    Resize image to width x height dimensions and normalize to [0-1]
    :param image_path: image that will be resized
    :param w: resize width
    :param h: resize height
    :param output_path_: output file path
    :param output_filename: output file name
    :return: normalized processed image
    """
    # This will resize the image to width x height dimensions and then normalize it to [0-1]
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_resized = img.resize(size=(w, h))
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        # vgg16 model mean
        mean = [0.5, 0.5, 0.5]
        # data transformation
        data_transform = transforms.Compose([
            transforms.Resize((h, w)),
            # transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize((mean[0], mean[1], mean[2]), (1, 1, 1)),
        ])
        texture_data = data_transform(img)
        img_resized.save(output_path_ + output_filename)
        texture_data = texture_data.unsqueeze(0).to(device)
        print(texture_data.min(), texture_data.max(),"xx")

        return texture_data

    else:
        print("There is no image found")
        return None


def compute_layer_output(img_array,model):
    """
    Compute each layer's output
    :param img_array: image data
    :param model: VGG model
    :return: each layer's output in a array
    """
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    vgg = model
    #img_array = Tensor(img_array)
    vgg.forward(img_array)

    content_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, 6: vgg.conv3_1,
                                7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4,
                                14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5 })

    outputs = dict()
    for i in range(len(content_layers_list)):
        outputs[i] = content_layers_list[i]
    print("All layers' outputs have been computed successfully.")
    return outputs


def post_process_and_display(cnn_output, output_path, output_filename,input_file, save_file=True, histogram_matched=True):
    """
    This function take input noise (1, channels, w, h) shapped array
    and ultimately output the synthesised texture image.
    :param cnn_output:
    :param output_path:
    :param output_filename:
    :param input_file:
    :param save_file:
    :param histogram_matched:
    :return:
    """
    x = cnn_output.cpu().detach().numpy()
    x_ = np.squeeze(x)

    std = [1,1,1]
    mean = [0.5, 0.5, 0.5]

    # denormlization
    x_[0, :, :] = x_[0, :, :] * std[0] + mean[0]
    x_[1, :, :] = x_[1, :, :] * std[1] + mean[1]
    x_[2, :, :] = x_[2, :, :] * std[2] + mean[2]

    x_ *= 255
    x_ = np.clip(x_, 0, 255)
    x_ = np.transpose(x_,(1,2,0))

    cv2.imwrite(output_path+"/matched.jpg",x_)

    img = Image.fromarray(x_.astype('uint8'), mode='RGB')
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + "s"+output_filename)
    """
    if histogram_matched:
        img_ = cv2.imread(output_path+ "s" +output_filename)
        matched_img = histogram_matching(new_texture,input_file)
        cv2.imwrite(output_path + output_filename, matched_img)
        matched_img.save(output_path + output_filename)
    """
    return output_path + "s"+output_filename

def histogram_matching(org_image, match_image, n_bins=100):
    '''
    Matches histogram of each color channel of org_image with histogram of match_image
    :param org_image: image whose distribution should be remapped
    :param match_image: image whose distribution should be matched
    :param grey: True if images are greyscale
    :param n_bins: number of bins used for histogram calculation
    :return: org_image with same histogram as match_image
    '''
    matched_image = np.zeros_like(org_image)
    for i in range(3):
        hist, bin_edges = np.histogram(match_image[:, :, i].ravel(), bins=n_bins, density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist * np.diff(bin_edges))
        inv_cdf = scipy.interpolate.interp1d(cum_values, bin_edges, bounds_error=True)
        r = np.asarray(uniform_hist(org_image[:, :, i].ravel()))
        r[r > cum_values.max()] = cum_values.max()
        matched_image[:, :, i] = inv_cdf(r).reshape(org_image[:, :, i].shape)

    return matched_image

def uniform_hist(X):
    '''
    Maps data distribution onto uniform histogram
    :param X: data vector
    :return: data vector with uniform histogram
    '''

    Z = [(x, i) for i, x in enumerate(X)]
    Z.sort()
    n = len(Z)
    Rx = [0] * n
    start = 0  # starting mark
    for i in range(1, n):
        if Z[i][0] != Z[i - 1][0]:
            for j in range(start, i):
                Rx[Z[j][1]] = float(start + 1 + i) / 2.0;
            start = i
    for j in range(start, n):
        Rx[Z[j][1]] = float(start + 1 + n) / 2.0;
    return np.asarray(Rx) / float(len(Rx))

def match_his_output(match,src,opt_dir):
    """
    Output matched image
    :param match: image whose distribution should be matched
    :param src: image whose distribution should be remapped
    :param opt_dir: output directory
    :return: matched image data
    """
    img_src = cv2.imread(src)
    img_mat = cv2.imread(match)
    mathced = histogram_matching(img_src,img_mat)
    opt_dir = opt_dir+"/matched.jpg"
    cv2.imwrite(opt_dir,mathced)

    return mathced

