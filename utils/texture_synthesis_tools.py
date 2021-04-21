from PIL import Image
import numpy as np
import os
import torch
import scipy
import scipy.interpolate
import cv2


def resize_img(image_path, w, h, output_path_, output_filename):
    # This will resize the image to width x height dimensions and then normalize it to [0-1]
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_resized = img.resize(size=(w, h))
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        img_resized.save(output_path_ + output_filename)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255
        img_array_ = np.squeeze(img_array)
        img_array_ = img_array_ * 255
        img = Image.fromarray(img_array_.astype('uint8'), mode='RGB')
        img.save("test.jpg")

        return img_array

    else:
        print("There is no image found")


def compute_layer_output(img_array,model):
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    vgg = model
    img_array = Tensor(img_array)
    vgg.forward(img_array)

    content_layers_list = dict({0: vgg.conv1_1, 1: vgg.conv1_2, 2: vgg.pool1, 3: vgg.conv2_1, 4: vgg.conv2_2, 5: vgg.pool2, 6: vgg.conv3_1,
                                7: vgg.conv3_2, 8: vgg.conv3_3, 9: vgg.pool3, 10: vgg.conv4_1, 11: vgg.conv4_2, 12: vgg.conv4_3, 13: vgg.pool4,
                                14: vgg.conv5_1, 15: vgg.conv5_2, 16: vgg.conv5_3, 17: vgg.pool5 })

    outputs = dict()
    for i in range(len(content_layers_list)):
        outputs[i] = content_layers_list[i]
    print("All layers' outputs have been computed successfully.")
    return outputs


def post_process_and_display(cnn_output, output_path, output_filename,input_file, save_file=True):
    # This will take input_noise of (1, w, h, channels) shapped array taken from tensorflow operation
    # and ultimately displays the image

    x = cnn_output.cpu().detach().numpy()
    x = np.squeeze(x)

    x = (x - x.min()) / (x.max() - x.min())
    for i in range(255):
        print(x[i,i,0],x[i,i,1],x[i,i,2])
    x *= 255
    x = np.clip(x, 0, 255)
    img = Image.fromarray(x.astype('uint8'), mode='RGB')
    #img.show()
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + "s"+output_filename)
        #img_ = cv2.imread(output_path+ "s" +output_filename)


        #matched_img = histogram_matching(new_texture,input_file)
        #cv2.imwrite(output_path + output_filename, matched_img)

        #matched_img.save(output_path + output_filename)

    return x

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



src = r"C:\Users\User\PycharmProjects\pythonProject\CCFontGANs\image_resources\outputs\sTexture_9_C3_final22200.jpg"
match = r"C:\Users\User\PycharmProjects\pythonProject\CCFontGANs\data\Texture_8.jpg"
img = cv2.imread(src)
img_ = cv2.imread(match)

print(img_)
cv2.mean(img)
a = histogram_matching(img,img_)

for i in range(255):
    print(a[i,i,0],a[i,i,1],a[i,i,2])
a = histogram_matching(img,img_)
print(cv2.mean(a))
cv2.imwrite("test.jpg",a)
img = Image.fromarray(a.astype('uint8'), mode='RGB')
img.save("test2.jpg")
