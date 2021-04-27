import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_img", type=str, default="../results/GAN_opt/static/116500/img24.png", help="input ground truth image url")
parser.add_argument("--texture_img", type=str, default="../data/texture_data/desert_final_av4.jpg", help="input ground truth image url")
parser.add_argument("--opt_size", type=int, default="256", help="output generated font image size")
opt = parser.parse_args()

def mixture(src, texture_, size):
    """
    Apply synthesised texture on generated glyph images
    :param src: sorce glyph image
    :param texture_: source texture image
    :return:
    """

    s_h, s_w, s_c = src.shape
    texture = cv2.resize(texture_,(s_h,s_w),cv2.INTER_AREA)
    imgray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(imgray,127,255,0)

    for i in range(s_h):
        for j in range(s_w):
            if (thresh[j,i] == 0):
                texture[j,i,0] = 255
                texture[j,i,1] = 255
                texture[j,i,2] = 255

    texture = cv2.resize(texture,(size,size),cv2.INTER_AREA)

    cv2.imwrite("../output.png",texture)

    cv2.imshow("contours",texture)
    cv2.waitKey(0)
    cv2.destroyWindow()

#mixture(opt.src_img,opt.texture_img, opt.opt_size)
