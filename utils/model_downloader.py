# Python code to download vgg16 model and pre-trained glyph models.
# @@@@@ gdown package need to be downloaded first! @@@@@

# pip install gdown

import gdown

def vgg16_model_downloader():
    """
    vgg16 model downloader for texture synthesis network
    :return:
    """
    # download vgg16 model for texture synthesis from google drive
    # model name
    vgg16_model_name = "./vgg16.npy"
    # model url
    vgg16_model_url = "1wMXkLIDOiOepUoAjbhuCx7RssCTRRUp1"
    gdown.download("https://drive.google.com/uc?id={}".format(vgg16_model_url),vgg16_model_name)

def glyph_pretrained_downloader(model="oldschool"):
    model_name = model
    model_url = model