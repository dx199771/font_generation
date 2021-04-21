# Python code to download vgg16 model and pre-trained glyph models.
# @@@@@ gdown package need to be downloaded first! @@@@@

# pip install gdown

import gdown

def vgg16_model_downloader():
    vgg16_model_name = "./vgg16.npy"
    vgg16_model_url = "1wMXkLIDOiOepUoAjbhuCx7RssCTRRUp1"
    gdown.download("https://drive.google.com/uc?id={}".format(vgg16_model_url),vgg16_model_name)

def glyph_pretrained_downloader(model="oldschool"):
    model_name = model
    model_url =