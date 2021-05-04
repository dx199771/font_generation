Welcome!

In this project, we proposed a interpretable font generation system with synthesised texture based on Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs).

The system includes two main networks: A glyph generation network, A texture synthesis network and some useful tools (font parser, texture applier etc.).

IMPORTANT: 
"""GPU is required for both testing and training !!!"""
"""Go to this website: https://colab.research.google.com/drive/1i-sxC_zZ6qgz9ovRLgx08ogdaQG4EXKE?usp=sharing"""
"""OR run playground.ipynb to test and train the system, there is further instructions !!!"""

REQUIRED:
training and testing:
pip install pytorch

parser:
pip install beautifulsoup4

You can view the source files from GitHub repository
GitHub Repository: https://github.com/dx199771/font_generation

There are some pre-trained glyph generation models and pre-synthesised texture images provided.
Pre-trained model can be found in ./models folder, pre-synthesised texture images can be found in ./data/texture_data folder.

""" If you want to train your own glyph model """
Notebook provide a downloader, you can download glyph training data in the notebook.
Or you can manually download font dataset from google drive: https://drive.google.com/drive/folders/1kmd-IQOUktefLss5WiF7ZASKrJxlzSNv?usp=sharing
download them and drag them into ./data/GAN_training_data/   folder.

if you want to train you texture, you need to download the pre-trained vgg16.npy file from google drive: https://drive.google.com/file/d/1wMXkLIDOiOepUoAjbhuCx7RssCTRRUp1/view?usp=sharing
download them and drag it into ./nets, or automatically download them in the notebook.

If you have any questions about operating the system, please contact my email address: xudong9771@gmail.com

Many Thanks,
Xu Dong