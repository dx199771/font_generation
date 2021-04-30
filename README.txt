Welcome!

In this project, we proposed a interpretable font generation system with synthesised texture based on Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs).

The system includes two main networks: A glyph generation network, A texture synthesis network and some useful tools (font parser, texture applier etc.).

IMPORTANT: 
"""GPU is required for both testing and training !!!"""
"""Run the playground.ipynb, there is further instructions !!!"""

You can view the source files from GitHub repository
GitHub Repository: https://github.com/dx199771/Interpretable-font-generation

or you can play around the pre-trained model from Google Colab:
Google Colab: https://colab.research.google.com/drive/1L1350DiqWSHR9_GZ24cRQWfTD1L3nV4A?usp=sharing

There are some pre-trained glyph generation models and pre-synthesised texture images provided.
Pre-trained model can be found in ./models folder, pre-synthesised texture images can be found in ./data/texture_data folder.

""" If you want to train your own glyph model """
You need to download font dataset from google drive: https://drive.google.com/drive/folders/1kmd-IQOUktefLss5WiF7ZASKrJxlzSNv?usp=sharing
download them and drag them into ./data/GAN_training_data/ folder

if you want to train you texture, you need to download the vgg16.npy file from google drive: https://drive.google.com/file/d/1wMXkLIDOiOepUoAjbhuCx7RssCTRRUp1/view?usp=sharing
download them and drag it into ./nets



If you have any questions about operating the system, please contact my email address: xudong9771@gmail.com

Many Thanks,
Xu Dong