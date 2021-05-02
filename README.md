# style-glyph-generator
In this project, We propose a font generator based on Generative Adversarial Networks (GANs) and texture synthesis with Convolutional Neural Networks.

## Installation
GPU is required for both testing and training.
```bash
pip install pytorch
```

Some generated font images are shown below:![resultsfinal](https://user-images.githubusercontent.com/33721483/116821929-07c77400-ab74-11eb-9bbb-a4af8e80add8.png)

The system contains two sub-nets: A glyph generation netowrk to generate glyph image and a texture synthesis network to synthesis texture and apply them on glyph images.
![infogan](https://user-images.githubusercontent.com/33721483/116822588-82de5980-ab77-11eb-9473-3cf392070b9b.png)
![infogan2](https://user-images.githubusercontent.com/33721483/116822592-8671e080-ab77-11eb-8a22-2e554c67f758.png)

## Dataset
We also provide a font image dataset with 7 different font theme. You can download from [google drive](https://drive.google.com/drive/folders/1kmd-IQOUktefLss5WiF7ZASKrJxlzSNv?usp=sharing)
![font dataset](https://user-images.githubusercontent.com/33721483/116822879-3e53bd80-ab79-11eb-98a9-33222ae03a8a.jpg)


## Getting Started

To run the code, you can either run the .ipynb notebook or run the main.py.

## If you want to train your own glyph model
Notebook provide a downloader, you can download glyph training data in the notebook.
Or you can manually download font dataset from google drive: [link](https://drive.google.com/drive/folders/1kmd-IQOUktefLss5WiF7ZASKrJxlzSNv?usp=sharing)
download them and drag them into ./data/GAN_training_data/   folder.


if you want to train you texture, you need to download the pre-trained vgg16.npy file from google drive: [link](https://drive.google.com/file/d/1wMXkLIDOiOepUoAjbhuCx7RssCTRRUp1/view?usp=sharing)
download them and drag it into ./nets, or automatically download them in the notebook.

If you have any questions about operating the system, please contact my email address: xudong9771@gmail.com
