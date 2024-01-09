# DeepFaceDetect: Decoding Reality
This project endeavours to distinguish authentic images (real) from those generated using deepfake technology, utilizing convolutional neural networks (CNN) in deep learning.

## Project Approach

In this project, our strategy involves utilizing a diverse set of Convolutional Neural Network (CNN) models. The main goal is to develop a model proficient in discerning between genuine images (real) and those generated through deepfake technology, with a specific emphasis on Nvidia's StyleGAN (https://github.com/NVlabs/stylegan).

The dataset utilized in this project can be accessed here: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data.

### Exploratory Data Analysis

- check for class imbalance
    - looked at each set of images for class imbalance, found that each set was labeled with binary classes(real, fake) at a 50/50 balance
- visually check images for anomalies, differences between the classes
    - visually checked a sample of real and fake images from each set of images(training, validation, testing)
    - there was nothing 
- check image sizes
    - checked the sizes of images within all the different sets of images
    - found that all the images were the same size
- check for rgb color
    - checked all images for rgb color
    - found that all images are rgb
- created color histograms for sample images
    - created color histograms for a sample of images
    - realized that I do not possess the knowledge to find anything of value from them
  
### Image Processing

- data
  - use imagegenerator to process images
  - scale image data between 0 and 1 for modelling purposes
  - use the horizontal flip parameter for image augmentation to help increase the models ability to generalize


### Model Creation

- Model Selection
  - VGG16
  - Xception
  - ResNet50
  - Ensemble Method


### Model Deployment/App Creation


## Results/Findings


## Future Plans

