18737 Foundation of Privacy Project -- Noise Robustness of Deep Neural Networks

By Qingyu Zhu, Chiaai Lin and Yuxin Jiang

FA19 Carnegie Mellon University

## Summary
In this project, we use a public image dataset to train a DNN, compare the performance of self-trained and pre-trained DNN and test the robustness of models against image changes of blur, contrast and brightness.

We aim to identify common adversarial examples targeting image recognition DNNs, understand why such attack works and propose suggestions for improving the security level of DNNs.

## Repo Contents
The following are the contents of this github repository.

### Model
Our self-trained DNN network model trained by CIFAR-10 dataset.

### Self-trained DNN network Model
classify.py -- Train the 4-layer DNN network model with Pytorch, process image changes with openCV.
data_processing.py -- Process CIFAR-10 dataset images.

### Pre-trained DNN network Model
pretrain.py -- The pre-trained model provided by Gluon.[6]

### doc
1. project prosoal 
2. project report
3. presentation

### Dependencies
1. fire
2. torch
3. torchvision
4. PIL
5. numpy
6. openCV
7. matplotlib
8. mxnet
9. gluoncv

### Running Example
1. running self-trained model
'''
python3 classify.py test xxx.jpg 
'''
2. running pre-trained model
'''
python3 pretrain.py --model cifar_resnet110_v1 --input-pic xxx.jpg
'''

## Testing Results
https://docs.google.com/spreadsheets/d/1MKY9RiGvFBqiAYkXDmvK2R-DT_n-Xkut7Zy9N3oKhH0/edit#gid=0
(Please use west.cmu.edu email to view)

## Brief Report
https://docs.google.com/document/d/1EYGEnkVp9d0hJeXhNyoXfPwccQA8VdWh-ea8QcMgZgs/edit

## Reference
[1] Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

[2] Gilmer & Hendrycks, "A Discussion of 'Adversarial Examples Are Not Bugs, They Are Features':
Adversarial Example Researchers Need to Expand What is Meant by 'Robustness'", Distill, 2019.

[3] Gong, Yuan & Poellabauer, Christian. (2018). Protecting Voice Controlled Systems Using Sound
Source Identification Based on Acoustic Cues.

[4] Junko Yoshida. (2019). “AI Tradeoff: Accuracy or Robustness?” [Web]
https://www.eetimes.com/ai-tradeoff-accuracy-or-robustness/

[5] DNN Adversaria: https://github.com/andac-demir/DNNAdversarial/blob/master/CNNclassify.py

[6] https://gluon-cv.mxnet.io/build/examples_classification/demo_cifar10.html


