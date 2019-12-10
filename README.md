# Noise Robustness of Deep Neural Networks

## Dependencies
1. fire
2. torch
3. torchvision
4. PIL
5. numpy
6. openCV
7. matplotlib
8. mxnet
9. gluoncv

## Running Example
1. running self-trained model
```
python3 classify.py test xxx.jpg 
```
2. running pre-trained model
```
python3 pretrain.py --model cifar_resnet110_v1 --input-pic xxx.jpg
```

## Testing Results
https://docs.google.com/spreadsheets/d/1MKY9RiGvFBqiAYkXDmvK2R-DT_n-Xkut7Zy9N3oKhH0/edit#gid=0
(Please use west.cmu.edu email to view)

## Brief Report
https://docs.google.com/document/d/1EYGEnkVp9d0hJeXhNyoXfPwccQA8VdWh-ea8QcMgZgs/edit

##Reference
● Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
● Gilmer & Hendrycks, "A Discussion of 'Adversarial Examples Are Not Bugs, They Are Features':
Adversarial Example Researchers Need to Expand What is Meant by 'Robustness'", Distill, 2019.
● Gong, Yuan & Poellabauer, Christian. (2018). Protecting Voice Controlled Systems Using Sound
Source Identification Based on Acoustic Cues.
● Junko Yoshida. (2019). “AI Tradeoff: Accuracy or Robustness?” [Web]https://www.eetimes.com/ai-tradeoff-accuracy-or-robustness/
● DNN Adversarial: https://github.com/andac-demir/DNNAdversarial
