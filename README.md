# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

# What I did
This model is used for driving behavior cloning. Images and steering angles are recorded using simulator. These images form the training set. Then, train a convolutional neural network and predict the steering angles using it while self-driving in the simulator. The training dataset is obtained according to the following procedure:

(1) I first tried my best to drive in the middle of the lane for three loops;

(2) Then, I did the recovery trick for some difficult turns several times;

(3) I put all of the center images I recorded to dataset. Then, a train-validation-test split was done, the ratio of the training data, validation data, and test data are 60%, 20% and 20% respectively;

The neural network architechture is obtained as follows:

(1) I set LeNet as the starting point. Since LeNet is using 32323 pictures, I resize each image to 32643. Then, I make all of the data zero-mean and normalized;

(2) I choose Adam as optimizer since it can consider the learning rate decay automatically; The batch size is set to 128 which is appropriate for my GPU memory. The number of epochs is set to 20 according to the results of ModelCheckpoint. The val_loss barely improve after 20 epochs. Also, this is a regression problem, so I set the loss function to MSE.

(3) Then, I start to try different architechture and parameters. I have tried changed the following: (a) Active function (b) The number of convolutional layers: 2, 3, 4 (c) The depth and filter size of convolutional layers; (d) The number of fully-connected layers and neurons in the fully-connected layers. (e) Drop-out I check the test accuracy, and also run the model using simulator. At last, I choose the model I use. Please provide a detailed description of the model architecture (number of layers, type of layers, orders of the layers) and the parameter settings for each layer ( dimensions of the layer, activation type). The description should be detailed enough to allow your readers to reconstruct a similar network. Optional, please consider to include a simple sketch that will demonstrate the general pattern with a detailed parameter settings of the architecture.

The model architecture is described as follows:

(1)The number of layers are 11 layers;The order of the layers are convolutional layer, max-pooling layer, convolutional layer, max-pooling layer, convolutional layer, max-pooling layer, drop-out layer, and at last four fully-connected layers.

(2) All activation functions are relu;

(3) The depth of three convolutional layers are 32, 64 and 96;

(4) The number of neurons for fully-connected layers are 120, 80, 43 and 1;

(5) After chose the model, I am using all the data to train the neural network again(including validation data and test data). The dimensions for each layer are as follows: conv1: input is 32643, output is 296132; pool1: 153032 conv2: 122764 pool2: 61364 conv3: 31096 pool3: 1596 fc1: 120 fc2: 80 fc3: 43 fc4: 1

(6) The filter size for the convolutional layers are 44. The filter size for pooling layers is 22;

(7) The keep-prob is 0.5 for drop-out layer.
