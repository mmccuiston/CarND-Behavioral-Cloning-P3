# **Behavioral Cloning** 

* Summarize the results with a written report

[//]: # (Image References)

[recover-dirt]: ./examples/dirt-recover.jpg "Dirt Recovery"
[recover-turn]: ./examples/turn-recover.jpg "Turn Recovery"
[recover-line]: ./examples/line-recover.jpg "Line Recovery"
[recover-bridge]: ./examples/bridge-recover.jpg "Bridge Recovery"
[center]: ./examples/center.jpg "Center Driving"
[normal]: ./examples/normal.jpg "Normal"
[flipped]: ./examples/flipped.jpg "Flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model starts with a normalizion layer and a cropping layer to remove parts of the image above the horizon and below the hood of the vehicle.  Following this the network has 5 convolutional layers.  The first three use a 5x5 kernel and a 2 by 2 stride.  They have 24, 36, and 48 feature maps respectively.  They are also all followed by a RELU activation layer.  There is then two more convolution layers that use a 3x3 kernel, 64 feature maps, and a 1 by 1 stride.  Both of these layers are also followed by RELU activation layers.  The network is then flattened and followed by three fully connected layers of size 100, 50, and 1.


#### 2. Attempts to reduce overfitting in the model

In order to reduce the likelihood of overfitting I did the following.
1. Trained the network on a variety of datasets including different driving styles (recovery, center driving, and corners)
2. I collected stopped training once the validation error began to increase/oscillate.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right side of the road, and driving smoothly around corners.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a Lenet architecture and see how the model performed.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropout layers after the pooling layers.  This helped with the overfitting but I still had some trouble keeping the vehicle on the road in autonomous mode.

Then I decided to try a more advanced network architecture based on the one created by the NVIDIA automonomous vehicle team.  This architecture used 5 convolution layers ranging in depth from 24-64 and with kernel sizes ranging from 3-5.  All of the convolution layers are followed by a RELU layer.  The convolution layers are then followed by 4 fully connected layers of size 100, 50, 10, and 1.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.  It usually got stuck at the bridge with the black guard rail and sometimes drove off the road when turning around a corner with a dirt shoulder.  To combat this I took extra care to collect more training samples in these areas where the vehicle performed recovey maneuvers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 56-73) consisted of a convolution neural network with the following layers and layer sizes ...

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps around the track while driving in the center of the lane.

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it were to steer off the road.  I would stop the recording when going off-road and start the recording when steering back onto the road.  I spent extra time to record more examples in troublesome spots including the bridge and where there were dirt shoulders.  Here are some examples.

![alt text][recover-dirt]
![alt text][recover-line]
![alt text][recover-bridge]
![alt text][recover-turn]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would cause the model to generalize better to steer both left and right.

![alt text][normal]
![alt text][flipped]


After the collection process and augmentation, I had collected 36406 data points. I then preprocessed this data by normalizing it.  I then cropped the image to exclude the hood of the vehicle and anything above the horizon.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the monotonically decreasing validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
