**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/center.jpg "center lane "
[image3]: ./examples/recovery1.jpg "Recovery Image"
[image4]: ./examples/recovery2.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./examples/MSE_4.png "High overfitting"
[image9]: ./examples/MSE_5.png "low overfitting"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modified version of Nvidia model used for end to end deep learning. The model is defined (model.py lines 108-179)

This is the original model of Nvidia 
![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

I have added to the  Nvidia model dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14-35). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

To enhance the generalization of the mode I have made the following processing on the training data:

1.   flipping the training images and also their corresponding labels by multiplying the steering angle by -1 
2.   cropping the unconcerned parts of the images by defining the region of interest for my model 
3.   Also I have used the images from the side mirror cameras and added a defined correction value to the measured steering angle according to the camera position 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 181).

The number of epochs have been tuned during the experiments many time to prevent overfitting and the final chosen number was 10 epochs

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, smooth drives around curves and one lap clock-wise

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to be able to handle all the common and critical cases happens during the autonomous drive mode.

My first step was to use a convolution neural network model similar to the one used by Nvidia as it has been used for a similar operations which is to drive autonomously in any track so I chose it to be my start point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

![alt text][image8]

To combat the overfitting, I modified the model so that I added dropout layers.

Then I've preprocessed/Augment the images to extend the training set and also I have used the data generator concept in feeding the data to my network to reduce the model ram consumption.

![alt text][image9]

The final step was to run the simulator to see how well the car was driving around track one. At the beginning the vehicle drove well but was always tending to drive straight and in curves(specially sharp curves). After data preprocessing and data augmentation (which shall be mentioned afterwards) in addition to the regularization all these issues have been fixed and the car had an elegant performance which is not just memorizing the training data but it could also predict smartly the best attitude in critical cases.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160,320,3 image   				            |
| normalization layer   | Lambda and Cropping2D to set the RIO 	        | 
| Convolution 5x5    	| 2x2 stride, valid padding, 24 filter 	        |
| RELU                  |                                               |
| dropout				| 										        |
| Convolution 5x5    	| 2x2 stride, valid padding, 36 filter       	|
| RELU                  |                                               |
| dropout				| 										        |
| Convolution 5x5 	    | 2x2 stride, valid padding, 48 filter       	|
| RELU	                |            									|
| dropout				|            									|
| Convolution 3x3 	    | 1x1 stride, valid padding, 64 filter       	|
| RELU	                |            									|
| dropout				|            									|
| Convolution 3x3 	    | 1x1 stride, valid padding, 64 filter       	|
| RELU	                |            									|
| dropout				|            									|
| Flatten				| Flatten the output of conv5, output 1x1164	|
| fully conneted layer 	| input 1x1164, output 1x100					|
| dropout				|    											|
| fully conneted layer 	| input 1x100, output 1x50						|
| dropout				|    											|
| fully conneted layer 	| input 1x50, output 1x10						|
| dropout				|    											|
| fully conneted layer 	| input 1x10, output 1						    |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to recover back to the lane center in case of the car went to one of the road edges. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]


Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would improve the generalization as the car always pull to the left and to eliminate this bias I had to flip images to teach the model to turn right also if needed.

After the collection process, I had 81030 number of data points. I then preprocessed this data by normalizing the images with Lambda layer and also cropping the unconcerned parts of the images like the scenery parts "trees and hills"


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10.


