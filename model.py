import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense,Flatten,Lambda,Conv2D,Activation,Dropout,MaxPooling2D,Cropping2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
#Var that will hold the experment number that will be used in the saved results pdf 
exp_no = '7'

#current path for the images folder
current_path = './data/IMG'
csv_path = './data/driving_log.csv'

#Angle correction Value
angle_correction_val = 0.25

#dropout keep prob
keep_prob = 0.8   #80 % keep and drop 20%

########################## read the csv sheet to get the samples and divide them to training and validation samples ######################################################################### 

#list with the CSV lines 
lines = [] 

with open (csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            lines.append(line)

#spilt the read sampels to training set and validation set(80% training and 20% validation) 
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))

######################################################################################################################################
# function to batch the training samples according to the batch size parameter and return this batch of the data after preprocessing it.
#the caller can get a batch after another by calling this function 
############################### #######################################################################################################
  
def PreprocAugGenerator(curr_img_path,samples,batch_size = 32):

    num_samples = len(samples)

    while 1 :
    
        shuffle(samples)
        
        print('samples shuffeld')
    
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            
            angles = []
            
            for sample in batch_samples:
                
                str_angle = float(sample[3])
                
                
                for i in range(3):
                    #extract the image name and concatenate them to the current path of the images 
                    image_source_path = sample[i]
                    image_name = image_source_path.split('\\')[-1]
                    image_current_path = curr_img_path + '/' + image_name
    
                    #open images and append them to the images list
                    image = cv2.imread(image_current_path)
                    images.append(image)
            
                    #flip image and append it also to the images list 
                    images.append(cv2.flip(image,1))
    
                    #extract the measurments value for the steering angle and add them to the measurments list
                    #Center Camera 
                    if(i == 0):
                        angles.append(str_angle)
                        angles.append((str_angle * -1))
                    #Left Camera add the correction angle parameter  
                    elif(i == 1):
                        angles.append(str_angle + angle_correction_val )
                        angles.append((str_angle + angle_correction_val ) * -1 )
                    #Right Camera subtract the correction angle parameter
                    elif(i == 2):
                        angles.append(str_angle - angle_correction_val)
                        angles.append((str_angle - angle_correction_val) * -1)
                        
            x_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(x_train, y_train)
                    
                    
#get the train generator data 
train_generator  = PreprocAugGenerator(current_path,train_samples,batch_size = 32)

#get the validation generator 
validation_generator = PreprocAugGenerator(current_path,validation_samples,batch_size = 32)


#The Model Arch. "initially I will use the Nividia End-End arch."
model = Sequential()

#normalization layer 
model.add(Lambda(lambda x: (x / 255.0) - 0.5 , input_shape = (160,320,3)))

model.add(Cropping2D(cropping = ((70,25),(0,0))))

#1st 5x5 cov layer with 24 filter depth and 2x2 stride 
model.add(Conv2D(24,5,5,subsample=(2,2)))

#1st layer activation 
model.add(Activation('relu'))

#1st conv dropout regularization layer 
model.add(Dropout(keep_prob))

#2nd 5x5 cov layer with 36 filter depth and 2x2 stride 
model.add(Conv2D(36,5,5,subsample=(2,2)))

#2nd layer activation 
model.add(Activation('relu'))

#2nd layer dropout 
model.add(Dropout(keep_prob))

#3rd 5x5 cov layer with 48 filter depth and 2x2 stride 
model.add(Conv2D(48,5,5,subsample=(2,2)))

#3rd layer activation 
model.add(Activation('relu'))

#3rd layer dropout
model.add(Dropout(keep_prob))

#4th 3x3 cov layer with 64 filter depth and 1x1 stride 
model.add(Conv2D(64,3,3,subsample=(1,1)))

#4th layer activation 
model.add(Activation('relu'))

#4th layer dropout
model.add(Dropout(keep_prob))

#5th 3x3 cov layer with 64 filter depth and 1x1 stride 
model.add(Conv2D(64,3,3,subsample=(1,1)))

#5th layer activation 
model.add(Activation('relu'))

#5th layer dropout
model.add(Dropout(keep_prob))


#Flatten layer
model.add(Flatten())

#1st fully connected layer 
model.add(Dense(100))

model.add(Dropout(keep_prob))

#2nd fully connected layer 
model.add(Dense(50))

model.add(Dropout(keep_prob))

#3rd fully connected layer 
model.add(Dense(10))

model.add(Dropout(keep_prob))

#output layer 
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

train_samples_per_epoch = ((len(train_samples)*3 )*2)
validation_samples_per_epoch = ((len(validation_samples) * 3  ) * 2) 

history_object = model.fit_generator(train_generator, samples_per_epoch=train_samples_per_epoch, validation_data=validation_generator,
                                    nb_val_samples=validation_samples_per_epoch, nb_epoch=10)

#save the modle in the output folder with the experment number
model.save('./output/model.h5_'+exp_no)

# histogram of label frequency
'''hist, bins = np.histogram(y_train, bins=len(np.unique(y_train)))
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
plt.savefig('./output/data_histogram_'+exp_no)'''


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

#save the experiment results in pdf 
plt.savefig('./output/MSE_'+exp_no)

plt.figure()
plt.show()


