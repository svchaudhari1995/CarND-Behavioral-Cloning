




import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


lines=[] 
with open('driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in  reader:
        lines.append(line)
images=[]
measurements=[]
i=0;
images=[]
for line in lines[1:]:

    measurement=float(line[3])  #get steering angle of the car in the image
    col=np.random.choice([0,1,2]) # randomly select number 
    filename=line[col].split('/')[-1] #get filename from the column
#         image=mpimg.imread("IMG/"+filename)
    images.append(plt.imread("../data/IMG/"+filename)) #get image from the filepath 
    if(col==1):
        measurements.append(measurement+0.25)  #add 0.25 to steering angle for left images
    elif(col==2):
        measurements.append(measurement-0.25) #subtract 0.25 to steering angle for right images
    else:
        measurements.append(measurement)  

X_train=np.array(images)  #convert to numpy array
y_train=np.array(measurements)    #convert to numpy array


print(X_train.shape)

#Using sequential model in  keras
model=Sequential()
model.add(Cropping2D(cropping=((65, 25),(0, 0)),input_shape =(160, 320, 3))) #cropping unwanted section of image
model.add(Lambda(lambda x:(x/(255/2))-1.)) #normalizing images between value from -1 to 1
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam') #use adam optimizer
checkpoint = ModelCheckpoint(filepath='model.h5' ,monitor='val_loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5,callbacks=callbacks_list)
model.save('model.h5') #saving model
model.summary()
print("Done")