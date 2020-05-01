#!/usr/bin/env python
# coding: utf-8

# ### Hackerearth Deep learning competition : Detect emotions of your favorite toons (Tom and Jerry) !

# ### 1. Frames generation code

# In[ ]:


import cv2
videoFile = "video.mp4"
import math
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = 'images/' +  str(int(x)) + ".jpg";x+=1
        cv2.imwrite(filename, frame)

cap.release()
print ("Done!")


# In[ ]:





# ### 2. Segmentation of Tom and Jerry faces 
# Use code for both train frames and test frames
# 

# In[ ]:


import cv2
import os


def facecrop(image,character):
    
    if character['name'] == "Tom":

        img = cv2.imread(image)
        cascade = cv2.CascadeClassifier(character['cascade'])
        
        faces = cascade.detectMultiScale(img,scaleFactor=1.10, 
                        minNeighbors=40, 
                        minSize=(24, 24),
                        flags=cv2.CASCADE_SCALE_IMAGE)
        
        
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

            sub_face = img[y:y+h, x:x+w]
            fname, ext = os.path.splitext(image)
        
            cv2.imwrite(fname+ext, sub_face)

    elif character['name'] == "Jerry":

        img = cv2.imread(image)
        cascade = cv2.CascadeClassifier(character['cascade'])
       
        faces = cascade.detectMultiScale(img,scaleFactor=1.10, 
                        minNeighbors=20, 
                        minSize=(24, 24),flags=cv2.CASCADE_SCALE_IMAGE)
        
        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

            sub_face = img[y:y+h, x:x+w]
            fname, ext = os.path.splitext(image)
            
            cv2.imwrite(fname+ext, sub_face)


# In[ ]:


characters1 = [
    {
        'name':      "Tom",
        'cascade':   './haar_cascades/tom.xml'
        
    }
]
characters2 = [
    {
        'name':      "Jerry",
        'cascade':   './haar_cascades/jerry.xml'
    }
]

folder_path= '' #folder path
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    for ch in characters1:
        facecrop(img,ch)
for img in os.listdir(folder_path):
    img = os.path.join(folder_path, img)
    for ch in characters2:
        facecrop(img,ch)


# In[ ]:





# ### 3. Model development and Training

# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
 
# define cnn model
def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(60, 60, 3))
    # mark loaded layers as not trainable
    classes_num=5
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(classes_num, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# run the test harness for evaluating a model
def run_test_harness():
# define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('train_data_folder/',
    class_mode='categorical', batch_size=32, target_size=(60, 60))
    # fit model
    model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=60, verbose=1)
    # save model
    model.save('result.h5')
 # entry point, run the test harness
run_test_harness()


# In[ ]:





# ### 4. Testing on test data

# In[19]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os


folder_path = './test_data_folder/'  #after segmentation
prediction=[]
model = load_model('result.h5')
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(60, 60))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 60, 60, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img
 
# load an image and predict the class
def run_example():
    # load the image
    
    for img in os.listdir(folder_path):
    # load model
#         print(img)
        x=img
        img = os.path.join(folder_path, img)
        img = load_image(img)
       
    # predict the class
        result = model.predict(img)
        m=max(result[0])

        itemindex = np.where(result[0]==m)

        if(itemindex[0][0])==0:
            prediction.append((x,'angry'))
        elif(itemindex[0][0])==1:
            prediction.append((x,'happy'))
        elif(itemindex[0][0])==2:
            prediction.append((x,'sad'))
        elif(itemindex[0][0])==3:
            prediction.append((x,'surprised'))
        elif(itemindex[0][0])==4:
            prediction.append((x,'Unknown'))
            
    
# entry point, run the example
run_example()


# In[21]:


# to get results as per expected format
def Sort_Tuple(tup):  
    
    tup.sort(key = lambda x: (int(x[0].split('.')[0]), x[0]))  
    return tup 
l=Sort_Tuple(prediction)
res=[]
for i in l:
    res.append(i[1])
# print(res)


# ### 5. Make results.csv file

# In[ ]:


import pandas as pd
data=pd.read_csv('Test.csv')
data['Emotion'] = res
data.to_csv('results.csv',index=False)


# In[ ]:




