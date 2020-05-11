import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import os
import cv2
import my_read
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)

def SPPCovidNet(class_no,input_height,input_width): 
    input_images=tf.keras.layers.Input(shape=(input_height,input_width,3))

    x = tf.keras.layers.Conv2D(8,(3,3),strides=(1,1),padding='same',use_bias=False)(input_images)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

    x = tf.keras.layers.Conv2D(16,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

    #first triple
    x = tf.keras.layers.Conv2D(32,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(16,(1,1),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(32,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

    #second triple
    x = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(32,(1,1),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

    #third triple
    x = tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(64,(1,1),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D((2,2),strides=(2,2))(x)

    #fourth triple
    x = tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(128,(1,1),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    #ending network
    L1 = tf.keras.layers.MaxPooling2D((7,7),strides=(1,1),padding='valid')(x)
    L2 = tf.keras.layers.MaxPooling2D((6,6),strides=(1,1),padding='valid')(x)
    L3 = tf.keras.layers.MaxPooling2D((4,4),strides=(1,1),padding='valid')(x)

    FL1 = tf.keras.layers.Flatten()(L1)
    FL2 = tf.keras.layers.Flatten()(L2)
    FL3 = tf.keras.layers.Flatten()(L3)
    
    x = tf.keras.layers.Concatenate(axis=1)([FL1,FL2,FL3])
    x = tf.keras.layers.Dense(class_no,activation='softmax')(x)

    # Create model.
    model=tf.keras.models.Model(inputs=input_images,outputs=x)
    model.summary()

    return model


def divide_data(iter_no,image1,image2,image3,div_data1,div_data2,div_data3):
    print("image1",image1.shape)
    print("image2",image2.shape)
    print("image3",image3.shape)
    print("div_data1",div_data1)
    print("div_data2",div_data2)
    print("div_data3",div_data3)
    
    delete_index1=np.arange(div_data1[iter_no],div_data1[iter_no+1])
    train_data1=np.delete(image1,delete_index1,axis=0)
    test_data1=image1[div_data1[iter_no]:div_data1[iter_no+1]]

    delete_index2=np.arange(div_data2[iter_no],div_data2[iter_no+1])
    train_data2=np.delete(image2,delete_index2,axis=0)
    test_data2=image2[div_data2[iter_no]:div_data2[iter_no+1]]

    delete_index3=np.arange(div_data3[iter_no],div_data3[iter_no+1])
    train_data3=np.delete(image3,delete_index3,axis=0)
    test_data3=image3[div_data3[iter_no]:div_data3[iter_no+1]]

    #combine the data
    train_data=np.concatenate((train_data1,train_data2,train_data3),axis=0)
    test_data=np.concatenate((test_data1,test_data2,test_data3),axis=0)

    return train_data,test_data
    
#parameters, train pterygium for 250 normal and 250 pterygium - test: 271 normal and 78 pterygium
covid_19_dir="C:/Users/asyra/Desktop/database/covid19 radiography database/COVID-19"
normal_dir="C:/Users/asyra/Desktop/database/covid19 radiography database/NORMAL"
viral_pneumonia_dir="C:/Users/asyra/Desktop/database/covid19 radiography database/Viral Pneumonia"
class_no=3
image_size=256

covid_19_name=my_read.get_all_non_hidden_files(covid_19_dir,'png','~')
normal_name=my_read.get_all_non_hidden_files(normal_dir,'png','~')
viral_pneumonia_name=my_read.get_all_non_hidden_files(viral_pneumonia_dir,'png','~')

for i in range(len(covid_19_name)):
    read_image=cv2.imread(covid_19_name[i])
    read_image=cv2.resize(read_image,(image_size,image_size))
    read_image=read_image.reshape(1,read_image.shape[0],read_image.shape[1],read_image.shape[2])
    if i==0:
        covid_image=np.copy(read_image)
    else:
        covid_image=np.append(covid_image,read_image,axis=0)

for i in range(len(normal_name)):
    read_image=cv2.imread(normal_name[i])
    read_image=cv2.resize(read_image,(image_size,image_size))
    read_image=read_image.reshape(1,read_image.shape[0],read_image.shape[1],read_image.shape[2])
    if i==0:
        normal_image=np.copy(read_image)
    else:
        normal_image=np.append(normal_image,read_image,axis=0)

for i in range(len(viral_pneumonia_name)):
    read_image=cv2.imread(viral_pneumonia_name[i])
    read_image=cv2.resize(read_image,(image_size,image_size))
    read_image=read_image.reshape(1,read_image.shape[0],read_image.shape[1],read_image.shape[2])
    if i==0:
        pneumonia_image=np.copy(read_image)
    else:
        pneumonia_image=np.append(pneumonia_image,read_image,axis=0)

fold_no=5
covid_div_data=[0]
normal_div_data=[0]
pneumonia_div_data=[0]
for i in range(fold_no):
    covid_div_data.append(int((i+1)*len(covid_image)/fold_no))
    normal_div_data.append(int((i+1)*len(normal_image)/fold_no))
    pneumonia_div_data.append(int((i+1)*len(pneumonia_image)/fold_no))
    
#create one hot for train data
covid_label=np.zeros((covid_image.shape[0],class_no))
covid_label[:,0]=1

normal_label=np.zeros((normal_image.shape[0],class_no))
normal_label[:,1]=1

pneumonia_label=np.zeros((pneumonia_image.shape[0],class_no))
pneumonia_label[:,2]=1
print("finish loading data and labels")

accuracy=[]
for iter_no in range(fold_no):
    print(str(iter_no)+" fold!")
    train_images,test_images=divide_data(iter_no,covid_image,normal_image,pneumonia_image,covid_div_data,normal_div_data,pneumonia_div_data)
    train_labels,test_labels=divide_data(iter_no,covid_label,normal_label,pneumonia_label,covid_div_data,normal_div_data,pneumonia_div_data)

    print("train_images",train_images.shape)
    print("train_labels",train_labels.shape)
    print("test_images",test_images.shape)
    print("test_labels",test_labels.shape)

    checkpoint_path="test/cp.ckpt"
    checkpoint_dir=os.path.dirname(checkpoint_path)

    #create checkpoint callback
    cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)
    input_height,input_width = image_size,image_size
    model=SPPCovidNet(class_no,input_height,input_width)
    epoch_no=100
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    H=model.fit(x=train_images,y=train_labels,epochs=epoch_no,batch_size=64,verbose=2)

    # plot the training loss and accuracy
    N = np.arange(0,epoch_no)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show(block=False)
    plt.pause(1)

    loss,acc=model.evaluate(test_images,test_labels,batch_size=1,verbose=2)
    print("current accuracy:",acc)
    accuracy.append(acc)

    tf.keras.backend.clear_session()

accuracy=np.array(accuracy)
print("accuracy",accuracy)
mean_accuracy=np.mean(accuracy,axis=0)
print("mean_accuracy",mean_accuracy)







