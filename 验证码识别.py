# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:02:18 2019

@author: fz
"""

import numpy as np
import os
import pickle

from keras.utils import np_utils
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
from matplotlib import pyplot as plt
import cv2 as cv
from tensorflow import reshape
import copy
from random import random

#files=[]
#for i in listdir('./G7_captcha/'):
#    files.append(i)
#
#X_test = np.zeros((len(files), 50, 200, 3), dtype = np.uint8)
#y_test = np.zeros((len(files), word_len * word_class), dtype=np.uint8)
#for i in range(len(files)):
#    X_test[i]=cv.imread('./G7_captcha/'+files[i])
#    y_test[i]=captcha_to_vec(files[i].replace('.jpg',''))
#pickle.dump((X_test,y_test),open('./data/g7_captcha_train_data.pkl', 'wb'))



#验证码所包含的字符 _表示未知
captcha_word = "_23456789abcdefghjmnpqrtuwxyz"

#图片的长度和宽度
width = 200
height = 50

#每个验证码所包含的字符数
word_len = 5
#字符总数
word_class = len(captcha_word)

#验证码素材目录
train_dir = 'train'

#生成字符索引，同时反向操作一次，方面还原
char_indices = dict((c, i) for i,c in enumerate(captcha_word))
indices_char = dict((i, c) for i,c in enumerate(captcha_word))

def del_noise(img,number):
    height = img.shape[0]
    width = img.shape[1]

    img_new = copy.deepcopy(img)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            point = [[], [], []]
            count = 0
            point[0].append(img[i - 1][j - 1])
            point[0].append(img[i - 1][j])
            point[0].append(img[i - 1][j + 1])
            point[1].append(img[i][j - 1])
            point[1].append(img[i][j])
            point[1].append(img[i][j + 1])
            point[2].append(img[i + 1][j - 1])
            point[2].append(img[i + 1][j])
            point[2].append(img[i + 1][j + 1])
            for k in range(3):
                for z in range(3):
                    if point[k][z] == 0:
                        count += 1
            if count <= number:
                img_new[i, j] = 255
    return img_new

def img_process(img):
    img=img[10:50,40:180]
    # 灰度化
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    result = cv.adaptiveThreshold(grayImage, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 1)
    # 去噪声
    img = del_noise(result, 6)
    #            img = del_noise(img, 4)
    #            show_img(img)#4
    #            img = del_noise(img, 3)
    #            show_img(img)#4
    # 加滤波去噪
    im_temp = cv.bilateralFilter(src=img, d=15, sigmaColor=130, sigmaSpace=150)
    im_temp = im_temp[1:-1,1:-1]
    im_temp = cv.copyMakeBorder(im_temp, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[255])
    im_temp = cv.resize(im_temp, (width,height), interpolation=cv.INTER_CUBIC)
    tmp=np.zeros((height, width,3), dtype = np.uint8)
    for i in range(len(im_temp)):
        for j in range(len(im_temp[i])):
            tmp[i][j]=np.array([im_temp[i][j],im_temp[i][j],im_temp[i][j]])
    return tmp

#验证码字符串转数组
def captcha_to_vec(captcha):    
    #创建一个长度为 字符个数 * 字符种数 长度的数组
    vector = np.zeros(word_len * word_class)
    
    #文字转成成数组
    for i,ch in enumerate(captcha):
        idex = i * word_class + char_indices[ch]
        vector[idex] = 1
    return vector

#把数组转换回文字
def vec_to_captcha(vec):
    text = []
    #把概率小于0.5的改为0，标记为错误
    vec[vec < 0.5] = 0
        
    char_pos = vec.nonzero()[0]
    
    for i, ch in enumerate(char_pos):
        text.append(captcha_word[ch % word_class])
    return ''.join(text)

#test_vec = captcha_to_vec("2ngF4")
#vec_test = vec_to_captcha(test_vec)

##print(test_vec)
##print(vec_test)
##
    ##获取目录下样本列表
def data_process():
    image_list = []
    train_dir=r'./G7_captcha'
    #
    for item in os.listdir(train_dir):
        image_list.append(item)
    np.random.shuffle(image_list)
    
    
    #创建数组，储存图片信息。结构为(50321, 36, 120, 3)，50321代表样本个数，然后是宽度和高度。
    # 3代表图片的通道数，如果对图片进行了灰度处理，可以改为单通道 1
    X = np.zeros((len(image_list), height, width, 3), dtype = np.uint8)
    # 创建数组，储存标签信息
    y = np.zeros((len(image_list), word_len * word_class), dtype = np.uint8)
    
    for i,img in enumerate(image_list):
        if i % 10000 == 0:
            print(i)
        img_path = train_dir + "/" + img
        #读取图片
        raw_img = image.load_img(img_path, target_size=(height, width))
        #讲图片转为np数组
        X[i] = image.img_to_array(raw_img)
        #讲标签转换为数组进行保存
        y[i] = captcha_to_vec(img.split('.')[0][:word_len])
    
    
    
    #保存成pkl文件
    file = open('./data/g7_captcha_train_data.pkl','wb')
    pickle.dump((X,y) , file)

#读取pickle文件
file = open('./data/g7_captcha_train_data.pkl', 'rb')
X, y = pickle.load(file)

X_test = np.zeros((len(X), height, width, 3), dtype = np.uint8)
for i in range(len(X)):
    X_test[i]=img_process(X[i])
del X
X=X_test
#创建输入，结构为 高，宽，通道
input_tensor = Input( shape=(height, width, 3))

x = input_tensor

#构建卷积网络
#两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = Convolution2D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)


x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = Convolution2D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

x = Convolution2D(128, 3, padding='same', activation='relu')(x)
x = Convolution2D(128, 3, padding='same',activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
x = Flatten()(x)
#为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
x = Dropout(0.5)(x)

#最后连接5个分类器，每个分类器是29个神经元，分别输出29个字符的概率。
#Dense就是常用的全连接层
x = [Dense(word_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(word_len)]
output = concatenate(x)


#构建模型
model = Model(inputs=input_tensor, outputs=output)

#因为训练可能需要数个小时，所以这里加载了之前我训练好的参数。准确率为94%
#可以直接使用此参数继续进行训练，也可以自己从头开始训练
##model.load_weights('captcha_weights.0.9430.hdf5')

#这里优化器选用Adadelta，学习率0.1
opt = Adadelta(lr=0.1)
#opt='adam'
#编译模型以供训练，损失函数使用 categorical_crossentropy，使用accuracy评估模型在训练和测试时的性能的指标
model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#每次epoch都保存一下权重，用于继续训练
batch_num=str(random())[2:]
checkpointer = ModelCheckpoint(filepath="data/output/g7/weights"+batch_num+".{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5", 
                               verbose=2, save_weights_only=True)
#开始训练，validation_split代表10%的数据不参与训练，用于做验证集
#我之前训练了50个epochs以上，这里根据自己的情况进行选择。如果输出的val_acc已经达到你满意的数值，可以终止训练
model.fit(X, y, epochs= 500,callbacks=[checkpointer], validation_split=0.2)


#保存权重和模型
model.save_weights('data/output/g7/captcha_model_weights.h5')
model.save('data/output/g7/captcha__model.h5')

#测试验证方法
def testCaptcha(index): 
    raw_img = X[index]
    true_label = y[index]
    
    X_test = np.zeros((1, height, width, 3), dtype = np.float32)
    X_test[0] = image.img_to_array(raw_img)
    
    result = model.predict(X_test)
    
    vex_test = vec_to_captcha(result[0])
    true_test = vec_to_captcha(true_label)
    
    plt.imshow(raw_img)
    plt.show()
    
    print("原始：",true_test,"预测", vex_test)

def transfer_img(img_path):
    return cv.resize(cv.imread(img_path), (width,height), interpolation=cv.INTER_CUBIC)

def test_tmp(img):
    img=img_process(img)
    X_test = np.zeros((1, height, width, 3), dtype = np.float32)
    X_test[0] = image.img_to_array(img)
    result = model.predict(X_test)
    print(vec_to_captcha(result[0]))
    plt.imshow(img)
    plt.show()
def rename_tmp(file):
    path=r'./tmp_captcha'
    img=cv.imread(os.path.join(path,file))
    img=img_process(img)
    X_test = np.zeros((1, height, width, 3), dtype = np.float32)
    X_test[0] = image.img_to_array(img)
    result = model.predict(X_test)
    os.rename(os.path.join(path,file),os.path.join(path,vec_to_captcha(result[0])+".jpg"))
   
    print(vec_to_captcha(result[0]))
#test_tmp(cv.resize(X[1][0:100, 45:165], (width,height)))
#选5张验证码进行验证
for i in range(5):
    testCaptcha(i)
