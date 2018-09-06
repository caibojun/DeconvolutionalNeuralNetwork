#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:03:00 2018

@author: jumtsai
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'''Import this part for using Tensor Board to visualizing each nodes in CNN.
'''
#DCNN's TensorFlow(GPU) Version
from astropy.io import fits
import os, glob, time
import logging,logging.handlers
import numpy as np
import tensorflow as tf
from manager import GPUManager
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#--------------------------------Envir Default---------------------------------#
gm = GPUManager()
max_step=150
log_dir='CNNinfo/'
save_dir=log_dir+'restore/'
checkpoint_dir=log_dir+'model/'
num,weight,height=(1,4096,4096)

#--------------------------------Logging Module--------------------------------#
LOG_FILE = log_dir+'train_detail.log' 
if os.path.isfile(LOG_FILE) is True:
    os.remove(LOG_FILE)
  
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes = 10*1024*1024, backupCount = 5) 
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'  
  
formatter = logging.Formatter(fmt)  
handler.setFormatter(formatter)     
  
logger = logging.getLogger('train_detail')  
logger.addHandler(handler)         
logger.setLevel(logging.DEBUG)  
  
#----------------------------------Function------------------------------------# 
def Padding(Input,ker_size):
    '''Image Padding Function
    '''
    xpadpre = int(np.floor(ker_size[0]/2.0))
    xpadpost = ker_size[0] - xpadpre 
    ypadpre = int(np.floor(ker_size[1]/2.0))
    ypadpost = ker_size[1] - ypadpre 
    paddings = [[0,0],[xpadpre,xpadpost],[ypadpre,ypadpost],[0,0]]
    padded = tf.pad(Input,paddings,"SYMMETRIC")
    return padded

def batch_normal(input,is_train ,is_out=True,decay=0.9999):
    with tf.name_scope('BN'):
        scale=tf.Variable(tf.ones([input.get_shape()[-1]]))
        beta=tf.Variable(tf.zeros([input.get_shape()[-1]]))
        pop_mean=tf.Variable(tf.zeros([input.get_shape()[-1]]),trainable=False)
        pop_var=tf.Variable(tf.ones([input.get_shape()[-1]]),trainable=False)
    
        if is_train:
            if is_out:
                batch_mean,batch_var = tf.nn.moments(input,[0,1,2])
            else:
                batch_mean,batch_var = tf.nn.moments(input,[0])
            
            train_mean = tf.assign(pop_mean,pop_mean*decay+batch_mean*(1-decay))
            train_var = tf.assign(pop_var,pop_var*decay+batch_var*(1-decay))
            with tf.control_dependencies([train_mean,train_var]):
                return tf.nn.batch_normalization(input,batch_mean,batch_var,beta,scale,0.0001)
        else:
            return tf.nn.batch_normalization(input,pop_mean,pop_var,beta,scale,0.0001)

def Conv_Layer(Input, k_num, k_size, p_size = 2, activity_func = None):
    '''Add convolutional layer Function, if tensor is Input, output size would be
    smaller than Input size.
    '''
    with tf.name_scope('Convolutional_Layer'):
        padded = Padding(Input, k_size)
        raw_image = tf.layers.Input(tensor = padded)
        shape=raw_image.get_shape().as_list()
        weights = tf.Variable(tf.truncated_normal([k_size[0],k_size[1],int(shape[3]),k_num],stddev=15.0,dtype=tf.float32))
        biases = tf.Variable(tf.truncated_normal([k_num],stddev=5.0,dtype=tf.float32))  
        unBN = tf.add(tf.nn.conv2d(raw_image, weights, strides=[1, 1, 1, 1], padding='VALID'), biases)
        Conv = batch_normal(unBN,is_train=True)
        if activity_func is not None:
            Act =activity_func(Conv,)    
            down_sample = tf.layers.max_pooling2d(Act, p_size, strides = 1, padding = 'valid')
        else:
            down_sample = tf.layers.max_pooling2d(Conv, p_size, strides = 1, padding = 'valid')
        return down_sample

def Block(feature, num, kernel):
    c_in = Conv_Layer(feature, num, [kernel,1], activity_func = tf.nn.relu)
    c_out= Conv_Layer(c_in, num, [1,kernel], activity_func = tf.nn.relu)
    return c_out -c_in

def mse(r,x):
    with gm.auto_choice(mode=0):
        return tf.reduce_mean(tf.square(r-x))

def tobatch(array,w,h):
    pixelsize=array.shape[0]*array.shape[1]
    batch=np.zeros([int(pixelsize/(h*w)),w,h],dtype=np.dtype('>i2'))
    k=0
    for i in range(int(array.shape[0]/h)):
        for j in range(int(array.shape[1]/w)):
            batch[k]=array[h*i:h*(i+1),w*j:w*(j+1)]
            k+=1
    return batch  

def toarray(mesh):
    n,w,h=mesh.shape
    n_w=n_h=int(np.sqrt(n*w*h))
    grid=np.sqrt(n)
    array=np.zeros([n_w,n_h])
    for i in range(n):
        m=int(i/grid)
        n=int(i%grid)
        array[m*w:(m+1)*w,n*h:(n+1)*h]=mesh[i]
    return array
#------------------------------------Input-------------------------------------#

with tf.name_scope('Placeholder'):
    blur = tf.placeholder(tf.float32,[1 ,4096, 4096, 1], name='Blur') 
    oril = tf.placeholder(tf.float32,[1 ,4096, 4096, 1], name='Oril')
    imgsize = oril.get_shape().as_list()
    batch_size = imgsize[0]
    
    tf.summary.image('Blur_input', blur, 10)
    tf.summary.image('Oril_input', oril, 10)

#-----------------------------Add Hidden Layers--------------------------------#
with tf.name_scope('Hidden_Layer'):                                                                                                     #Using multi GPUs
    with gm.auto_choice(mode=0):                                                                                                           #Allocating single GPUs
        cl1 = Conv_Layer(blur, 16, [63,63], activity_func = tf.nn.relu)
        cl2 = Block(cl1-blur, 16, 63)
        cl3 = Block(cl2, 32, 31)
        cl4 = Block(cl3, 64, 15)
        cl5 = Block(cl4,128, 11)
        cl6 = Block(cl5, 144, 9)
        cl7 = Block(cl6, 192, 7)
        cl8 = Block(cl7, 256, 5)
        cl9 = Block(cl8, 512, 3)
        cl10 = Block(cl9,1024, 1)

        dense = tf.layers.dense(cl10, 1)

	#dc3 = DeConv_Layer(cl3,12,1,activity_func = tf.nn.relu)
  
with tf.name_scope('Loss'):
    pre = tf.reshape(dense,imgsize)+blur
    loss=mse(pre,oril)
    tf.summary.image('Output', pre, 10)
    tf.summary.scalar('loss', loss)  
    
with tf.name_scope('Train'): 
    step=tf.Variable(0,trainable=False)
    learnrate=tf.train.exponential_decay(5.0, step, 100, 0.96, staircase = True)
    train_step = tf.train.AdadeltaOptimizer(learnrate).minimize(loss, global_step = step)

#---------------------------------Read Data------------------------------------#   
files=glob.glob('gauss127/*.fits')
files.sort(reverse = True)
trainset=open('train.txt','r')
train=[]
for trainname in trainset:
    train.append(trainname.split('\n')[0])
   
#-----------------------------------Initiate-----------------------------------#    
init = tf.global_variables_initializer() 
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config = config) as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + 'test', sess.graph)
    saver = tf.train.Saver(max_to_keep=128)
    sess.graph.finalize() 
#----------------------------------Iteration-----------------------------------#

    for epoch in range(1,max_step+1):
        trainloss = []
        testloss = []
        s=time.clock()
        for fitsfile in files:
            name=fitsfile.split('/')[-1]
            blurred=fits.open('gauss127/'+name)[0].data
            try:
                original=fits.open('original/'+name)[0].data
            except IOError:
                original=np.zeros(blurred.shape)
            blurred=tobatch(blurred,weight,height)
            original=tobatch(original,weight,height)
            blurred=blurred.reshape([num,weight,height,1])
            original=original.reshape([num,weight,height,1]) 
            if name in train:
                epoch_result =np.zeros([num,weight,height])
                for batch in range(0,num):
                    Input_x=np.zeros([1,weight,height,1],dtype=np.float32)
                    Input_y=np.zeros([1,weight,height,1],dtype=np.float32)
                    Input_x[0]=np.float32(blurred[batch])
                    Input_y[0]=np.float32(original[batch])
                    _,lvalue,summary,result=sess.run([train_step,loss,merged,pre],
                                                     feed_dict={blur:Input_x,oril:Input_y})
                    train_writer.add_summary(summary, epoch)
                    saver.save(sess, checkpoint_dir + 'model'+str(batch)+'.ckpt', global_step=batch+1,write_meta_graph=False,write_state=False)  
                    epoch_result[batch]=result.reshape(weight,height)
                epoch_result=toarray(epoch_result)
                recon=np.int16(epoch_result)
                trainloss.append(lvalue)        
                train_writer.close() 
                if os.path.isfile(save_dir+'Train_'+name) is True:
                    os.remove(save_dir+'Train_'+name)
                fits.HDUList([fits.PrimaryHDU(recon)]).writeto(save_dir+'Train_'+name)
            else:
                epoch_result =np.zeros([num,weight,height])
                for batch in range(0,num):
                    Input_x=np.zeros([1,weight,height,1],dtype=np.float32)
                    Input_y=np.zeros([1,weight,height,1],dtype=np.float32)
                    Input_x[0]=np.float32(blurred[batch])
                    Input_y[0]=np.float32(original[batch])
                    saver.restore(sess, checkpoint_dir +'model'+str(batch)+'.ckpt-'+str(batch+1)) 
                    lvalue,summary,result=sess.run([loss,merged,pre],
                                                   feed_dict={blur:Input_x,oril:Input_y})
                    test_writer.add_summary(summary, epoch)
                    epoch_result[batch]=result.reshape(weight,height)
                epoch_result=toarray(epoch_result)
                recon=np.int16(epoch_result)
                testloss.append(lvalue)     
                test_writer.close()   
                if os.path.isfile(save_dir +'Test_'+name) is True:
                    os.remove(save_dir+'Test_'+name)
                fits.HDUList([fits.PrimaryHDU(recon)]).writeto(save_dir+'Test_'+name)
        e=time.clock()
        print('Epoch %d mean train loss is %e, time is %f.'%(epoch,np.mean(trainloss),(e-s)))
        print('Epoch %d mean test loss is %e.'%(epoch,np.mean(testloss)))
        logger.info('Epoch %d mean train loss is %e, time is %f'%(epoch,np.mean(trainloss),(e-s))) 
        logger.info('Epoch %d mean test loss is %e.'%(epoch,np.mean(testloss)))
        if os.path.isfile('residual/'+name) is True:
            os.remove('residual/'+name)
        original=toarray(original.reshape(num,weight,height))
        fits.HDUList([fits.PrimaryHDU(original-recon)]).writeto('residual/'+name)