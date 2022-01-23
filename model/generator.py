import os
import math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

class DataGen(keras.utils.Sequence):
    
    def __init__(self, path, batch_size, image_size, labels=True, shuffle=True ,aug_ratio=0.5, seed = 42):
        self.image_size = image_size
        self.batch_size = batch_size
        #setting the ratio of augmented samples
        self.aug_ratio = aug_ratio
        self.path = path
        self.list_IDs = os.listdir(os.path.join(self.path, 'images'))
        self.labels = labels
        if labels:
            self.label_IDs = os.listdir(os.path.join(self.path, 'labels'))
        self.shuffle = shuffle
        self.on_epoch_end()
        # adding 2 arrays for images and labels augmeters in the class
        #check this link https://www.tensorflow.org/api_docs/python/tf/keras/layers/
        self.augment_inputs =[tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
                              tf.keras.layers.RandomFlip(mode="vertical", seed=seed),                              
                              tf.keras.layers.RandomRotation(factor=0.2,seed=seed),
                              tf.keras.layers.RandomZoom(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1),seed=seed)] 
        self.augment_labels = [tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
                              tf.keras.layers.RandomFlip(mode="vertical", seed=seed),                             
                              tf.keras.layers.RandomRotation(factor=0.2,seed=seed),
                              tf.keras.layers.RandomZoom(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1),seed=seed)]
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, 'images', id_name)

        ## Reading Image
        image = cv2.imread(image_path, 1)[:,:,::-1]
        image = cv2.resize(image, self.image_size)
        
        if self.labels:
            mask_path = os.path.join(self.path, 'labels', id_name)
            mask = cv2.imread(mask_path, 1)[:,:,::-1]
            mask = cv2.resize(mask[:,:,1], self.image_size)
            mask = np.expand_dims(mask, axis=-1)
            mask = mask/255.0
            
        ## Normalizaing 
        image = image/255.0
        
        if self.labels:
            return image, mask
        
        else:
            return image
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.list_IDs):
            self.batch_size = len(self.list_IDs) - index*self.batch_size
        
        files_batch = self.list_IDs[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            
            if self.labels:
                _img, _mask = self.__load__(id_name)
                mask.append(_mask)
                
            else:
                _img = self.__load__(id_name) 
            image.append(_img)
            
        image = np.array(image)
        mask  = np.array(mask)
        #selecting random index from augmenters array
        randomIdx = np.random.randint(len(self.augment_inputs))
        #calculating the number of augmented samples based on ratio and epoch size
        num_augmented = int(self.aug_ratio*len(image))
        #augment iamges
        image[:num_augmented] = self.augment_inputs[randomIdx](image[:num_augmented])
        if self.labels:
            #augment labels
            image[:num_augmented] = self.augment_labels[randomIdx](image[:num_augmented])
            return image, mask
        else:
            return image
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return math.ceil(len(self.list_IDs) / self.batch_size)