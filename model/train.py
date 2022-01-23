import tensorflow as tf
import os
from model.generator import DataGen
from keras.callbacks import EarlyStopping
from model.metric import dice_coef, precision, recall
from model.model import unet_model


class TrainModel:

    def __init__(self, train_path='data/train', valid_path='data/validation', **kwargs):
        self.train_path, self.valid_path = train_path, valid_path
        self.training_images, self.training_labels = os.path.join(self.train_path, 'images/'), \
                                                     os.path.join(self.train_path, 'labels/')
        self.validation_images, self.validation_labels = os.path.join(self.valid_path, 'images/'), \
                                                         os.path.join(self.valid_path, 'labels/')
        self.train_dir_fname = os.listdir(self.training_images)
        self.validation_dir_fname = os.listdir(self.validation_images)
        self.parameters = {
            'image_size': (256, 256),
            'epochs': 2000,
            'num_channels': 3,
            'batch_size': 2,
            'optimizer': tf.keras.optimizers.Adam(lr=1e-4),
            'loss': 'binary_crossentropy',
            'metric': [dice_coef, precision, recall]
        }

        train_steps = len(self.train_dir_fname) // self.parameters['batch_size']
        valid_steps = len(self.validation_dir_fname) // self.parameters['batch_size']

        train_generator = DataGen(train_path, batch_size=self.parameters['batch_size'],
                                  image_size=self.parameters['image_size'])

        validation_generator = DataGen(valid_path, batch_size=self.parameters['batch_size'],
                                       image_size=self.parameters['image_size'])

        self.parameters['train_steps'] = train_steps
        self.parameters['valid_steps'] = valid_steps
        self.parameters['train_generator'] = train_generator
        self.parameters['validation_generator'] = validation_generator
        self.__unet_summary = None
        self.__model_history = self._train()

    def _train(self):
        unet = unet_model(
            (self.parameters['image_size'[0]], self.parameters['image_size'[1]], self.parameters['num_channels']),
            n_classes=1)

        self.__unet_summary = unet.summary()

        callbacks = EarlyStopping(monitor='val_dice_coef', patience=200, mode='max', restore_best_weights=True)

        unet.compile(optimizer=self.parameters['optimizer'],
                     loss=self.parameters['loss'],
                     metrics=self.parameters['metric'])

        history = unet.fit(self.parameters['train_generator'],
                           validation_data=self.parameters['validation_generator'],
                           steps_per_epoch=self.parameters['train_steps'],
                           validation_steps=self.parameters['valid_steps'],
                           epochs=self.parameters['epochs'], callbacks=[callbacks])

        unet.save('model/my_model.h5')

        return history

    def get_history(self):
        return self.__model_history

    def get_summary(self):
        return self.__unet_summary
