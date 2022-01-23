import tensorflow as tf
from model.metric import dice_coef, precision, recall
from model.generator import DataGen
from model.predict import Predict
from model.train import TrainModel

parameters = {
    'image_size': (256, 256),
    'batch_size': 2,
}

# To Train the model:
trainModel = TrainModel()
model_history = trainModel.get_history()
model_summary = trainModel.get_summary()

test_images = 'data/test'

unet = tf.keras.models.load_model('model/my_model.h5', custom_objects={'recall': recall,
                                                                       'precision': precision,
                                                                       'dice_coef': dice_coef})

test_generator = DataGen(test_images, batch_size=parameters['batch_size'],
                         image_size=parameters['image_size'], labels=False)

output = unet.predict(test_generator)

# To Predict one image:
image_directory = 'real_data'
predict = Predict(image_directory, unet, parameters)
