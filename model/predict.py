import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from model.coin_measurement import CoinMeasurement


def _display_coin(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Coin Segmenting', 'Mask Application']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = display_list[i].astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


class Predict:

    def __init__(self, directory, unet, parameters, show_coin=False):
        self.unet = unet
        self.parameters = parameters
        self.show_coin = show_coin
        for path in os.listdir(directory):
            if path.endswith(".png"):
                path = os.path.join(directory, path)
                image_list = [path, path]
                image_filenames = tf.constant(image_list)
                dataset = tf.data.Dataset.from_tensor_slices((image_filenames))
                image_ds = dataset.map(self._process_path)
                processed_image_ds = image_ds.map(self._preprocess)
                prediction = processed_image_ds.batch(self.parameters['batch_size']).cache()
                self._show_predictions(prediction, 1)

    def _process_path(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def _preprocess(self, image):
        input_image = tf.image.resize(image, self.parameters['image_size'], method='nearest')
        return input_image

    def _display(self, display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            img = np.array(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    def _create_mask(self, pred_mask):
        pred_mask[pred_mask >= 0.25] = 1
        pred_mask[pred_mask < 0.25] = 0
        size = np.sum(pred_mask)
        return pred_mask, size

    def _show_predictions(self, dataset=None, num=1):
        """
        Displays the first image of each of the num batches
        """
        if dataset:
            for image in dataset.take(num):
                pred_masks = self.unet.predict(image)
                img, pred_mask = np.copy(image[0]), np.copy(pred_masks[0])
                pred_mask, wound_size = self._create_mask(pred_mask)
                coin_measurement = CoinMeasurement(img)
                coin_size, coin_image, _ = coin_measurement.get_data()

                image_tensor = tf.image.convert_image_dtype(img, dtype=tf.uint8)
                image_cv = cv2.cvtColor(image_tensor.numpy(), cv2.COLOR_RGB2BGR)

                image_cv, pred_mask = cv2.resize(image_cv, (256,256)), cv2.resize(pred_mask, (256,256)).astype("uint8")
                pred_mask = cv2.bitwise_and(image_cv[:,:,::-1], image_cv[:, :, ::-1], mask=pred_mask)
                pred_mask = cv2.bitwise_or(coin_image, pred_mask)

                if coin_size != 0.0:
                    print('Wound size = ', wound_size, 'px\t, Coin size = ', coin_size, 'px\t, Ratio = ', '{:.2f}'.format(wound_size/coin_size))

                # Displaying the output mask
                if self.show_coin:
                    _display_coin([image_cv[:, :, ::-1], _, coin_image])
                self._display([image_cv[:, :, ::-1], pred_mask])