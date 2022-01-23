import cv2
import numpy as np
import tensorflow as tf
from skimage.morphology import disk, dilation, erosion
from scipy import ndimage as ndi

class CoinMeasurement:

    def __init__(self, image):
        self._n_white_pix, self._masked, self._mask = self._coin_segmenting_and_measurement(image)

    def get_data(self):
        return self._n_white_pix, self._masked, self._mask

    def _undesired_objects(self, image):

        """
        A function used to detect the largest area covered in an image and filters out the areas that are smaller
        than this object accordigly lossing undesired objects from the input image

        Input: filtered binary image
        Output: Binary image of the largest area object
        """

        image = image.astype('uint8')
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        try:
            max_size = sizes[1]
        except:
            print('No coin detected')
        try:
            for i in range(2, nb_components):
                    y = sizes[i]
                    if y > max_size:
                        max_label = i
                        max_size = sizes[i]

        except:
            print("There is no coin detected")

        img2 = np.zeros(output.shape)
        img2[output == max_label] = 255
        return img2

    def _coin_segmenting_and_measurement(self, image):

        #  Image preprocessing: converting the input image from RGB to Gray Scale ,then bluing the image
        #  while preserving the edges using the medium filter from the openCV libraries
        # Get a Numpy BGR image from an RGB tf.Tensor
        image_tensor = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        image_cv = cv2.cvtColor(image_tensor.numpy(), cv2.COLOR_RGB2BGR)
        image_smooth = cv2.medianBlur(image_cv, 19)

        #  Detecting the edges using canny edge detector:
        image_edges = cv2.Canny(image_smooth, 21, 51)

        # Now, we'll fill the closed edges using fill_holes in order to fill the coin edges with binary value
        # using binary dilations
        fill = ndi.binary_fill_holes(image_edges)

        # Finally, we'll apply the disk morphological operation to remove unwanted edges ,
        # the edges that are not circular shaped
        mask1 = disk(5)
        x = erosion(fill, selem=mask1)
        x = dilation(x, selem=mask1)

        # Then, we clean the binary image in case more than one circular object appears in the image we select
        # the largest in area
        cleaned = self._undesired_objects(x)/255
        mask = cleaned.astype("uint8")

        # Finally, we apply the output mask to visualize the output on the original image
        masked = cv2.bitwise_and(image_cv, image_cv, mask=mask)
        n_white_pix = np.sum(mask == 1)
        return n_white_pix, masked, mask