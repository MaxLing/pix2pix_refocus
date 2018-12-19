import os, cv2
import tensorflow as tf

class Dataset():
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        self.image_size = image_size # original 512*512

        self.defocus_dir = self.data_dir + '/train/'
        self.focus_dir = self.data_dir + '/train_gt/'

        # read image id, assuming paired images are provided
        self.image_ids = [id for id in os.listdir(self.defocus_dir)]

    def load_data_batch(self):
        # tf dataset from generator
        types = tf.float32
        shapes = tf.TensorShape([self.image_size[0], self.image_size[1], 3])
        return tf.data.Dataset.from_generator(self.pairs_generator, (types, types), (shapes, shapes))

    def pairs_generator(self):
        for id in self.image_ids:
            # note change BGR to RGB
            defocus_image = cv2.imread(self.defocus_dir + id)[...,::-1]
            focus_image = cv2.imread(self.focus_dir + id)[...,::-1]

            yield defocus_image, focus_image
