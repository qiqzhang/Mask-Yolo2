import tensorflow as tf


class BaseDataGenerator:

    def __init__(self, config):
        self.config = config

        self.images_list,self.labels_list,self.dataset_size  = self.get_images_labels_list()
       #self.labels_list = self.get_labels_list()

        self.dataset = tf.data.Dataset.from_tensor_slices((tf.constant(self.images_list),tf.constant(self.labels_list)))


        self.dataset = self.dataset.map(self.parse_function,num_parallel_calls=4)
        self.dataset = self.dataset.shuffle(self.dataset_size)
        self.dataset = self.dataset.batch(self.config.batch_size)
        self.dataset = self.dataset.prefetch(1)

        self.iterator = self.dataset.make_initializable_iterator()

    def init_itorator(self):
        return self.iterator.initializer

    def next_batch(self):
        return self.iterator.get_next()

    def parse_function(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def get_images_list(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def get_labels_list(self):
        raise NotImplementedError('Must be implemented by the subclass.')

    def get_images_labels_list(self):
        raise NotImplementedError('Must be implemented by the subclass.')