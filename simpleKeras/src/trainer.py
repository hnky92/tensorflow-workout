import tensorflow as tf

from tensorflow import keras
from src.model import LSTMModel
from src.generator import ManageData

from absl import flags

FLAGS = flags.FLAGS

class Trainer(object):
    def __init__(self):
        self.model = LSTMModel()

        self.tr_dataset = self._make_dataset(FLAGS.src_train_data)
        self.vl_dataset = self._make_dataset(FLAGS.src_valid_data)


    def _make_dataset(self, src_data):
        """
        make tf.dataset from python generator
        set training iteration parameters
        """
        generator = ManageData(FLAGS.src_token2id, [src_data], FLAGS.max_len).generator

        dataset = tf.data.Dataset.from_generator(generator,
                                                (tf.int32, tf.int32),
                                                (tf.TensorShape([None]),tf.TensorShape([])))

        dataset = dataset.shuffle(buffer_size=1024).batch(FLAGS.batch_size)

        return dataset


    def train(self):

        self.model.compile(optimizer='Adam',
                           metrics=['accuracy'],
                           loss='mse')
                           #loss=keras.losses.CategoricalCrossentropy(from_logits=True,
                           #                                          label_smoothing=0.2))

        self.model.fit(self.tr_dataset, 
                       epochs=FLAGS.epochs, 
                       validation_data=self.vl_dataset,
                       steps_per_epoch=(FLAGS.num_train_data/FLAGS.batch_size))



