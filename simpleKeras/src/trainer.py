import tensorflow as tf
import os, datetime

from tensorflow import keras
from src.model import LSTMModel
from src.generator import ManageData

from absl import flags

FLAGS = flags.FLAGS

class Trainer(object):
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=FLAGS.mem_limit)])
            except RuntimeError as e:
                print(e)

        self.model = LSTMModel(vocab_size = FLAGS.vocab_size, 
                               vec_dim = FLAGS.vec_dim, 
                               max_len = FLAGS.max_len, 
                               num_class = FLAGS.num_class, 
                               train = True, 
                               src_token2id = FLAGS.src_token2id, 
                               src_w2v = FLAGS.src_w2v)

        self.tr_dataset = self._make_dataset(FLAGS.src_train_data)
        self.vl_dataset = self._make_dataset(FLAGS.src_valid_data, repeat=False)


    def _make_dataset(self, src_data, repeat=True):
        """
        make tf.dataset from python generator
        set training iteration parameters
        """
        generator = ManageData(FLAGS.src_token2id, [src_data], FLAGS.max_len).generator

        dataset = tf.data.Dataset.from_generator(generator,
                                                (tf.int32, tf.int32),
                                                (tf.TensorShape([None]),tf.TensorShape([None])))

        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.shuffle(buffer_size=1024).batch(FLAGS.batch_size)

        return dataset


    def train(self):
        savepath = os.path.join('checkpoint', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


        self.model.compile(optimizer='Adam',
                           metrics=['accuracy'],
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                     label_smoothing=0.1))

        callbacks = [keras.callbacks.EarlyStopping(monitor='loss', 
                                                   patience=3),
                     keras.callbacks.TensorBoard(log_dir=os.path.join(savepath,'logs'), 
                                                 write_graph=True, 
                                                 update_freq='epoch'),
                     keras.callbacks.ModelCheckpoint(os.path.join(savepath,'model.h5'),
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)] 

        self.model.fit(self.tr_dataset, 
                       epochs=FLAGS.epochs, 
                       validation_data=self.vl_dataset,
                       callbacks=callbacks,
                       steps_per_epoch=int((FLAGS.num_train_data/FLAGS.batch_size)))



