from tensorflow import keras
from tensorflow.keras import layers

from absl import flags

FLAGS = flags.FLAGS

class LSTMModel(keras.Model):
    """
    1. load w2v embedding -> init embedding layer
    2. bidirectional lstm output
    """

    def __init__(self):
        super(LSTMModel, self).__init__()

        self.embed_layer = layers.Embedding(FLAGS.vocab_size+2, 
                                            FLAGS.vec_dim, 
                                            input_length=FLAGS.max_len,
                                            mask_zero=True)

        self.bidi_lstm = layers.Bidirectional(layers.LSTM(FLAGS.vec_dim, 
                                                          return_sequences=False))

        self.dense_out = layers.Dense(FLAGS.num_class)


    def call(self, inputs):
        x = self.embed_layer(inputs)
        x = self.bidi_lstm(x)
        x = self.dense_out(x)

        return x

